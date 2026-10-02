import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from tokenizer import TokenizerBPE01
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import pickle as pkl
import json

# CTC
from ctc_utils import *

# Models
from models import models

print("Training set up...")
# Reallocate hugging face cache
import os
from pathlib import Path
from datasets import load_dataset
from datasets import config
print("Huggingface cache location:", config.HF_DATASETS_CACHE)


# Train hyperaparmeters
import argparse

parser = argparse.ArgumentParser(description="Trainning hyperparameters")
parser.add_argument("--model_idx", type=int, help="Model index in models list")

parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate (Default: 0.004)", default=1e-4)
parser.add_argument("-b", "--batch_size", type=int, help="Batch size (Default: 32)", default=32)
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs (Default: 10", default=10)

parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path (Default: None)", default="None")
parser.add_argument("--normalizer_path", type=str, help="Normalizer (Default: normalizer1.pkl)", default="normalizer1.pkl")

parser.add_argument("--cuda", type=int, help="Device for Pytorch 1 or 0 (Default: True)", default=1)
parser.add_argument("--seed", type=int, help="Random seed (Default: 42)", default=42)
parser.add_argument("--final", type=int, help="Final model if 1 (Default: 0)", default=1)



args = parser.parse_args()
MOD_NO = args.model_idx

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

TOK_PATH = args.tokenizer_path
if TOK_PATH == "None": TOK_PATH = None
NORM_PATH = args.normalizer_path

SEED = args.seed

print("Trainning parameters")
print(f"Model no                    : {MOD_NO}")

print(f"Learning rate               : {LEARNING_RATE}")
print(f"Batch size                  : {BATCH_SIZE}")
print(f"Num epochs                  : {EPOCHS}")

print(f"Random seed                 : {SEED}")

print(f"Tokenizer                   : {TOK_PATH}")
print(f"Normalizer path             : {NORM_PATH}")
print()

# Load data
print("Loading data...")
# Full train with full set
train_set = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="train.100"
)

val_set = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="validation"
)
print()


# Device
if args.cuda == 1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
torch.manual_seed(SEED)
print("Device:", device)
print()

# train, validation, test split
print("Train-test split...")
# test_set = dataset[validation_end:]
print("Train size     : ", len(train_set))
print("Validation size: ", len(val_set))
print()

# Data preparation
def get_input_and_target(dataset):
    # Input output
    input_audios = dataset['audio']
    target_sentences = dataset['text']

    return input_audios, target_sentences


class SpeechDataset(Dataset):

    def __init__(self, input_audios, target_sentences, tokenizer, mfcc_transform, device):
        self.input_audios = input_audios
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer
        self.mfcc_transform = mfcc_transform
        self.device = device

    def __len__(self):
        return len(self.input_audios)

    def __getitem__(self, index):
        # Preprocess input
        try:
            input_audio = torch.tensor(self.input_audios[index])
        except:
            # For codec
            input_audio = torch.tensor(self.input_audios[index]['array'])

        mfccs = self.mfcc_transform(input_audio).to(device).transpose(0, 1)

        # Tokenize output
        tokens = torch.tensor(self.tokenizer.tokenize(self.target_sentences[index]), dtype=torch.int32).to(device)

        # Original length
        seq_len = len(mfccs)
        targ_len = len(tokens)

        return mfccs, tokens, seq_len, targ_len


# Collate function
def collate_fn(batch):
    audios, targets, seq_lens, targ_lens = zip(*batch)
    
    audios = pad_sequence(
        audios,
        batch_first=True,
        padding_value=0.0  # Padding for MFCCs
    )

    targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=pad_id
        )

    return audios, targets, seq_lens, targ_lens


# Prepare
# Mel-frequency ceptral coeffiecients
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
        "n_fft": 1200,
        "win_length": 1200, # 75ms timeframe
        "hop_length": 600,  
        "n_mels": 40
    }
    )

# Tokenizer
add_blank = ['<Blank>']
if TOK_PATH:
    print("Getting tokenizer...")
    tokenizer = TokenizerBPE01()
    tokenizer.loadFromJSON(TOK_PATH)
else:
    print("Initiating tokenizer...")
    tokenizer = TokenizerBPE01()
    TOK_PATH = "vocab.json"
    tokenizer.saveToJSON(TOK_PATH)

tokenizer.update(add_blank)
blank_id = tokenizer.vocab['<Blank>']
pad_id = len(tokenizer.vocab)
print("Blank ID:", blank_id)
print("Pad ID:", pad_id)
print()


# Make train dataset
input_audios, target_sentences = get_input_and_target(train_set)

train_speech_dataset = SpeechDataset(
    input_audios=input_audios,
    target_sentences=target_sentences,
    tokenizer=tokenizer,
    mfcc_transform=mfcc_transform,
    device=device
)

train_loader = DataLoader(
    train_speech_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_audios, val_targets = get_input_and_target(val_set)

val_speech_dataset = SpeechDataset(
    input_audios=val_audios,
    target_sentences=val_targets,
    tokenizer=tokenizer,
    mfcc_transform=mfcc_transform,
    device=device
)

del train_set
del val_set

# Normalizer
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
    
if os.path.exists(NORM_PATH):
    print("Getting normalizer...")
    with open(NORM_PATH, 'rb') as f:
        normalizer = pkl.load(f)
        normalizer.to(device)
        print(normalizer.mean.device)
        print("Loaded successfully.")
else:
    print(f"Initiating normalizer")
    X = train_speech_dataset[0][0]
    mean = X.mean(dim=0).to(device)
    std = X.std(dim=0).to(device)
    normalizer = Normalize(mean, std)
    normalizer.to(device)

    with open(NORM_PATH, 'wb') as f:
        pkl.dump(normalizer, f)
        print("Saved normalizer")
print()

# Model setup
print("Setting up model...")
with open('models_config.json', 'r') as file:
    cfg = json.load(file)

model = cfg[str(MOD_NO)]
if args.final == 0:
    model_dir = Path("trained_models") / TOK_PATH.split(".")[0] / model["model_type"]
elif args.final == 1:
    model_dir = Path("trained_models") / "final" / TOK_PATH.split(".")[0] / model["model_type"]

print("Model info:")
print(model)
print()

model_no = model["model_no"]
model_type = model["model_type"]
model_parameters = model["model_parameters"]

# Fully trainned from start
MOD_PATH = os.path.join(model_dir, f"model{model_no}.pt")
# For first time continute trainning
# if args.final == 1:
#     MOD_PATH = Path("trained_models") / TOK_PATH.split(".")[0] / model["model_type"] / f"model{model_no}.pt"

model = models[model_type](
    input_size=13,
    output_size=len(tokenizer.vocab),
    **model_parameters
).to(device)

# Creating folder if not exist
model_dir.mkdir(parents=True, exist_ok=True)

# Results saving path
RES_PATH = os.path.join(model_dir, f"results_model{model_no}.csv")

if os.path.exists(RES_PATH):
    with open(RES_PATH, 'r', encoding='utf-8') as f:
        current_epoch = sum(1 for line in f) - 1
    print("Results path found, current epoch:", current_epoch)
else:
    pd.DataFrame(data={
        "Epoch":[],
        "Train":[],
        "Validation":[]
    }).to_csv(RES_PATH, index=False)
    current_epoch = 0
    print("File not exist. New file created.")
print()

# Model saving path


if MOD_PATH:
    try:
        model.load_state_dict(
            torch.load(MOD_PATH, map_location=device)
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(e)
        print(f"New model initiated and saved at: {MOD_PATH}")

print("Model directory  :", model_dir)
print("Model path       :", MOD_PATH)
print("Result path      :", RES_PATH)
print()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Loss function using torch
criterion = torch.nn.CTCLoss(
    blank=blank_id,
    reduction="mean",
    zero_infinity=True,
)


# Trainning
print("Trainning...")

for epoch in range(1, EPOCHS+1):
    print(f"[Epoch: {epoch}/{EPOCHS}]")
    # Train on random a small proportion of trainning data
    # Train
    model.train()
    full_set_loss = []
    for i, batch in enumerate(train_loader):

        x, y, seq_len, targ_len = batch
        optimizer.zero_grad()
        logits = model(normalizer(x))
        log_probs = torch.log_softmax(logits, dim = -1).transpose(0, 1)
        loss = criterion(log_probs=log_probs, targets=y, input_lengths=torch.tensor(seq_len, dtype=int, device=device), target_lengths=torch.tensor(targ_len, dtype=int, device=device))
        #loss = ctc_loss(logits, y, seq_len, blank_id, pad_id, device)
        loss.backward()
        # # Check grad
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad:\n{param.grad}\n")
        #     else:
        #         print(f"{name} grad: None (no gradient computed)\n")
        optimizer.step()
        full_set_loss.append(loss.item())
        #print(f"Batch {i} loss: {loss.item()}")
       
    train_loss = float(np.mean(full_set_loss))
    print(f"Train loss     : {train_loss}")

    # Evaluate
    model.eval()
    full_set_loss = []
    with torch.no_grad():
        for x, y, seq_len, targ_len in val_speech_dataset:
            logits = model(normalizer(x).unsqueeze(0))
            log_probs = torch.log_softmax(logits, dim = -1).transpose(0, 1)
            loss = criterion(log_probs, y.unsqueeze(0), torch.tensor([seq_len,], dtype=int, device=device), torch.tensor([targ_len], dtype=int, device=device))
            full_set_loss.append(loss.item())
        # Check single output
        with torch.no_grad():
            output, prob = beam_search(logits, 10, blank_id)
            print(prob)
        print("True:", tokenizer.tokenToWords(y.cpu().numpy().tolist()))
        print("Predicted:", tokenizer.tokenToWords(collpase(output[0], blank_id)))

    val_loss = float(np.mean(full_set_loss))
    print(f"Validation loss: {val_loss}")
    # Save results
    pd.DataFrame({
        "Epoch": [current_epoch + epoch],
        "Train": [train_loss],
        "Validation": [val_loss]
    }).to_csv(RES_PATH, mode='a', index=False, header=False)
    print("Results saved.")
    print()

    # Save checkpoint B
    if epoch % 3 == 0:
        torch.save(model.state_dict(), MOD_PATH)
        print("Saved B checkpoint B")

torch.save(model.state_dict(), MOD_PATH)
print("Saved B checkpoint B")

    

