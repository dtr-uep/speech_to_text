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

# Utils
from utils import *

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

parser.add_argument("--train_ratio", type=float, help="First ratio of dataset for training (Default: 0.5)", default=0.5)
parser.add_argument("--val_ratio", type=float, help="Next ratio of dataset for validation (Default: 0.2)", default=0.2)
parser.add_argument("--batch_ratio", type=float, help="Proportion of batches per epoch for trainning (Default: 1.0)", default=1.0)

parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path (Default: None)", default="None")
parser.add_argument("--normalizer_path", type=str, help="Normalizer (Default: normalizer1.pkl)", default="normalizer1.pkl")

parser.add_argument("--cuda", type=int, help="Device for Pytorch 1 or 0 (Default: True)", default=1)
parser.add_argument("--seed", type=int, help="Random seed (Default: 42)", default=42)
parser.add_argument("--final", type=int, help="Final model if 1 (Default: 0)", default=0)



args = parser.parse_args()
MOD_NO = args.model_idx

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

TRAIN_RATIO = args.train_ratio
VALIDATION_RATIO = args.val_ratio
BATCH_RATIO = args.batch_ratio

TOK_PATH = args.tokenizer_path
if TOK_PATH == "None": TOK_PATH = None
NORM_PATH = args.normalizer_path

SEED = args.seed


print("Trainning parameters")
print(f"Model no                    : {MOD_NO}")

print(f"Learning rate               : {LEARNING_RATE}")
print(f"Batch size                  : {BATCH_SIZE}")
print(f"Num epochs                  : {EPOCHS}")

print(f"Trainset ratio              : {TRAIN_RATIO}")
print(f"Valset ratio                : {VALIDATION_RATIO}")
print(f"Ratio of batches per epoch  : {BATCH_RATIO}")
print(f"Random seed                 : {SEED}")

print(f"Tokenizer                   : {TOK_PATH}")
print(f"Normalizer path             : {NORM_PATH}")
print()

# Load data
print("Loading data...")
train_set = load_dataset(
    "pollen-robotics/speech-commands-v0.02",
    split="train"
)

val_set = load_dataset(
    "pollen-robotics/speech-commands-v0.02",
    split="validation"
)

print("Train size:", len(train_set))
print("Validation size:", len(val_set))
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
# print("Train-test split...")
# test_ratio = 1 - TRAIN_RATIO - VALIDATION_RATIO

# train_end = int(len(dataset) * TRAIN_RATIO)
# validation_end = train_end + int(len(dataset) * VALIDATION_RATIO)

# train_set = dataset[:train_end]
# val_set = dataset[train_end:validation_end]
# # test_set = dataset[validation_end:]
# print("Train size     : ", train_end)
# print("Validation size: ", validation_end - train_end)
# print(f"Test set index : {validation_end}-{len(dataset)}")
# print()
# del dataset


# Data preparation
def get_input_and_target(dataset):
    input_audios = dataset["audio"]
    target_sentences = dataset["label"]

    return input_audios, target_sentences


class SpeechDataset(Dataset):

    def __init__(
        self,
        input_audios,
        target_sentences,
        tokenizer,
        mfcc_transform,
        device,
        eos_id,
    ):

        self.input_audios = input_audios
        self.target_sentences = target_sentences

        self.tokenizer = tokenizer
        self.mfcc_transform = mfcc_transform

        self.device = device
        self.eos_id = eos_id

    def __len__(self):
        return len(self.input_audios)

    def __getitem__(self, index):

        try:

            input_audio = torch.tensor(
                self.input_audios[index]
            )

        except:
            input_audio = torch.tensor(
                self.input_audios[index]["array"]
            )
        mfccs = self.mfcc_transform(
            input_audio
        ).transpose(0, 1)


        token_ids = self.tokenizer.tokenize(
            self.target_sentences[index]
        )

        # Add EOS to the actual target
        token_ids = token_ids.tolist()
        token_ids.append(self.eos_id)
        token_ids = token_ids + [self.eos_id]

        tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        return mfccs, tokens


# Collate function
def collate_fn(batch):

    audios, targets = zip(*batch)

    audios = pad_sequence(
        audios,
        batch_first=True,
        padding_value=0.0,
    )

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_id,
    )

    return audios, targets


# Prepare
# Mel-frequency ceptral coeffiecients
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "n_mels": 40
}
    )

# Tokenizer
special_tokens = ['<PAD>', '<BOS>', '<EOS>']
if TOK_PATH:
    print("Getting tokenizer...")
    tokenizer = TokenizerBPE01()
    tokenizer.loadFromJSON(TOK_PATH)
else:
    print("Initiating tokenizer...")
    tokenizer = TokenizerBPE01()
    TOK_PATH = "vocab.json"
    tokenizer.saveToJSON(TOK_PATH)

tokenizer.update(special_tokens)
pad_id = tokenizer.vocab["<PAD>"]
bos_id = tokenizer.vocab["<BOS>"]
eos_id = tokenizer.vocab["<EOS>"]
print()
print("Tokenizer information")
print("----------------------------------------")
print("Vocabulary size:", len(tokenizer.vocab))
print("PAD ID         :", pad_id)
print("BOS ID         :", bos_id)
print("EOS ID         :", eos_id)
print("----------------------------------------")
print()


# Make train dataset
train_set = train_set.filter(
    lambda x: len(x["audio"]["array"]) <= 32000
)

val_set = val_set.filter(
    lambda x: len(x["audio"]["array"]) <= 32000
)

input_audios, target_sentences = get_input_and_target(train_set)

train_speech_dataset = SpeechDataset(
    input_audios=input_audios,
    target_sentences=target_sentences,
    tokenizer=tokenizer,
    mfcc_transform=mfcc_transform,
    device=device,
    eos_id=eos_id,
)


val_audios, val_targets = get_input_and_target(
    val_set
)

val_speech_dataset = SpeechDataset(
    input_audios=val_audios,
    target_sentences=val_targets,
    tokenizer=tokenizer,
    mfcc_transform=mfcc_transform,
    device=device,
    eos_id=eos_id,
)

train_loader = DataLoader(
    train_speech_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_speech_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
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

MOD_PATH = os.path.join(model_dir, f"model{model_no}.pt")
# For first time continute trainning
if args.final == 1:
    MOD_PATH = Path("trained_models") / TOK_PATH.split(".")[0] / model["model_type"] / f"model{model_no}.pt"

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
print()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Loss function using torch
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_id,
)


# Trainning
print("Trainning...")

for epoch in range(1, EPOCHS+1):
    print(f"[Epoch: {epoch}/{EPOCHS}]")
    # Train on random a small proportion of trainning data
    max_batches = int(len(train_loader) * BATCH_RATIO)
    # Train
    model.train()
    full_set_loss = []
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        bos = torch.full(
            (y.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=device,
        )

        decoder_input = torch.cat(
            [
                bos,
                y[:, :-1],
            ],
            dim=1,
        )


        optimizer.zero_grad()


        invalid = decoder_input[
            (decoder_input < 0) |
            (decoder_input >= model.embedding.num_embeddings)
        ]


        logits = model(normalizer(x), decoder_input)
        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            y.reshape(-1),
        )

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
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            bos = torch.full(
                (y.size(0), 1),
                bos_id,
                dtype=torch.long,
                device=device,
            )

            decoder_input = torch.cat(
                [
                    bos,
                    y[:, :-1],
                ],
                dim=1,
            )
            logits = model(normalizer(x), decoder_input)
            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                y.reshape(-1),
            )
            full_set_loss.append(loss.item())

        # Check single output
        x_example, y_example = next(
        iter(val_loader)
    )

    x_example = x_example.to(device)

    # Prevent generation from becoming
    # unreasonably long.
    max_decode_length = (
        y_example.size(1) + 20
    )

    predicted = greedy_decode(
        model=model,
        x=x_example[:1],
        normalizer=normalizer,
        bos_id=bos_id,
        eos_id=eos_id,
        max_length=max_decode_length,
    )

    predicted_ids = predicted[0].tolist()

    # Remove EOS and PAD-like values from
    # the sequence before converting to text.

    if eos_id in predicted_ids:

        predicted_ids = predicted_ids[
            :predicted_ids.index(eos_id)
        ]

    try:

        predicted_text = tokenizer.tokenToWords(
            predicted_ids
        )

    except Exception as e:

        predicted_text = (
            f"[token conversion failed: {e}]"
        )

    actual_ids = y_example[0].tolist()

    if eos_id in actual_ids:

        actual_ids = actual_ids[
            :actual_ids.index(eos_id)
        ]

    try:

        actual_text = tokenizer.tokenToWords(
            actual_ids
        )

    except Exception as e:

        actual_text = (
            f"[token conversion failed: {e}]"
        )

    print()
    print("Example prediction")
    print("----------------------------------------")
    print("Target    :", actual_text)
    print("Prediction:", predicted_text)
    print("----------------------------------------")
    print()


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
    if epoch % 5 == 0:
        torch.save(model.state_dict(), MOD_PATH)
        print("Saved B checkpoint B")

    

