\# Speech Recognition Experiments



An exploration of automatic speech recognition using PyTorch,

covering acoustic feature extraction, CTC-based recognition, sequence-to-sequence

encoder-decoder models, decoding algorithms, and architecture comparison.



\## Project Scope



This repository contains two main experimental pipelines:



\### 1. CTC Speech Recognition (`ctc\_01/`)



Pipeline:



Audio

→ MFCC

→ Sequence Model

→ Token Logits

→ CTC Loss

→ CTC Decoding + Beam Search

→ Text



Architectures evaluated:

\- RNN

\- GRU

\- LSTM

\- Bidirectional GRU/LSTM

\- Deep BiLSTM

\- CNN-BiLSTM

\- ...



Dataset:

LibriSpeech ASR, an English speech corpus derived from read audiobooks.

The experiment uses the clean configuration, with:



* train.100 for model training (\~100 hours of speech)
* validation for model selection and validation
* test for final evaluation



Final test results:

\- Mean WER: 101.3%

\- Mean CER: 86.8%
Testing CTC algorithm, capture some sound representations of character combinations, suggesting phonemes prediction modelling.



\### 2. Encoder-Decoder Speech Recognition (`enc\_dec01/`)



Pipeline:



Audio

→ MFCC

→ Encoder

→ Latent representation

→ Autoregressive decoder

→ Greedy Inferrence

→ Text



Experiments include:

\- RNN encoder-decoder

\- GRU/LSTM variants

\- Attention

\- Transformer-based architectures

\- Teacher forcing



Dataset:

Speech Commands v0.02, using the Pollen Robotics distribution.



This experiment focuses on limited-vocabulary short-utterance recognition,

rather than unrestricted continuous-speech ASR. Larger speech dataset like 

LibriSpeech ASR has longer sentences in each data point, causing either vanishing 

gradient for classical encoder or too large embedding size that need more 

handling and designs.





Final test results:

\- Mean WER: 19.7%

\- Mean CER: 19.6%





\## Feature Extraction



Audio is converted to MFCC features before being passed to the acoustic model.



Example configuration:



\- Sampling rate: 16 kHz

\- MFCC coefficients: 13

\- Window: 25 ms

\- Hop: 10 ms

\- Mel filters: 40



\## Evaluation



Models are evaluated using:



\- Word Error Rate (WER)

\- Character Error Rate (CER)

\- Training loss

\- Validation loss



Experimental CSV results are retained in the repository so that model

configurations and training behavior can be compared.



