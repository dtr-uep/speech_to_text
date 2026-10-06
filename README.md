\# Speech Recognition Experiments



An exploration of automatic speech recognition using PyTorch,

covering acoustic feature extraction, CTC-based recognition, sequence-to-sequence

encoder-decoder models, decoding algorithms, and architecture comparison.


Model demo is at: https://datsblog.xyz/interactives/speech-to-text



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

**Train — 28,539 samples**

\- WER: Mean 36.9% | Median 36.6% | Std 14.5%

\- WER P25/P75: 27.3% / 46.2% | Min/Max: 0.0% / 117.5%

\- CER: Mean 11.3% | Median 10.6% | Std 5.5%

\- CER P25/P75: 7.5% / 14.4% | Min/Max: 0.0% / 50.0%


**Validation — 2,703 samples**

\- WER: Mean 57.9% | Median 57.1% | Std 19.9%

\- WER P25/P75: 45.5% / 70.0% | Min/Max: 0.0% / 200.0%

\- CER: Mean 22.0% | Median 20.7% | Std 9.9%

\- CER P25/P75: 15.3% / 27.5% | Min/Max: 0.0% / 77.3%


**Test — 2,620 samples**

\- WER: Mean 56.5% | Median 55.6% | Std 20.3%

\- WER P25/P75: 43.3% / 68.5% | Min/Max: 0.0% / 200.0%

\- CER: Mean 21.6% | Median 20.4% | Std 9.8%

\- CER P25/P75: 14.8% / 26.9% | Min/Max: 0.0% / 71.4%



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

**Train — 84,849 samples**

\- WER: Mean 17.6% | Median 0.0% | Std 38.1%

\- WER P25/P75: 0.0% / 0.0% | Min/Max: 0.0% / 100.0%

\- CER: Mean 17.5% | Median 0.0% | Std 41.9%

\- CER P25/P75: 0.0% / 0.0% | Min/Max: 0.0% / 400.0%


**Validation — 9,981 samples**

\- WER: Mean 18.2% | Median 0.0% | Std 38.6%

\- WER P25/P75: 0.0% / 0.0% | Min/Max: 0.0% / 100.0%

\- CER: Mean 18.4% | Median 0.0% | Std 43.6%

\- CER P25/P75: 0.0% / 0.0% | Min/Max: 0.0% / 400.0%


**Test — 11,005 samples**
\- WER: Mean 19.7% | Median 0.0% | Std 39.8%

\- WER P25/P75: 0.0% / 0.0% | Min/Max: 0.0% / 100.0%

\- CER: Mean 19.6% | Median 0.0% | Std 44.1%

\- CER P25/P75: 0.0% / 0.0% | Min/Max: 0.0% / 400.0%





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



