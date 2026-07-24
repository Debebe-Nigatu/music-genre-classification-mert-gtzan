# Music Genre Classification with MERT-v1-330M on GTZAN

Fine-tunes [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M), a self-supervised music
understanding transformer, to classify audio clips into 10 music genres using the
[GTZAN](https://huggingface.co/datasets/marsyas/gtzan) dataset.

**Trained model:** [Atrac/MERT-v1-330M-finetuned-gtzan](https://huggingface.co/Atrac/MERT-v1-330M-finetuned-gtzan)

## What it does

MERT is a pretrained backbone, not a classifier — it only produces audio embeddings. This project
wraps it in a custom classification head and fine-tunes it to recognize genres, while keeping most
of the 330M-parameter backbone frozen so training fits on a single GPU.

## Project structure

1. **Load the dataset** — pulls GTZAN (443 train / 197 validation / 290 test clips) from the Hub.
2. **Build genre labels** — genres are read as plain text and mapped to numeric IDs:
   `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`.
3. **Create a train/test split** — GTZAN only ships one split, so a stratified 90/10 split is carved
   out, giving 398 training clips and 45 test clips.
4. **Extract audio features** — MERT's own feature extractor converts raw audio to model input at a
   24,000 Hz target sampling rate.
5. **Preprocess** — every clip is trimmed or padded to a fixed 30 seconds so batches are consistent.
6. **Build the model** — the MERT backbone is frozen except for its last 4 transformer layers, with a
   linear classification head on top. This leaves 50,395,146 of 315,439,242 parameters (~16%)
   trainable.
7. **Train** — fine-tunes for 10 epochs with gradient accumulation and mixed precision (fp16),
   evaluating and saving the best checkpoint each epoch.
8. **Push to the Hub** — uploads the fine-tuned model so it can be loaded and reused directly from
   `Atrac/MERT-v1-330M-finetuned-gtzan`.
9. **Run inference** — loads the trained model into an audio-classification pipeline to predict the
   genre of a new clip.

## Requirements

Python packages: `transformers`, `datasets`, `evaluate`, `accelerate`, `huggingface_hub`, `gradio`,
`librosa`, `soundfile`, `nnAudio`. A CUDA GPU is required for training (developed on a single T4).

## Usage

Open the notebook and run the cells top to bottom. Log in to the Hugging Face Hub when prompted if
you want the trained model pushed automatically.
