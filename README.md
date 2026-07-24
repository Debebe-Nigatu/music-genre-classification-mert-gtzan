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

1. **Environment check** — training was run on a single NVIDIA T4 GPU:
   ```
   NVIDIA-SMI 580.82.07   Driver Version: 580.82.07   CUDA Version: 13.0
   ```

2. **Load the dataset** — pulls GTZAN from the Hub (`confit/gtzan-parquet`):
   ```
   DatasetDict({
       train:      Dataset({features: ['audio', 'genre', 'label'], num_rows: 443})
       validation: Dataset({features: ['audio', 'genre', 'label'], num_rows: 197})
       test:       Dataset({features: ['audio', 'genre', 'label'], num_rows: 290})
   })
   ```

3. **Build genre labels** — genres are read as plain text and mapped to numeric IDs:
   ```
   Genres: ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
   ```

4. **Create a train/test split** — GTZAN only ships one split, so a stratified 90/10 split is carved
   out of the training data:
   ```
   DatasetDict({
       train: Dataset({features: ['audio', 'genre', 'label'], num_rows: 398})
       test:  Dataset({features: ['audio', 'genre', 'label'], num_rows: 45})
   })
   ```

5. **Extract audio features** — MERT's own feature extractor converts raw audio to model input:
   ```
   Target sampling rate: 24000
   ```

6. **Preprocess** — every clip is trimmed or padded to a fixed 30 seconds so batches are consistent:
   ```
   DatasetDict({
       train: Dataset({features: ['label', 'input_values', 'attention_mask'], num_rows: 398})
       test:  Dataset({features: ['label', 'input_values', 'attention_mask'], num_rows: 45})
   })
   ```

7. **Build the model** — the MERT backbone (loaded with `trust_remote_code=True`, which pulls its
   custom modeling code from the Hub) is frozen except for its last 4 transformer layers, with a
   linear classification head on top:
   ```
   Using device: cuda
   Trainable parameters: 50,395,146 / 315,439,242
   ```
   Only about 16% of the backbone's parameters are actually updated during training.

8. **Train** — fine-tunes for 10 epochs with gradient accumulation and mixed precision (fp16),
   evaluating and saving the best checkpoint each epoch based on accuracy.

9. **Push to the Hub** — uploads the fine-tuned model and training metadata:
   ```
   CommitInfo(
       commit_url='https://huggingface.co/Atrac/MERT-v1-330M-finetuned-gtzan/commit/aff6b257f9b1cf05eff7da3cb87e27433fef7d7d',
       commit_message='End of training',
       repo_url=RepoUrl('https://huggingface.co/Atrac/MERT-v1-330M-finetuned-gtzan')
   )
   ```

10. **Run inference** — loads the trained model into an audio-classification pipeline:
    ```
    The model 'MERTForAudioClassification' is not supported for audio-classification.
    Supported models are ['ASTForAudioClassification', 'HubertForSequenceClassification', ...]
    ```
    This warning is expected — since `MERTForAudioClassification` is a custom class rather than one
    of `transformers`' built-in architectures, the pipeline just can't confirm it's officially
    supported. It still runs correctly.

## Requirements

Python packages: `transformers`, `datasets`, `evaluate`, `accelerate`, `huggingface_hub`, `gradio`,
`librosa`, `soundfile`, `nnAudio`. A CUDA GPU is required for training (developed on a single T4).

## Usage

Open the notebook and run the cells top to bottom. Log in to the Hugging Face Hub when prompted if
you want the trained model pushed automatically.
