# Music Genre Classification with MERT-v1-330M on GTZAN

Fine-tunes [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M) for music genre classification
on [GTZAN](https://huggingface.co/datasets/marsyas/gtzan).

**Model on the Hub:** [Atrac/MERT-v1-330M-finetuned-gtzan](https://huggingface.co/Atrac/MERT-v1-330M-finetuned-gtzan)

## Setup

```bash
pip install -q --upgrade transformers datasets evaluate accelerate huggingface_hub gradio librosa soundfile nnAudio
```

## 1. Load dataset

```python
from datasets import load_dataset

gtzan = load_dataset("confit/gtzan-parquet")
print(gtzan)
```
```
DatasetDict({
    train:      Dataset({features: ['audio', 'genre', 'label'], num_rows: 443})
    validation: Dataset({features: ['audio', 'genre', 'label'], num_rows: 197})
    test:       Dataset({features: ['audio', 'genre', 'label'], num_rows: 290})
})
```

## 2. Genre labels

`genre` is a plain string column, so `id2label`/`label2id` are built manually:

```python
unique_genres = sorted(set(gtzan["train"]["genre"]))
label2id = {genre: i for i, genre in enumerate(unique_genres)}
id2label = {i: genre for i, genre in enumerate(unique_genres)}
```
```
Genres: ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
```

## 3. Train/test split

GTZAN ships a single split, so a stratified 90/10 split is carved out:

```python
from datasets import ClassLabel

gtzan["train"] = gtzan["train"].cast_column("genre", ClassLabel(names=unique_genres))
gtzan = gtzan["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="genre")
```
```
DatasetDict({
    train: Dataset({features: ['audio', 'genre', 'label'], num_rows: 398})
    test:  Dataset({features: ['audio', 'genre', 'label'], num_rows: 45})
})
```

## 4. Feature extraction

```python
from transformers import AutoFeatureExtractor

model_id = "m-a-p/MERT-v1-330M"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, trust_remote_code=True, do_normalize=True, return_attention_mask=True
)
sampling_rate = feature_extractor.sampling_rate
```
```
Target sampling rate: 24000
```

## 5. Preprocessing

Clips are capped at 30s and padded to a fixed length so batches stack cleanly:

```python
max_duration = 30.0

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    inputs["label"] = examples["genre"]
    return inputs

gtzan_encoded = gtzan.map(preprocess_function, batched=True, batch_size=100, num_proc=1,
                          remove_columns=[c for c in gtzan["train"].column_names
                                          if c not in ("input_values", "attention_mask", "label")])
gtzan_encoded.set_format(type="torch", columns=["input_values", "attention_mask", "label"])
```

## 6. Model

MERT is a self-supervised backbone, so it's wrapped in a classification head. The backbone is frozen
except for the last `num_unfrozen_layers`:

```python
class MERTForAudioClassification(nn.Module):
    def __init__(self, model_id, num_labels, id2label, label2id, num_unfrozen_layers=4):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.mert = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        for param in self.mert.parameters():
            param.requires_grad = False
        for layer in self.mert.encoder.layers[-num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        outputs = self.mert(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # downsample raw-waveform attention mask to frame-level before pooling
        feature_attention_mask = self.mert._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask
        )
        mask = feature_attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled_output = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        logits = self.classifier(pooled_output)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
```
```
Using device: cuda
Trainable parameters: 50,395,146 / 315,439,242
```

## 7. Training

```python
training_args = TrainingArguments(
    output_dir="MERT-v1-330M-finetuned-gtzan",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    warmup_steps=100,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=gtzan_encoded["train"], eval_dataset=gtzan_encoded["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
```

## 8. Push to Hub

```python
trainer.push_to_hub(
    dataset_tags=["marsyas/gtzan"], dataset="GTZAN",
    model_name="MERT-v1-330M-finetuned-gtzan",
    finetuned_from=model_id, tasks="audio-classification",
)
```
```
CommitInfo(commit_url='https://huggingface.co/Atrac/MERT-v1-330M-finetuned-gtzan/commit/aff6b257...',
           commit_message='End of training', ...)
```

## 9. Inference

```python
from transformers import pipeline

trainer.model.device = next(trainer.model.parameters()).device
pipe = pipeline("audio-classification", model=trainer.model, feature_extractor=feature_extractor)
pipe(audio_array_or_path)
```

> Note: `MERTForAudioClassification` is a custom class, so the pipeline logs a warning that it isn't
> an officially supported `audio-classification` architecture — it still runs fine.
