# Egyptian Arabic ↔ English Transformer (from scratch)

A Transformer model built from scratch in PyTorch (following the original "Attention Is All You Need" architecture) to translate between Egyptian Arabic and English.

> **Note:** This project is for learning purposes — the goal is to understand how a Transformer works end-to-end (tokenization, attention, training loop, evaluation) rather than to produce a production-quality translator.

## Project structure

```
.
├── config.py     # all training settings (batch size, seq_len, learning rate, etc.)
├── model.py      # the Transformer architecture (encoder, decoder, attention, etc.)
├── dataset.py    # PyTorch Dataset class that prepares text for the model
├── train.py      # loads data, builds tokenizers, trains the model, saves checkpoints
└── weights/      # saved model checkpoints (.pt files), created automatically
```

## Requirements

```bash
pip install torch datasets tokenizers tensorboard
```

## Dataset

Uses the [`IbrahimAmin/arz-en-parallel-corpus`](https://huggingface.co/datasets/IbrahimAmin/arz-en-parallel-corpus) dataset from Hugging Face — 25,000 training pairs and 1,851 test pairs of short, conversational Egyptian Arabic / English sentences, with columns `arz` and `en`.

```python
from datasets import load_dataset
ds_raw = load_dataset("IbrahimAmin/arz-en-parallel-corpus")
```

The dataset's column names must match `lang_src` / `lang_tgt` in `config.py`.

## Configuration

Key settings in `config.py`:

```python
{
    "batch_size": 32,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 80,
    "d_model": 512,
    "lang_src": "arz",
    "lang_tgt": "en",
}
```

`seq_len` and `batch_size` should be re-checked whenever the dataset changes — see "Checking seq_len" below.

## How training works

1. `get_ds()` loads the dataset and splits out train/validation sets.
2. A BPE tokenizer is trained separately for each language (only on the **training** split, to avoid data leakage) and saved to disk (`tokenizer_arz.json`, `tokenizer_en.json`).
3. Sentences longer than `seq_len` are filtered out.
4. `BillingualDataset` converts each sentence pair into padded token ID tensors (`[SOS]` + tokens + `[EOS]` + padding).
5. The model trains with cross-entropy loss (ignoring padding tokens), saving a checkpoint every 500 steps and at the end of every epoch.

Run training:
```python
from config import get_config
from train import train_model

config = get_config()
train_model(config)
```

## Checking seq_len before training

Don't guess — measure actual token lengths with the real tokenizer:

```python
import numpy as np

src_lens = [len(tokenizer_src.encode(item["arz"]).ids) for item in train_ds]
tgt_lens = [len(tokenizer_tgt.encode(item["en"]).ids) for item in train_ds]

print("99th percentile src:", np.percentile(src_lens, 99))
print("99th percentile tgt:", np.percentile(tgt_lens, 99))
```

Pick `seq_len` a bit above the higher 99th percentile, then filter out the rare longer outliers rather than sizing everything around them (this keeps training fast).

## Resuming from a checkpoint

Set `preload` in `config.py` to the epoch/step suffix of a saved file:
```python
"preload": "19"              # loads weights/tmodel_19.pt
# or
"preload": "14_step11000"    # loads a mid-epoch checkpoint
```

## Results so far (20 epochs)

- **Final training loss:** ~2.0
- **Validation loss:** ~4.58

The gap between these two numbers shows the model is **overfitting** — it performs well on short, common, frequently-seen sentence patterns, but struggles on longer or less common sentences that require real generalization. See translation samples for examples of both.

### What to try next
- Track validation loss every epoch (not just once at the end) to find the best checkpoint, rather than always using the last one
- Try beam search decoding instead of greedy decoding for better output quality
- Experiment with higher dropout to reduce overfitting
- A larger dataset would likely help generalization, since 25,000 pairs is small for training a Transformer fully from scratch

## Running inference

```python
from translate import translate

print(translate("إزيك عامل ايه؟"))
```

This loads a saved checkpoint, encodes the input sentence, and greedily decodes an English translation one token at a time.