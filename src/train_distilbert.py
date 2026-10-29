"""Fine-tune DistilBERT on the prepared Civil Comments split (run on a GPU, e.g. Colab).

Trains a single-logit head with soft-label BCE on the continuous `target` score,
using the same train.csv / valid.csv / identity_eval.parquet that prepare.py
writes for the baseline, so the only difference from the baseline is the model.

Outputs (in --out-dir):
    model/                     fine-tuned weights + tokenizer
    valid_scores.csv           `score` per valid.csv row, same order
    identity_eval_scores.csv   `score` per identity_eval.parquet row, same order
    train_config.json          hyperparameters, library versions, timings
    train_log.csv              training loss every --log-every steps

Score the outputs with the same evaluation code as the baseline:
    python src/evaluate.py  --scores <out>/valid_scores.csv         --out-dir metrics/distilbert
    python src/bias_eval.py --scores <out>/identity_eval_scores.csv --out-dir metrics/distilbert
"""
from pathlib import Path
import argparse
import json
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Collate:
    """Tokenize a batch with dynamic padding. A class rather than a closure so
    DataLoader workers can pickle it on Windows as well as Linux."""

    def __init__(self, tokenizer, max_len):
        self.tokenizer, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        enc = self.tokenizer([t for t, _ in batch], truncation=True, max_length=self.max_len,
                             padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch], dtype=torch.float32)
        return enc


def make_loader(texts, targets, tokenizer, max_len, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(list(zip(texts, targets)), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=Collate(tokenizer, max_len), generator=generator, num_workers=2)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits.squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed",
                   help="Folder with train.csv, valid.csv, identity_eval.parquet from prepare.py")
    p.add_argument("--out-dir", default="distilbert_out")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    set_seed(a.seed)
    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if use_amp else 'no GPU'})")

    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    eval_df = pd.read_parquet(data_dir / "identity_eval.parquet")
    text =lambda df: df["comment_text"].fillna("").astype(str).tolist()
    print(f"Train {len(train_df):,} | valid {len(valid_df):,} | identity eval {len(eval_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_name, num_labels=1).to(device)

    # Soft labels: the fraction of annotators who rated the comment toxic.
    train_loader = make_loader(text(train_df), train_df["target"].astype(float).tolist(),
                               tokenizer, a.max_len, a.batch_size, shuffle=True, seed=a.seed)
    valid_loader = make_loader(text(valid_df), [0.0] * len(valid_df),
                               tokenizer, a.max_len, a.eval_batch_size, shuffle=False, seed=a.seed)
    eval_loader = make_loader(text(eval_df), [0.0] * len(eval_df),
                              tokenizer, a.max_len, a.eval_batch_size, shuffle=False, seed=a.seed)

    total_steps = a.epochs * len(train_loader)
    no_decay = ("bias", "layer_norm.weight", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": a.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ], lr=a.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(a.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    log_rows, step, started = [], 0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += loss.item()
            if step % a.log_every == 0:
                avg = running / a.log_every
                running = 0.0
                log_rows.append({"step": step, "epoch": epoch, "loss": avg})
                elapsed = time.time() - started
                print(f"epoch {epoch} step {step}/{total_steps} loss {avg:.4f} "
                      f"({elapsed / 60:.1f} min, ~{elapsed / step * (total_steps - step) / 60:.1f} min left)")

        valid_scores = predict(model, valid_loader, device, use_amp)
        auc = roc_auc_score(valid_df["toxic"].astype(int), valid_scores)
        print(f"epoch {epoch} done: validation ROC-AUC {auc:.4f}")
    train_minutes = (time.time() - started) / 60

    print("Scoring identity hold-out...")
    eval_scores = predict(model, eval_loader, device, use_amp)

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    pd.DataFrame({"score": valid_scores}).to_csv(out_dir / "valid_scores.csv", index=False)
    pd.DataFrame({"score": eval_scores}).to_csv(out_dir / "identity_eval_scores.csv", index=False)
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    config = {
        **{k: v for k, v in vars(a).items() if k not in ("data_dir", "out_dir")},
        "loss": "soft-label BCE on target",
        "train_rows": len(train_df),
        "device": torch.cuda.get_device_name(0) if use_amp else "cpu",
        "fp16": use_amp,
        "train_minutes": round(train_minutes, 1),
        "final_valid_roc_auc": float(auc),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved model, scores and config to {out_dir} ({train_minutes:.1f} min training)")


if __name__ == "__main__":
    main()
