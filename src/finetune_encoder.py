"""Fine-tune a small text encoder on training-period press releases (extra ladder step).

- Model: distilroberta-base (82m parameters), sequence classification head.
- Data: press releases of deals announced up to 2015 for training, 2016-2018 for validation
  (time-ordered; early stopping on validation ROC AUC). 2019+ is never seen during training.
- Class weights: inverse class frequency in the loss.
- Input: the first 512 tokens of the press release.
- Device: Apple MPS, batch size 8.
- Watchdog: aborts and records the reason if memory exceeds 20 GB or runtime exceeds 60 minutes.

Outputs (git-ignored): data/interim/ft_scores_<target>.parquet with one score per deal that has
text, and results/finetune_<target>.json with the training log.

Usage: python src/finetune_encoder.py --target label_adverse
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
MODEL = "distilbert/distilroberta-base"
MAX_LEN, BATCH, LR, MAX_EPOCHS, PATIENCE = 512, 8, 2e-5, 6, 2
MAX_SECONDS, MAX_GB = 60 * 60, 20.0
TRAIN_END, VAL_YEARS = 2015, (2016, 2018)


def memory_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9          # bytes on macOS
    mps = torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0
    return max(rss, mps)


class Watchdog(Exception):
    pass


def check(t0: float) -> None:
    if time.time() - t0 > MAX_SECONDS:
        raise Watchdog(f"runtime exceeded {MAX_SECONDS // 60} minutes")
    m = memory_gb()
    if m > MAX_GB:
        raise Watchdog(f"memory {m:.1f} GB exceeded {MAX_GB:.0f} GB")


def batches(enc, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield j, {k: v[j] for k, v in enc.items()}


@torch.no_grad()
def predict(model, enc, idx, device, t0, bs=32):
    model.eval()
    out = np.zeros(len(idx))
    order = np.argsort(enc["attention_mask"][idx].sum(1).numpy())      # length-sorted for speed
    for i in range(0, len(idx), bs):
        j = idx[order[i:i + bs]]
        L = int(enc["attention_mask"][j].sum(1).max())
        logits = model(input_ids=enc["input_ids"][j, :L].to(device), attention_mask=enc["attention_mask"][j, :L].to(device)).logits
        out[order[i:i + bs]] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
        check(t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="label_adverse", choices=["label", "label_adverse"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    deals = pd.read_parquet(INTERIM / "deals.parquet", columns=["deal_id", "year", "label"])
    out_p = INTERIM / "outcomes.parquet"
    if out_p.exists():
        deals = deals.merge(pd.read_parquet(out_p)[["deal_id", "label_adverse"]], on="deal_id", how="left")
    text = pd.read_parquet(INTERIM / "text.parquet", columns=["deal_id", "press_release", "has_text"])
    df = deals.merge(text, on="deal_id")
    df = df[df["has_text"]].reset_index(drop=True)
    y = df[args.target].to_numpy(dtype=int)
    yr = df["year"].to_numpy()
    tr = np.where(yr <= TRAIN_END)[0]
    va = np.where((yr >= VAL_YEARS[0]) & (yr <= VAL_YEARS[1]))[0]
    log = {"target": args.target, "model": MODEL, "device": device, "n_train": int(len(tr)), "pos_train": int(y[tr].sum()),
           "n_val": int(len(va)), "pos_val": int(y[va].sum()), "epochs": [], "status": "running"}

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(df["press_release"].str.slice(0, 6000).tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    enc = {k: enc[k] for k in ("input_ids", "attention_mask")}
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    w_pos = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_pos], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = MAX_EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best_auc, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(args.seed)
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = rng.permutation(tr)
            tot = 0.0
            for step, (j, b) in enumerate(batches(enc, perm, BATCH), 1):
                L = int(b["attention_mask"].sum(1).max())
                logits = model(input_ids=b["input_ids"][:, :L].to(device), attention_mask=b["attention_mask"][:, :L].to(device)).logits
                loss = loss_fn(logits.float(), torch.tensor(y[j], device=device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                tot += float(loss)
                if step % 20 == 0:
                    check(t0)
            pv = predict(model, enc, va, device, t0)
            auc = float(roc_auc_score(y[va], pv)) if 0 < y[va].sum() < len(va) else float("nan")
            ap_ = float(average_precision_score(y[va], pv))
            log["epochs"].append({"epoch": epoch, "train_loss": tot / step, "val_auc": auc, "val_pr_auc": ap_,
                                  "minutes": (time.time() - t0) / 60, "memory_gb": memory_gb()})
            print(json.dumps(log["epochs"][-1]), flush=True)
            if auc > best_auc + 1e-4:
                best_auc, bad = auc, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        log["status"] = "completed"
    except Watchdog as e:
        log["status"] = f"aborted by watchdog: {e}"
        print(log["status"], flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        try:
            scores = predict(model, enc, np.arange(len(df)), device, time.time())   # scoring gets its own clock
            split = np.where(yr <= TRAIN_END, "train", np.where(yr <= VAL_YEARS[1], "val", "test"))
            pd.DataFrame({"deal_id": df["deal_id"], "ft_score": scores, "ft_split": split}).to_parquet(
                INTERIM / f"ft_scores_{args.target}.parquet", index=False)
            log["best_val_auc"] = best_auc
        except Watchdog as e:
            log["status"] += f"; scoring aborted: {e}"
    log["total_minutes"] = (time.time() - t0) / 60
    log["peak_memory_gb"] = memory_gb()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/finetune_{args.target}.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "epochs"}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
