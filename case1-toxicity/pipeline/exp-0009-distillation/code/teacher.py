"""exp-0009 stage 1: train the teacher (KcELECTRA) and relabel the unlabelled
pool.

Rigour requirements:
  - the teacher sees **our train split only**; val and test are never used, so
    nothing can leak.
  - the pool is external text only. **The original labels are discarded
    immediately**, and any sentence overlapping our val/test is removed.
  - the leakage check runs in code and aborts the script on failure.

Output: artifacts/pseudo_labels.csv (text, teacher_prob, orig_label — the last
kept only for the control arm).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from common import f1_binary, load_split  # noqa: E402
from datasets_survey import load_all  # noqa: E402
from kcelectra import BATCH, LR, MAXLEN, MODEL, SEED, TextDS  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"
EPOCHS = 2  # the point that was best on val in exp-0002


def build_pool() -> pd.DataFrame:
    """The unlabelled pool: external text only. Original labels are retained solely for the control analysis."""
    data = load_all()
    ours_train = set(load_split("train")["text"])
    ours_val = set(load_split("val")["text"])
    ours_test = set(load_split("test")["text"])

    frames = [data[k].assign(source=k) for k in ("unsmile", "APEACH", "kor-hate-sentence")]
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")

    # leakage barrier: drop every sentence overlapping our data, train included
    before = len(pool)
    pool = pool[~pool["text"].isin(ours_train | ours_val | ours_test)].reset_index(drop=True)
    print(f"unlabelled pool: {before} → {len(pool)} after removing leakage")

    # leakage assertions (abort on failure)
    assert not pool["text"].isin(ours_val).any(), "❌ val sentences remain in the pool"
    assert not pool["text"].isin(ours_test).any(), "❌ test sentences remain in the pool"
    print("✅ leakage check passed (pool ∩ val = ∅, pool ∩ test = ∅)")

    return pool.rename(columns={"label": "orig_label"})[["text", "orig_label", "source"]]


def train_teacher():
    """Train the teacher on our train split alone. val is inspected only to report performance; the hyperparameters were fixed back in exp-0002."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

    train_dl = DataLoader(TextDS(load_split("train"), tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, len(train_dl) * EPOCHS)

    model.train()
    for ep in range(EPOCHS):
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            if i % 150 == 0:
                print(f"  ep{ep} {i}/{len(train_dl)} loss={out.loss.item():.4f}", flush=True)
    return model, tok


@torch.no_grad()
def predict_probs(model, tok, texts: list[str]) -> np.ndarray:
    model.eval()
    df = pd.DataFrame({"text": texts, "label": [0] * len(texts)})
    dl = DataLoader(TextDS(df, tok), batch_size=128)
    out = []
    for batch in dl:
        batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        out.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
    return np.array(out)


def main():
    ART.mkdir(exist_ok=True)
    pool = build_pool()

    print("\ntraining the teacher (our train split only)…")
    model, tok = train_teacher()

    # teacher performance on val — recorded for the log, never used to select parameters
    val = load_split("val")
    pv = predict_probs(model, tok, val["text"].tolist())
    yv = val["label"].tolist()
    best = max(((th, f1_binary(yv, [int(p >= th) for p in pv]))
                for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
    print(f"teacher val F1 = {best[1]['f1']:.4f} (th={best[0]:.3f})")

    print(f"\nrelabelling the {len(pool)}-row unlabelled pool…")
    pool["teacher_prob"] = predict_probs(model, tok, pool["text"].tolist())

    # positive rate the teacher assigns under our definition vs the rate under the
    # original labels — direct evidence of the label-definition gap
    for th in (0.5,):
        teacher_pos = (pool["teacher_prob"] >= th).mean()
        orig_pos = pool["orig_label"].mean()
        agree = ((pool["teacher_prob"] >= th).astype(int) == pool["orig_label"]).mean()
        print(f"\n[label-definition gap] positive rate under the original labels {orig_pos:.3f} vs under the teacher, i.e. our definition, {teacher_pos:.3f}")
        print(f"                 agreement with the original labels: {agree:.3f}")

    pool.to_csv(ART / "pseudo_labels.csv", index=False)
    print(f"\nsaved: {ART / 'pseudo_labels.csv'} ({len(pool)} rows)")


if __name__ == "__main__":
    main()
