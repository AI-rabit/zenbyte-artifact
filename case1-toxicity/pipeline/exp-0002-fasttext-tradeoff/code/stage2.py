"""exp-0002 stage-2 sweep: around the stage-1 optimum, extend along epoch,
loss, wordNgrams and oversampling.

autotune was rejected after it ran out of memory (it attempted dim=719,
bucket=5.1M), and is replaced here by a manual extension whose memory ceiling
stays under our control. Two base configurations (best F1 / best under 5MB) are
varied one axis at a time.
Evaluation: val plus threshold tuning (test stays sealed).
"""
import itertools
import json

import fasttext
import numpy as np
import pandas as pd

from common import (ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, load_split,
                    train_with_retry)
from threshold import prob_positive
from train import make_input

fasttext.FastText.eprint = lambda x: None

BASES = {
    "best":  {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125},
    "small": {"dim": 32, "bucket": 50_000, "minn": 2, "maxn": 5, "lr": 0.5},
}


def make_oversampled(factor: int) -> str:
    """Build an input file with the positives of train replicated `factor` times."""
    path = DATA / f"train.raw.os{factor}.txt"
    if not path.exists():
        df = load_split("train")
        with open(path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                n = factor if r["label"] == 1 else 1
                for _ in range(n):
                    f.write(f"__label__{r['label']} {r['text']}\n")
    return str(path)


def eval_with_threshold(model):
    val = load_split("val")
    p1 = prob_positive(model, val["text"].tolist())
    y = val["label"].tolist()
    best = max(((th, f1_binary(y, (p1 >= th).astype(int).tolist()))
                for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
    return best


def run(base_name, base, *, epoch=25, loss="softmax", wordNgrams=2, oversample=1):
    inp = make_oversampled(oversample) if oversample > 1 else str(make_input("train", False))
    try:
        model, lr = train_with_retry(
            input=inp, dim=base["dim"], bucket=base["bucket"], minn=base["minn"],
            maxn=base["maxn"], wordNgrams=wordNgrams, epoch=epoch, lr=base["lr"],
            loss=loss, thread=1, verbose=0)
    except RuntimeError:
        return {"base": base_name, "epoch": epoch, "loss": loss, "wordNgrams": wordNgrams,
                "oversample": oversample, "f1": None, "error": "NaN-persistent"}
    th, m = eval_with_threshold(model)
    return {"base": base_name, "epoch": epoch, "loss": loss, "wordNgrams": wordNgrams,
            "oversample": oversample, "lr": lr, "th": th, "f1": round(m["f1"], 4),
            "precision": round(m["precision"], 3), "recall": round(m["recall"], 3),
            "int8_mb": round(int8_serialized_bytes(model) / 2**20, 2)}


def main():
    rows = []
    for base_name, base in BASES.items():
        rows.append(run(base_name, base))  # reference point
        for epoch in (50, 100):
            rows.append(run(base_name, base, epoch=epoch))
        for loss in ("hs", "ova"):
            rows.append(run(base_name, base, loss=loss))
        for wn in (1, 3):
            rows.append(run(base_name, base, wordNgrams=wn))
        for os_ in (2, 3):
            rows.append(run(base_name, base, oversample=os_))
        for r in rows[-9:]:
            print(json.dumps(r, ensure_ascii=False))
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS / "stage2_results.csv", index=False)
    print("\ntop results:")
    print(df.sort_values("f1", ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
