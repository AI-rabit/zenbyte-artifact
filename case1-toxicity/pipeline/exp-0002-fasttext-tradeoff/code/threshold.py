"""exp-0002 threshold tuning: retrain the top sweep configurations and optimize
the decision threshold on val.

fastText decides by argmax (i.e. 0.5) by default, but on imbalanced data
(17.6% positive) tuning the P(1) threshold on val is standard practice. The
threshold is chosen on val only and applied unchanged to test, so the selection
cannot leak.
"""
import json

import fasttext
import numpy as np

from common import ARTIFACTS, f1_binary, jamo_decompose, load_split
from train import make_input

fasttext.FastText.eprint = lambda x: None

# top sweep entries plus the small candidates (from sweep_results.csv)
CANDIDATES = [
    {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "jamo": False, "lr": 0.125},  # best F1
    {"dim": 32, "bucket": 50_000, "minn": 2, "maxn": 5, "jamo": False, "lr": 0.5},     # best under 5MB
    {"dim": 16, "bucket": 100_000, "minn": 2, "maxn": 5, "jamo": False, "lr": 0.5},    # smallest
    {"dim": 32, "bucket": 100_000, "minn": 2, "maxn": 4, "jamo": True, "lr": 0.5},     # best jamo variant
]


def prob_positive(model, texts):
    labels, probs = model.predict(texts, k=2)
    out = []
    for ls, ps in zip(labels, probs):
        d = {l: p for l, p in zip(ls, ps)}
        out.append(d.get("__label__1", 0.0))
    return np.array(out)


def main():
    val = load_split("val")
    for cfg in CANDIDATES:
        model = fasttext.train_supervised(
            input=str(make_input("train", cfg["jamo"])),
            dim=cfg["dim"], bucket=cfg["bucket"], minn=cfg["minn"], maxn=cfg["maxn"],
            wordNgrams=2, epoch=25, lr=cfg["lr"], thread=1, verbose=0,
        )
        texts = [jamo_decompose(t) if cfg["jamo"] else t for t in val["text"]]
        p1 = prob_positive(model, texts)
        y = val["label"].tolist()

        best = None
        for th in np.arange(0.05, 0.95, 0.025):
            m = f1_binary(y, (p1 >= th).astype(int).tolist())
            if best is None or m["f1"] > best[1]["f1"]:
                best = (round(float(th), 3), m)
        th, m = best
        print(json.dumps({**cfg, "best_threshold": th,
                          "precision": round(m["precision"], 3),
                          "recall": round(m["recall"], 3),
                          "f1": round(m["f1"], 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
