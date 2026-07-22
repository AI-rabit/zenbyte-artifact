"""exp-0002 data augmentation: does adding unsmile break through the plateau
at F1 ≈ 0.78?

Design:
- unsmile (all of train+valid tsv) is added to **our train split only**. val and
  test are left untouched, so results stay comparable.
- sentences overlapping our val/test are dropped (no evaluation leakage).
- labels: clean=1 → 0, everything else (hate categories, abuse and profanity) → 1.
- to cope with fastText's stochasticity: 5 repeats before and after augmentation,
  compared as mean±std.
- license: CC-BY-NC-ND 4.0 — PoC use only; commercial use would require
  retraining (a constraint recorded in the spec).

Note: the Korean strings below are column names in the unsmile dataset and are
kept verbatim.
"""
import json
import statistics
import unicodedata
import re

import numpy as np
import pandas as pd

from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, load_split, train_with_retry
from threshold import prob_positive

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def build_augmented_train(targeted: bool = False) -> str:
    """targeted=True: add only the positives closest to our own label definition
    (the '악플/욕설' abuse/profanity column) plus clean negatives, excluding rows
    that are hate-category only — this tests the label-mismatch hypothesis."""
    path = DATA / ("train.aug-targeted.txt" if targeted else "train.aug.txt")
    base = load_split("train")
    held = set(load_split("val")["text"]) | set(load_split("test")["text"]) | set(base["text"])

    frames = []
    for f in ("unsmile_train_v1.0.tsv", "unsmile_valid_v1.0.tsv"):
        frames.append(pd.read_csv(DATA / f, sep="\t"))
    un = pd.concat(frames, ignore_index=True)
    un["text"] = un["문장"].map(normalize)
    un["label"] = (un["clean"] == 0).astype(int)
    if targeted:
        un = un[(un["악플/욕설"] == 1) | (un["clean"] == 1)]
    before = len(un)
    un = un[(un["text"].str.len() > 0) & (~un["text"].isin(held))]
    un = un.drop_duplicates(subset="text")
    print(f"unsmile {before} → {len(un)} after removing leakage and duplicates (positive {un['label'].mean():.3f})")

    with open(path, "w", encoding="utf-8") as f:
        for _, r in base.iterrows():
            f.write(f"__label__{r['label']} {r['text']}\n")
        for _, r in un.iterrows():
            f.write(f"__label__{r['label']} {r['text']}\n")
    total_pos = base["label"].sum() + un["label"].sum()
    total = len(base) + len(un)
    print(f"augmented train: {total} rows (positive {total_pos}, {total_pos/total:.3f})")
    return str(path)


def eval_once(model):
    val = load_split("val")
    p1 = prob_positive(model, val["text"].tolist())
    y = val["label"].tolist()
    best = max(((th, f1_binary(y, (p1 >= th).astype(int).tolist()))
                for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
    return best


CONFIGS = {
    "best":  {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125},
    "small": {"dim": 32, "bucket": 50_000, "minn": 2, "maxn": 5, "lr": 0.25},
}
REPEATS = 5


def run_arm(name: str, cfg: dict, train_file: str):
    f1s, size = [], None
    for i in range(REPEATS):
        model, _ = train_with_retry(input=train_file, dim=cfg["dim"], bucket=cfg["bucket"],
                                    minn=cfg["minn"], maxn=cfg["maxn"], wordNgrams=2,
                                    epoch=25, lr=cfg["lr"], loss="softmax", thread=1, verbose=0)
        th, m = eval_once(model)
        f1s.append(m["f1"])
        size = int8_serialized_bytes(model)
    mean, std = statistics.mean(f1s), statistics.stdev(f1s)
    print(json.dumps({"arm": name, "f1_mean": round(mean, 4), "f1_std": round(std, 4),
                      "f1_runs": [round(x, 4) for x in f1s],
                      "int8_mb": round(size / 2**20, 2)}, ensure_ascii=False))
    return mean, std


def main():
    import sys
    if "--targeted" in sys.argv:
        aug = build_augmented_train(targeted=True)
        for cname, cfg in CONFIGS.items():
            run_arm(f"{cname}/targeted-aug", cfg, aug)
        return
    aug = build_augmented_train()
    plain = str(DATA / "train.raw.txt")
    for cname, cfg in CONFIGS.items():
        run_arm(f"{cname}/baseline", cfg, plain)
        run_arm(f"{cname}/augmented", cfg, aug)


if __name__ == "__main__":
    main()
