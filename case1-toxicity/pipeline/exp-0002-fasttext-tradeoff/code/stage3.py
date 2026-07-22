"""exp-0002 stage 3: a fine search of the 6–15MB band under the re-derived
budget (≤15MB).

Stages 1 and 2 explored only up to bucket = 250k, leaving larger buckets such
as 500k untried. Each configuration is averaged over 3 repeats (to cope with
fastText's stochasticity), evaluated on val with threshold tuning.
"""
import json
import statistics

import numpy as np

from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, load_split, train_with_retry
from threshold import prob_positive

# candidates in the 6–15MB band (the comment on each line is its expected int8 size)
CONFIGS = [
    {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125},  # 6.0MB (reference point)
    {"dim": 16, "bucket": 350_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~8MB
    {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~11.5MB
    {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 5, "lr": 0.125},  # ~11.5MB (wider n-gram range)
    {"dim": 24, "bucket": 350_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~11.5MB (larger dim)
    {"dim": 32, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125},  # 10.4MB (re-checked, averaged over repeats)
    {"dim": 32, "bucket": 350_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~14.8MB (top of the budget)
]
REPEATS = 3


def main():
    val = load_split("val")
    y = val["label"].tolist()
    results = []
    for cfg in CONFIGS:
        f1s, size = [], None
        for _ in range(REPEATS):
            model, _ = train_with_retry(input=str(DATA / "train.raw.txt"),
                                        dim=cfg["dim"], bucket=cfg["bucket"], minn=cfg["minn"],
                                        maxn=cfg["maxn"], wordNgrams=2, epoch=25, lr=cfg["lr"],
                                        loss="softmax", thread=1, verbose=0)
            p1 = prob_positive(model, val["text"].tolist())
            best = max(((th, f1_binary(y, (p1 >= th).astype(int).tolist()))
                        for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
            f1s.append(best[1]["f1"])
            size = int8_serialized_bytes(model)
        row = {**cfg, "int8_mb": round(size / 2**20, 2),
               "f1_mean": round(statistics.mean(f1s), 4),
               "f1_std": round(statistics.stdev(f1s), 4),
               "f1_runs": [round(x, 4) for x in f1s]}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
    best = max((r for r in results if r["int8_mb"] <= 15.0), key=lambda r: r["f1_mean"])
    print(f"\nbest under 15MB: {best}")


if __name__ == "__main__":
    main()
