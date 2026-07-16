"""exp-0002 3단계: 재산정된 예산(≤15MB)의 6~15MB 대역 정밀 탐색.

1·2단계는 bucket ≤ 250k까지만 탐색 → 500k 등 대형 버킷은 미답.
각 설정 3회 반복 평균 (fastText 확률성 대응), val + 임계값 튜닝.
"""
import json
import statistics

import numpy as np

from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, load_split, train_with_retry
from threshold import prob_positive

# 6~15MB 대역 후보 (int8 예상 크기 주석)
CONFIGS = [
    {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125},  # 6.0MB (기준점)
    {"dim": 16, "bucket": 350_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~8MB
    {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~11.5MB
    {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 5, "lr": 0.125},  # ~11.5MB (n-gram 폭 확대)
    {"dim": 24, "bucket": 350_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~11.5MB (dim 확대)
    {"dim": 32, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125},  # 10.4MB (재검, 반복평균)
    {"dim": 32, "bucket": 350_000, "minn": 2, "maxn": 4, "lr": 0.125},  # ~14.8MB (예산 상단)
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
    print(f"\n≤15MB 최고: {best}")


if __name__ == "__main__":
    main()
