"""exp-0002 임계값 튜닝: 스윕 상위 설정들을 재학습해 val에서 결정 임계값 최적화.

fastText 기본 판정은 argmax(=0.5)이나, 불균형 데이터(양성 17.6%)에서는
P(1) 임계값을 val로 튜닝하는 것이 표준 관행이다. 임계값은 val에서만 고르고
test에는 그대로 적용한다 (선택 누수 방지).
"""
import json

import fasttext
import numpy as np

from common import ARTIFACTS, f1_binary, jamo_decompose, load_split
from train import make_input

fasttext.FastText.eprint = lambda x: None

# 스윕 상위 + 소형 후보 (sweep_results.csv 기준)
CANDIDATES = [
    {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "jamo": False, "lr": 0.125},  # 최고 F1
    {"dim": 32, "bucket": 50_000, "minn": 2, "maxn": 5, "jamo": False, "lr": 0.5},     # ≤5MB 최고
    {"dim": 16, "bucket": 100_000, "minn": 2, "maxn": 5, "jamo": False, "lr": 0.5},    # 최소형
    {"dim": 32, "bucket": 100_000, "minn": 2, "maxn": 4, "jamo": True, "lr": 0.5},     # 자모 최고
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
