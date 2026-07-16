"""exp-0002 데이터 증강 실험: unsmile 추가가 고원(F1~0.78)을 뚫는가.

설계:
- unsmile(train+valid tsv 전량)을 **우리 train에만** 추가. val/test는 불변 (비교 가능성 유지).
- 우리 val/test와 문장 중복 제거 (평가 누수 방지).
- 라벨: clean=1 → 0, 그 외(혐오 카테고리/악플·욕설) → 1.
- fastText 확률성 대응: 증강 전/후 각 5회 반복 → mean±std 비교.
- 라이선스: CC-BY-NC-ND 4.0 — PoC 한정 사용, 상업화 시 재학습 (spec 제약).
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
    """targeted=True: 우리 라벨 정의와 가장 가까운 '악플/욕설' 양성 + clean 음성만 추가
    (혐오 카테고리 전용 행은 제외 — 라벨 기준 불일치 가설 검증)."""
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
    print(f"unsmile {before} → 누수·중복 제거 후 {len(un)} (양성 {un['label'].mean():.3f})")

    with open(path, "w", encoding="utf-8") as f:
        for _, r in base.iterrows():
            f.write(f"__label__{r['label']} {r['text']}\n")
        for _, r in un.iterrows():
            f.write(f"__label__{r['label']} {r['text']}\n")
    total_pos = base["label"].sum() + un["label"].sum()
    total = len(base) + len(un)
    print(f"증강 train: {total}건 (양성 {total_pos}건, {total_pos/total:.3f})")
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
            run_arm(f"{cname}/표적증강", cfg, aug)
        return
    aug = build_augmented_train()
    plain = str(DATA / "train.raw.txt")
    for cname, cfg in CONFIGS.items():
        run_arm(f"{cname}/기존", cfg, plain)
        run_arm(f"{cname}/증강", cfg, aug)


if __name__ == "__main__":
    main()
