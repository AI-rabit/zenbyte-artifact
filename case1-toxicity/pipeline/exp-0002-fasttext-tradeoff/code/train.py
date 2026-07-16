"""exp-0002 fastText 학습·평가 (단일 설정 실행 및 스윕의 공용 엔진).

사용:
  python train.py                       # 기본 설정 1회 (스켈레톤 확인용)
  python train.py --sweep               # 그리드 스윕 → ../artifacts/sweep_results.csv

평가: val set 기준 (test는 동작점 확정 후 최종 1회만 사용).
크기: int8 직렬화 실측 산정 (common.int8_serialized_bytes).
재현성: thread=1 고정 (fastText는 멀티스레드 시 비결정적).
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import fasttext
import pandas as pd

from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, jamo_decompose, load_split

fasttext.FastText.eprint = lambda x: None  # 경고 출력 억제


def make_input(split: str, jamo: bool) -> Path:
    """fastText 입력 파일 생성 (자모 분해 옵션 반영)."""
    suffix = "jamo" if jamo else "raw"
    path = DATA / f"{split}.{suffix}.txt"
    if path.exists():
        return path
    df = load_split(split)
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            text = jamo_decompose(r["text"]) if jamo else r["text"]
            f.write(f"__label__{r['label']} {text}\n")
    return path


def evaluate(model, split: str, jamo: bool) -> dict:
    df = load_split(split)
    texts = [jamo_decompose(t) if jamo else t for t in df["text"]]
    labels, _ = model.predict(texts)
    y_pred = [int(l[0].replace("__label__", "")) for l in labels]
    return f1_binary(df["label"].tolist(), y_pred)


def run_config(cfg: dict) -> dict:
    train_path = make_input("train", cfg["jamo"])
    t0 = time.time()
    # lr이 큰 일부 설정에서 fastText가 NaN으로 발산 → lr 반감 재시도 (최종 lr 기록)
    lr = cfg.get("lr", 0.5)
    for attempt in range(5):
        try:
            model = fasttext.train_supervised(
                input=str(train_path),
                dim=cfg["dim"], bucket=cfg["bucket"],
                minn=cfg["minn"], maxn=cfg["maxn"],
                wordNgrams=2, epoch=25, lr=lr, thread=1, verbose=0,
            )
            break
        except RuntimeError:
            lr /= 2
    else:
        raise RuntimeError(f"NaN persisted after retries: {cfg}")
    cfg = {**cfg, "lr": lr}
    train_sec = time.time() - t0
    metrics = evaluate(model, "val", cfg["jamo"])
    size = int8_serialized_bytes(model)
    result = {**cfg, **metrics, "int8_bytes": size, "int8_mb": round(size / 2**20, 2),
              "train_sec": round(train_sec, 1)}
    return result, model


DEFAULT = {"dim": 32, "bucket": 100_000, "minn": 2, "maxn": 4, "jamo": False, "lr": 0.5}

GRID = {
    "dim": [16, 32, 64],
    "bucket": [50_000, 100_000, 250_000],
    "minmax": [(0, 0), (2, 4), (2, 5)],
    "jamo": [False, True],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    if not args.sweep:
        result, model = run_config(DEFAULT)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        model.save_model(str(ARTIFACTS / "skeleton.bin"))
        return

    rows = []
    combos = list(itertools.product(GRID["dim"], GRID["bucket"], GRID["minmax"], GRID["jamo"]))
    for i, (dim, bucket, (minn, maxn), jamo) in enumerate(combos, 1):
        cfg = {"dim": dim, "bucket": bucket, "minn": minn, "maxn": maxn, "jamo": jamo}
        try:
            result, _ = run_config(cfg)
            rows.append(result)
            print(f"[{i}/{len(combos)}] {cfg} → F1={result['f1']:.3f}, {result['int8_mb']}MB")
        except RuntimeError as e:
            # 발산 설정도 기록한다 — 실패도 트레이드오프 곡선의 데이터다
            rows.append({**cfg, "f1": None, "error": str(e)})
            print(f"[{i}/{len(combos)}] {cfg} → 발산(NaN), 기록 후 계속")
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(ARTIFACTS / "sweep_results.csv", index=False)
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
