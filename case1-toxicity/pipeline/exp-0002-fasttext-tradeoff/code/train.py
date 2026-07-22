"""exp-0002 fastText training and evaluation (the shared engine for both a
single run and the sweep).

Usage:
  python train.py                       # one run at the default config (skeleton check)
  python train.py --sweep               # grid sweep → ../artifacts/sweep_results.csv

Evaluation: on the val set (test is opened exactly once, after the operating
point is fixed).
Size: measured int8 serialized size (common.int8_serialized_bytes).
Reproducibility: thread=1 is fixed, since fastText is non-deterministic when
multi-threaded.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import fasttext
import pandas as pd

from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, jamo_decompose, load_split

fasttext.FastText.eprint = lambda x: None  # silence warning output


def make_input(split: str, jamo: bool) -> Path:
    """Build the fastText input file, applying the jamo-decomposition option."""
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
    # at larger lr some configurations diverge to NaN → retry with lr halved
    # (the lr that finally worked is recorded)
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
            # diverged configurations are recorded too — a failure is still a point
            # on the trade-off curve
            rows.append({**cfg, "f1": None, "error": str(e)})
            print(f"[{i}/{len(combos)}] {cfg} → diverged (NaN); recorded, continuing")
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(ARTIFACTS / "sweep_results.csv", index=False)
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
