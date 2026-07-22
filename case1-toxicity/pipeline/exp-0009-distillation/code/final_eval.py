"""exp-0009 final evaluation: the distilled operating point, scored once on the
sealed test split.

Operating point: ours + teacher pseudo-labels (confidence ≥0.9), fastText
dim16 / bucket500k / minn2-maxn5. The threshold is chosen on val and applied
unchanged to test, so the selection cannot leak. 3 repeats.
Reference points: the exp-0002 baseline (test 0.744) and the KcELECTRA teacher
(test 0.853).
"""
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from common import f1_binary, int8_serialized_bytes, load_split, train_with_retry  # noqa: E402
from transfer_matrix import OP, prob_pos, write_ft  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"
DATA = Path(__file__).parent.parent / "data"
CONF = 0.9
REPEATS = 3


def build_train() -> pd.DataFrame:
    ours = load_split("train")[["text", "label"]]
    pool = pd.read_csv(ART / "pseudo_labels.csv")
    conf = pool[(pool["teacher_prob"] >= CONF) | (pool["teacher_prob"] <= 1 - CONF)]
    pseudo = pd.DataFrame({"text": conf["text"], "label": (conf["teacher_prob"] >= 0.5).astype(int)})
    print(f"training set: ours {len(ours)} + pseudo-labelled {len(pseudo)} = {len(ours) + len(pseudo)} rows")
    return pd.concat([ours, pseudo], ignore_index=True)


def main():
    train = build_train()
    val, test = load_split("val"), load_split("test")
    yv, yt = val["label"].tolist(), test["label"].tolist()

    path = write_ft(train, DATA / "final_distilled.txt")
    rows, best_model, best_f1_val = [], None, -1.0

    for i in range(REPEATS):
        model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                    loss="softmax", thread=1, verbose=0, **OP)
        pv = prob_pos(model, val["text"])
        th, mval = max(((t, f1_binary(yv, (pv >= t).astype(int).tolist()))
                        for t in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
        pt = prob_pos(model, test["text"])
        mtest = f1_binary(yt, (pt >= th).astype(int).tolist())
        rows.append({"run": i, "th": round(float(th), 3),
                     "val_f1": round(mval["f1"], 4), **{k: round(v, 4) for k, v in mtest.items()}})
        print(f"  run{i}: th={th:.3f} val={mval['f1']:.4f} | test P={mtest['precision']:.4f} "
              f"R={mtest['recall']:.4f} F1={mtest['f1']:.4f}")
        if mval["f1"] > best_f1_val:
            best_f1_val, best_model, best_th = mval["f1"], model, float(th)

    f1s = [r["f1"] for r in rows]
    size = int8_serialized_bytes(best_model)
    print(f"\n=== exp-0009 final (test, opened once) ===")
    print(f"  test F1 = {statistics.mean(f1s):.4f} ± {statistics.stdev(f1s):.4f}")
    print(f"  int8 size = {size / 2**20:.2f}MB (budget 15MB)")
    print(f"  chosen threshold = {best_th:.3f} (selected on val)")
    print(f"\n  comparison: keyword 0.568 | exp-0002 baseline 0.744 | **distilled {statistics.mean(f1s):.3f}** "
          f"| teacher (KcELECTRA, 420MB) 0.853")
    print(f"  improvement: {statistics.mean(f1s) - 0.7436:+.4f}")

    best_model.save_model(str(ART / "operating_point_distilled.bin"))
    (ART / "operating_point_meta.txt").write_text(
        f"threshold={best_th}\ntest_f1_mean={statistics.mean(f1s)}\nint8_bytes={size}\n")
    print(f"\n  model saved: artifacts/operating_point_distilled.bin")


if __name__ == "__main__":
    main()
