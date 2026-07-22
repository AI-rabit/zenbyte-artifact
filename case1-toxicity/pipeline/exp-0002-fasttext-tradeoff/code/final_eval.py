"""exp-0002 final evaluation: once the operating point is fixed, the sealed
test set is opened — a one-time procedure.

- operating point: dim16 / bucket500k / minn2 / maxn5 / wordNgrams2 / epoch25
  (best on val in stage 3, 10.79MB)
- the threshold is chosen on val and applied unchanged to test, so the
  selection cannot leak
- 3 repeats: test F1 reported as mean±std
- comparison: the keyword baseline (on test); the operating-point model is saved
  for use in exp-0003
"""
import json
import statistics

import numpy as np

from baseline_keyword import predict as keyword_predict
from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, load_split, train_with_retry
from threshold import prob_positive

OP = {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 5, "lr": 0.125}
REPEATS = 3


def main():
    val, test = load_split("val"), load_split("test")
    yv, yt = val["label"].tolist(), test["label"].tolist()

    # keyword baseline (test)
    kw = f1_binary(yt, [keyword_predict(t) for t in test["text"]])
    print(json.dumps({"model": "keyword", "split": "test",
                      **{k: round(v, 4) for k, v in kw.items()}}, ensure_ascii=False))

    rows = []
    best_model = None
    for i in range(REPEATS):
        model, _ = train_with_retry(input=str(DATA / "train.raw.txt"), wordNgrams=2,
                                    epoch=25, loss="softmax", thread=1, verbose=0, **OP)
        pv = prob_positive(model, val["text"].tolist())
        th = max(((t, f1_binary(yv, (pv >= t).astype(int).tolist()))
                  for t in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])[0]
        pt = prob_positive(model, test["text"].tolist())
        m = f1_binary(yt, (pt >= th).astype(int).tolist())
        rows.append({"run": i, "th": round(float(th), 3), **{k: round(v, 4) for k, v in m.items()}})
        print(json.dumps({"model": "fasttext-OP", "split": "test", **rows[-1]}, ensure_ascii=False))
        if best_model is None or m["f1"] >= max(r["f1"] for r in rows[:-1] or [{"f1": -1}]):
            best_model = model

    f1s = [r["f1"] for r in rows]
    print(json.dumps({"model": "fasttext-OP", "split": "test",
                      "f1_mean": round(statistics.mean(f1s), 4),
                      "f1_std": round(statistics.stdev(f1s), 4),
                      "int8_mb": round(int8_serialized_bytes(best_model) / 2**20, 2)},
                     ensure_ascii=False))
    best_model.save_model(str(ARTIFACTS / "operating_point.bin"))
    print(f"operating-point model saved: artifacts/operating_point.bin")


if __name__ == "__main__":
    main()
