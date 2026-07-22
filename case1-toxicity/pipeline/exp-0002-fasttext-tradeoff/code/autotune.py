"""exp-0002 fastText autotune: automatic hyperparameter search against val
(5-minute budget).

The manual grid stalled at a plateau around F1 ≈ 0.75, so the built-in
optimizer is used to widen the search space (loss, lr, epoch, ngram, dim, dsub
and so on) and check whether that plateau is real.
"""
import json
import sys

import fasttext
import numpy as np

from common import ARTIFACTS, DATA, f1_binary, int8_serialized_bytes, load_split
from threshold import prob_positive

fasttext.FastText.eprint = lambda x: None

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 300

model = fasttext.train_supervised(
    input=str(DATA / "train.raw.txt"),
    autotuneValidationFile=str(DATA / "val.raw.txt"),
    autotuneDuration=DURATION,
    autotuneMetric="f1:__label__1",
    verbose=3,
)

args = model.f.getArgs()
print(json.dumps({k: getattr(args, k) for k in
                  ("dim", "bucket", "minn", "maxn", "wordNgrams", "epoch", "lr", "loss")},
                 default=str, ensure_ascii=False))

val = load_split("val")
p1 = prob_positive(model, val["text"].tolist())
y = val["label"].tolist()
best = max(((th, f1_binary(y, (p1 >= th).astype(int).tolist()))
            for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
print(f"autotune model: int8={int8_serialized_bytes(model)/2**20:.2f}MB, "
      f"argmax F1={f1_binary(y, [int(p >= 0.5) for p in p1])['f1']:.4f}, "
      f"th={best[0]:.3f} F1={best[1]['f1']:.4f} (P={best[1]['precision']:.3f} R={best[1]['recall']:.3f})")
model.save_model(str(ARTIFACTS / "autotune_best.bin"))
