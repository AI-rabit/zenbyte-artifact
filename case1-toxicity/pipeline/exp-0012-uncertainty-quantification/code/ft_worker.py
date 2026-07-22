"""exp-0012: a single-run fastText worker, executed in a fresh process per run
so that runs stay isolated.

Background: repeating training inside one process was observed to start
producing NaN persistently from some point on (run0 and run1 succeeded, then
run2 hit NaN on all six lr halvings — process state contamination is suspected).
The worker takes one JSON job file, performs a single training run (including
train_with_retry) and saves the probability vectors.

usage: python ft_worker.py <job.json>
job = {"input": str, "cfg": {...}, "out_prefix": str, "evals": a subset of ["val","test"]}
output: {out_prefix}_{eval}_probs.npy + {out_prefix}_meta.json (final_lr, sec)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common12 import ART  # noqa: E402

from common import load_split, train_with_retry  # noqa: E402
from transfer_matrix import prob_pos  # noqa: E402

TRAIN_KW = dict(wordNgrams=2, epoch=25, loss="softmax", thread=1, verbose=0)


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    t0 = time.time()
    model, final_lr = train_with_retry(input=job["input"], **TRAIN_KW, **job["cfg"])
    sec = round(time.time() - t0, 1)
    for name in job["evals"]:
        df = load_split(name)
        np.save(ART / f"{job['out_prefix']}_{name}_probs.npy", prob_pos(model, df["text"]))
    (ART / f"{job['out_prefix']}_meta.json").write_text(
        json.dumps({"final_lr": final_lr, "sec": sec}), encoding="utf-8")


if __name__ == "__main__":
    main()
