"""exp-0012: fastText 1런 워커 — 런마다 새 프로세스로 격리 실행된다.

배경: 같은 프로세스에서 학습을 반복하면 어느 시점부터 NaN이 지속 발생하는 현상 관측
(run0·run1 성공 후 run2가 lr 반감 6회 전부 NaN — 프로세스 상태 오염 의심).
워커는 JSON 잡 파일 하나를 받아 1회 학습(train_with_retry 포함)하고 확률 벡터를 저장한다.

usage: python ft_worker.py <job.json>
job = {"input": str, "cfg": {...}, "out_prefix": str, "evals": ["val","test"] 부분집합}
출력: {out_prefix}_{eval}_probs.npy + {out_prefix}_meta.json (final_lr, sec)
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
