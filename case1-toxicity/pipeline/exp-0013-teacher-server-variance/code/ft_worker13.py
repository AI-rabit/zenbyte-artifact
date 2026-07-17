"""exp-0013: fastText 1런 워커 (프로세스 격리 — exp-0012 [수정 1] 승계).

usage: python ft_worker13.py <train.txt> <out_prefix>
설정은 Q7 고정(FT_Q2_CFG). val/test 확률을 artifacts/에 저장.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common13 import ART, FT_Q2_CFG  # noqa: E402

from common import load_split, train_with_retry  # noqa: E402
from transfer_matrix import prob_pos  # noqa: E402


def main():
    train_path, prefix = sys.argv[1], sys.argv[2]
    model, final_lr = train_with_retry(input=train_path, wordNgrams=2, epoch=25,
                                       loss="softmax", thread=1, verbose=0, **FT_Q2_CFG)
    for name in ("val", "test"):
        df = load_split(name)
        np.save(ART / f"{prefix}_{name}_probs.npy", prob_pos(model, df["text"]))
    (ART / f"{prefix}_meta.json").write_text(json.dumps({"final_lr": final_lr}), encoding="utf-8")


if __name__ == "__main__":
    main()
