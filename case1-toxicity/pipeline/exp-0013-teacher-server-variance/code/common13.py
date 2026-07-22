"""exp-0013 shared paths and settings. Every setting is inherited unchanged from
exp-0002/0009/0010/0011 and must not be modified."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESEARCH = HERE.parent.parent
ART = HERE.parent / "artifacts"
DATA = HERE.parent / "data"

for exp in ("exp-0002-fasttext-tradeoff", "exp-0008-dataset-survey",
            "exp-0010-constrained-benchmark"):
    sys.path.insert(0, str(RESEARCH / exp / "code"))

EXP9_ART = RESEARCH / "exp-0009-distillation" / "artifacts"

TEACHER_SEEDS = [42, 43, 44]          # fixed in advance by the spec
TEACHER_EPOCHS = 2                    # exactly as in exp-0009 teacher.py (the best point on val in exp-0002)
CONF = 0.9                            # confidence filter, exactly as in exp-0009/0011

SVM_VEC = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
               sublinear_tf=True, max_features=500_000)   # the deployed operating point (exp-0011)
SVC_KW = dict(C=0.5, max_iter=5000)
FT_Q2_CFG = {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125}  # exp-0010 fair re-tuning


def ensure_dirs():
    ART.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)


def append_csv(path: Path, row: dict):
    import csv
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
