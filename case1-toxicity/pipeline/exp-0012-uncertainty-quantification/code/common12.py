"""exp-0012 shared code: path setup, reuse of the earlier experiments' code,
and saving helpers.

Pre-registered principle (from the spec): every number comes from a script's
output and nowhere else. The training settings reuse those of exp-0009/0010/0011
byte for byte and are not modified.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESEARCH = HERE.parent.parent
ART = HERE.parent / "artifacts"
DATA = HERE.parent / "data"

for exp, sub in (("exp-0002-fasttext-tradeoff", "code"),
                 ("exp-0008-dataset-survey", "code"),
                 ("exp-0010-constrained-benchmark", "code"),
                 ("exp-0011-svm-deployment", "code")):
    sys.path.insert(0, str(RESEARCH / exp / sub))

BOOT_SEED = 20260718
B = 10_000
R = 10

EXP9_ART = RESEARCH / "exp-0009-distillation" / "artifacts"
EXP11_ART = RESEARCH / "exp-0011-svm-deployment" / "artifacts"

# the fair re-tuning optimum (from exp-0010 fair_tuning — Q2)
FT_Q2_CFG = {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125}
# the exp-0002/0009 operating point OP (identical to exp-0008 transfer_matrix.OP — Q3, Q4)
FT_OP_CFG = {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 5, "lr": 0.125}

SVM_Q2_VEC = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                  sublinear_tf=True, max_features=500_000)   # exp-0011 deployed operating point
SVM_Q3_VEC = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                  sublinear_tf=True, max_features=500_000)   # exp-0010 first benchmark
SVC_KW = dict(C=0.5, max_iter=5000)


def ensure_dirs():
    ART.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)


def append_csv(path: Path, row: dict):
    """Writes the header automatically and appends after every run, so an interruption loses nothing."""
    import csv
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
