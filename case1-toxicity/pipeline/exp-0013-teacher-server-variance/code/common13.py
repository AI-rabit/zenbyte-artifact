"""exp-0013 공용: 경로·설정. 원설정은 exp-0002/0009/0010/0011에서 그대로 승계 (변경 금지)."""
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

TEACHER_SEEDS = [42, 43, 44]          # 사전 고정 (spec)
TEACHER_EPOCHS = 2                    # exp-0009 teacher.py 그대로 (exp-0002 val 최고 지점)
CONF = 0.9                            # 신뢰도 필터 (exp-0009/0011 그대로)

SVM_VEC = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
               sublinear_tf=True, max_features=500_000)   # 배포 동작점 (exp-0011)
SVC_KW = dict(C=0.5, max_iter=5000)
FT_Q2_CFG = {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125}  # exp-0010 공정 재튜닝


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
