"""exp-0012 공용: 경로 설정, 기존 실험 코드 재사용, 저장 헬퍼.

원칙 (spec 사전 등록): 모든 수치는 스크립트 산출물에서만 나온다. 학습 설정은
exp-0009/0010/0011의 원설정을 바이트 단위로 재사용하며 변경하지 않는다.
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

# 공정 재튜닝 최적점 (exp-0010 fair_tuning 결과 — Q2)
FT_Q2_CFG = {"dim": 16, "bucket": 250_000, "minn": 2, "maxn": 4, "lr": 0.125}
# exp-0002/0009 동작점 OP (exp-0008 transfer_matrix.OP와 동일 — Q3·Q4)
FT_OP_CFG = {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 5, "lr": 0.125}

SVM_Q2_VEC = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                  sublinear_tf=True, max_features=500_000)   # exp-0011 배포 동작점
SVM_Q3_VEC = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                  sublinear_tf=True, max_features=500_000)   # exp-0010 1차 벤치마크
SVC_KW = dict(C=0.5, max_iter=5000)


def ensure_dirs():
    ART.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)


def append_csv(path: Path, row: dict):
    """헤더 자동, 런마다 즉시 append (중단 내성)."""
    import csv
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
