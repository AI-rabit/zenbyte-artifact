"""exp-0011 추기: 배포 모델(TF-IDF+SVM int8)의 경고율 캘리브레이션 표 재산출.

배경: exp-0004의 캘리브레이션 표(경고율/오경고율)는 구 fastText 모델(th=0.375) 기준이었다.
모델이 두 번 교체되며(0.375 → 0.425 → 0.475) 사용자 관점 지표가 배포 실체와 어긋났다.
이 스크립트는 배포되는 int8 ZBSV 모델로 같은 표를 val에서 재산출한다.

검증 기준(사전 정의): th=0.475의 val F1이 exp-0011 기록치 0.8263과 ±0.001 내로 일치해야 한다.
(불일치 시 모델/데이터가 배포 실체와 다른 것이므로 표 전체가 무효)

실행: source .venv-research/bin/activate 후
  python calibrate_threshold.py
"""
import csv
from pathlib import Path

from reference_svm import ZBSVModel, ART

DATA = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff" / "data"
RECORDED_VAL_F1 = 0.8263  # exp-0011 result.md: int8 양자화 후 val F1
THRESHOLDS = [0.20, 0.30, 0.375, 0.425, 0.475, 0.55, 0.70]


def main() -> None:
    model = ZBSVModel(ART / "toxicity_model.zbsv")
    rows = list(csv.DictReader(open(DATA / "val.csv", encoding="utf-8")))
    probs = [model.prob_toxic(r["text"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    n = len(rows)
    n_pos = sum(labels)
    n_neg = n - n_pos
    print(f"val {n}문장 (양성 {n_pos}, 음성 {n_neg})\n")

    print("| 임계값 | 경고율(전체) | 재현율 | 정밀도 | F1 | 오경고율(정상문장 중) |")
    print("|---|---|---|---|---|---|")
    f1_at_deploy = None
    for th in THRESHOLDS:
        pred = [p >= th for p in probs]
        tp = sum(1 for pr, y in zip(pred, labels) if pr and y == 1)
        fp = sum(1 for pr, y in zip(pred, labels) if pr and y == 0)
        warn = sum(pred)
        prec = tp / warn if warn else 0.0
        rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fa = fp / n_neg  # 오경고율: 정상 문장 중 경고 비율
        mark = " ← 배포" if th == 0.475 else ""
        print(f"| {th} | {warn / n:.1%} | {rec:.1%} | {prec:.1%} | {f1:.3f} | {fa:.1%} |{mark}")
        if th == 0.475:
            f1_at_deploy = f1

    delta = abs(f1_at_deploy - RECORDED_VAL_F1)
    status = "PASS" if delta <= 0.001 else "FAIL"
    print(f"\n검증: val F1@0.475 = {f1_at_deploy:.4f} vs 기록치 {RECORDED_VAL_F1} (Δ {delta:.4f}) → {status}")
    if status == "FAIL":
        raise SystemExit("검증 실패 — 모델/데이터가 배포 실체와 다르다. 표 무효.")


if __name__ == "__main__":
    main()
