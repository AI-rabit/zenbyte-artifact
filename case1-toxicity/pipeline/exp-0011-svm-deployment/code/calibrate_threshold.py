"""exp-0011 addendum: recompute the warning-rate calibration table for the
deployed model (TF-IDF + SVM, int8).

Background: the calibration table in exp-0004 (warning rate / false-warning
rate) was computed for the old fastText model at th=0.375. The model was then
replaced twice (0.375 → 0.425 → 0.475), leaving the user-facing figures out of
step with what is actually deployed. This script recomputes the same table on
val using the int8 ZBSV model that ships.

Pre-registered acceptance criterion: val F1 at th=0.475 must agree with the
0.8263 recorded in exp-0011 to within ±0.001. If it does not, the model or the
data differ from what is deployed and the whole table is void.

Run: activate the research virtualenv, then
  python calibrate_threshold.py
"""
import csv
from pathlib import Path

from reference_svm import ZBSVModel, ART

DATA = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff" / "data"
RECORDED_VAL_F1 = 0.8263  # exp-0011 result.md: val F1 after int8 quantization
THRESHOLDS = [0.20, 0.30, 0.375, 0.425, 0.475, 0.55, 0.70]


def main() -> None:
    model = ZBSVModel(ART / "toxicity_model.zbsv")
    rows = list(csv.DictReader(open(DATA / "val.csv", encoding="utf-8")))
    probs = [model.prob_toxic(r["text"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    n = len(rows)
    n_pos = sum(labels)
    n_neg = n - n_pos
    print(f"val: {n} sentences ({n_pos} positive, {n_neg} negative)\n")

    print("| threshold | warning rate (all) | recall | precision | F1 | false-warning rate (of ordinary sentences) |")
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
        fa = fp / n_neg  # false-warning rate: the share of ordinary sentences warned about
        mark = " ← deployed" if th == 0.475 else ""
        print(f"| {th} | {warn / n:.1%} | {rec:.1%} | {prec:.1%} | {f1:.3f} | {fa:.1%} |{mark}")
        if th == 0.475:
            f1_at_deploy = f1

    delta = abs(f1_at_deploy - RECORDED_VAL_F1)
    status = "PASS" if delta <= 0.001 else "FAIL"
    print(f"\ncheck: val F1@0.475 = {f1_at_deploy:.4f} vs recorded {RECORDED_VAL_F1} (Δ {delta:.4f}) → {status}")
    if status == "FAIL":
        raise SystemExit("check failed — the model or data differ from what is deployed; the table is void.")


if __name__ == "__main__":
    main()
