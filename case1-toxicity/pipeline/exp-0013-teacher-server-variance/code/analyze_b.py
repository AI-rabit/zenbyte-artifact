"""exp-0013 Phase B analysis: teachers.csv + cascade.csv → verdicts for Q5–Q7,
applying the spec's decision rules mechanically."""
import csv
import json

import pandas as pd

from common13 import ART, TEACHER_SEEDS


def read_cascade():
    """cascade.csv has a different column count per role, because append_csv fixes
    the header from the first row's keys — so it is parsed positionally.

    gold-only       : seed, role, th, val_f1, precision, recall, f1
    student_svm     : seed, role, n_train, th, val_f1, precision, recall, f1
    student_fasttext: the above plus final_lr
    """
    rows = []
    with open(ART / "cascade.csv", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r[0] == "seed":
                continue
            if r[0] == "gold-only":
                rows.append({"seed": r[0], "role": r[1], "th": float(r[2]), "f1": float(r[6])})
            else:
                rows.append({"seed": r[0], "role": r[1], "th": float(r[3]), "f1": float(r[7])})
    return pd.DataFrame(rows)


def main():
    teachers = pd.read_csv(ART / "teachers.csv")
    cascade = read_cascade()
    assert list(teachers["seed"]) == TEACHER_SEEDS, "teacher seed count/order ≠ planned"

    gold = cascade[cascade["seed"] == "gold-only"].iloc[0]
    svm = cascade[cascade["role"] == "student_svm"].query("seed != 'gold-only'")
    ft = cascade[cascade["role"] == "student_fasttext"]
    assert len(svm) == len(ft) == len(TEACHER_SEEDS), "cascade row count ≠ planned"

    out = {"seeds": TEACHER_SEEDS, "gold_only_f1": round(float(gold["f1"]), 4)}

    tf1 = teachers["test_f1"]
    out["Q5_teacher_ceiling"] = {
        "originally_recorded": 0.853,
        "per_seed": {int(s): round(float(v), 4) for s, v in zip(teachers["seed"], tf1)},
        "mean": round(float(tf1.mean()), 4),
        "range": [round(float(tf1.min()), 4), round(float(tf1.max()), 4)],
        "verdict": "no verdict (the variance of a reference point is reported, not tested)"}

    gains = {int(s): round(float(f) - float(gold["f1"]), 4)
             for s, f in zip(svm["seed"], svm["f1"])}
    out["Q6_distill_gain_by_teacher"] = {
        "per_seed_gain": gains,
        "per_seed_f1": {int(s): round(float(f), 4) for s, f in zip(svm["seed"], svm["f1"])},
        "verdict": "supported" if all(g > 0 for g in gains.values()) else
                ("rejected" if all(g <= 0 for g in gains.values()) else "weakened")}

    svm_f1 = {int(s): float(f) for s, f in zip(svm["seed"], svm["f1"])}
    ft_f1 = {int(s): float(f) for s, f in zip(ft["seed"], ft["f1"])}
    margins = {s: round(svm_f1[s] - ft_f1[s], 4) for s in svm_f1}
    out["Q7_rank_by_teacher"] = {
        "svm": {s: round(v, 4) for s, v in svm_f1.items()},
        "fasttext": {s: round(v, 4) for s, v in ft_f1.items()},
        "margin": margins,
        "verdict": "supported" if all(m > 0 for m in margins.values()) else
                ("rejected" if all(m <= 0 for m in margins.values()) else "weakened")}

    # ── Auxiliary (outside the verdict; explicitly post-hoc descriptive statistics):
    #    per-seed sampling uncertainty.
    # A paired bootstrap, to show how the margin compares with test-sample noise.
    # The verdicts above were already decided by the pre-registered rule alone
    # (3/3 in the same direction); nothing in this section changes them.
    import numpy as np
    from common import load_split
    test = load_split("test")
    y = test["label"].to_numpy()
    rng = np.random.default_rng(20260718)
    idx = rng.integers(0, len(y), size=(10_000, len(y)), dtype=np.int32)

    def f1_boot(pred):
        tp = ((y == 1) & (pred == 1)).astype(np.int8)[idx].sum(axis=1).astype(float)
        fp = ((y == 0) & (pred == 1)).astype(np.int8)[idx].sum(axis=1).astype(float)
        fn = ((y == 1) & (pred == 0)).astype(np.int8)[idx].sum(axis=1).astype(float)
        d = 2 * tp + fp + fn
        return np.where(d > 0, 2 * tp / d, 0.0)

    def preds(tag, th):
        return (np.load(ART / f"{tag}_test_probs.npy") >= th).astype(np.int8)

    th_of = {(r["seed"], r["role"]): r["th"] for _, r in cascade.iterrows()}
    b_gold = f1_boot(preds("student_goldonly", th_of[("gold-only", "student_svm")]))
    aux = {}
    for s in TEACHER_SEEDS:
        b_svm = f1_boot(preds(f"student_seed{s}", th_of[(str(s), "student_svm")]))
        b_ft = f1_boot(preds(f"ft_seed{s}", th_of[(str(s), "student_fasttext")]))
        gain = b_svm - b_gold
        aux[s] = {"Q6_gain_ci95": [round(float(np.percentile(gain, 2.5)), 4),
                                    round(float(np.percentile(gain, 97.5)), 4)],
                  "Q7_svm_beats_ft_rate": round(float((b_svm > b_ft).mean()), 4)}
    out["auxiliary_paired_bootstrap_outside_verdict"] = aux

    # reproduction gate (seed 42, ±0.01 — exceeding it is recorded, not fatal)
    t42 = teachers[teachers["seed"] == 42].iloc[0]
    s42 = svm_f1[42]
    out["reproduction_gate_seed42"] = {
        "teacher_val_recorded_0.858": round(float(t42["val_f1"]), 4),
        "student_test_recorded_0.8034": round(s42, 4),
        "within_0.01": bool(abs(t42["val_f1"] - 0.858) <= 0.01 and abs(s42 - 0.8034) <= 0.01)}

    with open(ART / "analysis_b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
