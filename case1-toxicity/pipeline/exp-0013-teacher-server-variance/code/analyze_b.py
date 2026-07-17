"""exp-0013 Phase B 분석: teachers.csv + cascade.csv → Q5~Q7 판정 (spec 기준 기계 적용)."""
import csv
import json

import pandas as pd

from common13 import ART, TEACHER_SEEDS


def read_cascade():
    """cascade.csv는 역할별로 열 수가 다르다(append_csv가 첫 행 키로 헤더 고정) — 위치 기반 파싱.

    gold-only     : seed, role, th, val_f1, precision, recall, f1
    student_svm   : seed, role, n_train, th, val_f1, precision, recall, f1
    student_fasttext: 위 + final_lr
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
    assert list(teachers["seed"]) == TEACHER_SEEDS, "교사 seed 수/순서 ≠ 계획"

    gold = cascade[cascade["seed"] == "gold-only"].iloc[0]
    svm = cascade[cascade["role"] == "student_svm"].query("seed != 'gold-only'")
    ft = cascade[cascade["role"] == "student_fasttext"]
    assert len(svm) == len(ft) == len(TEACHER_SEEDS), "캐스케이드 행 수 ≠ 계획"

    out = {"seeds": TEACHER_SEEDS, "gold_only_f1": round(float(gold["f1"]), 4)}

    tf1 = teachers["test_f1"]
    out["Q5_teacher_ceiling"] = {
        "원기록": 0.853,
        "per_seed": {int(s): round(float(v), 4) for s, v in zip(teachers["seed"], tf1)},
        "mean": round(float(tf1.mean()), 4),
        "range": [round(float(tf1.min()), 4), round(float(tf1.max()), 4)],
        "판정": "판정 없음 (참조선 분산 보고)"}

    gains = {int(s): round(float(f) - float(gold["f1"]), 4)
             for s, f in zip(svm["seed"], svm["f1"])}
    out["Q6_distill_gain_by_teacher"] = {
        "per_seed_gain": gains,
        "per_seed_f1": {int(s): round(float(f), 4) for s, f in zip(svm["seed"], svm["f1"])},
        "판정": "유지" if all(g > 0 for g in gains.values()) else
                ("기각" if all(g <= 0 for g in gains.values()) else "약화")}

    svm_f1 = {int(s): float(f) for s, f in zip(svm["seed"], svm["f1"])}
    ft_f1 = {int(s): float(f) for s, f in zip(ft["seed"], ft["f1"])}
    margins = {s: round(svm_f1[s] - ft_f1[s], 4) for s in svm_f1}
    out["Q7_rank_by_teacher"] = {
        "svm": {s: round(v, 4) for s, v in svm_f1.items()},
        "fasttext": {s: round(v, 4) for s, v in ft_f1.items()},
        "margin": margins,
        "판정": "유지" if all(m > 0 for m in margins.values()) else
                ("기각" if all(m <= 0 for m in margins.values()) else "약화")}

    # ── 보조 (판정 외, 사후 기술 통계임을 명시): seed별 표본 불확실성 ──
    # margin이 test 표본 잡음 대비 어느 수준인지 보이기 위한 paired bootstrap.
    # 판정은 위의 사전 등록 기준(3/3 방향)만으로 이미 내려졌고, 이 절은 판정을 바꾸지 않는다.
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
        aux[s] = {"Q6 이득 CI95": [round(float(np.percentile(gain, 2.5)), 4),
                                    round(float(np.percentile(gain, 97.5)), 4)],
                  "Q7 SVM>ft 우세율": round(float((b_svm > b_ft).mean()), 4)}
    out["보조_paired_bootstrap(판정외)"] = aux

    # 재현 게이트 (seed 42, ±0.01 — 초과 시 중단 아닌 기록)
    t42 = teachers[teachers["seed"] == 42].iloc[0]
    s42 = svm_f1[42]
    out["재현_게이트_seed42"] = {
        "교사 val (기록 0.858)": round(float(t42["val_f1"]), 4),
        "학생 test (기록 0.8034)": round(s42, 4),
        "±0.01 내": bool(abs(t42["val_f1"] - 0.858) <= 0.01 and abs(s42 - 0.8034) <= 0.01)}

    with open(ART / "analysis_b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
