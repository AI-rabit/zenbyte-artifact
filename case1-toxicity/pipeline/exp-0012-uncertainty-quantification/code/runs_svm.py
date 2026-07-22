"""exp-0012 Phase A (1/3): the Q1 gate plus the SVM-side training (deterministic,
one run each).

Q1: probabilities for every test row from the deployed int8 ZBSV → check that
    F1@0.475 reproduces the 0.8050 recorded in exp-0011, within ±0.0005. On
    failure the script stops immediately — the spec forbids continuing before
    the cause is understood.
Q2: the fairly re-tuned SVM (char_wb(2,4)/500k/C=0.5, distilled 38,019) →
    pick_threshold on val → test probabilities.
Q3: the first-benchmark SVM (char_wb(2,5)/500k/C=0.5) — gold and gold+distilled,
    one run each → test probabilities.

Every probability vector, threshold and metric is saved under artifacts/. The
analysis in analyze.py works only from those saved outputs.
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from common12 import (ART, EXP11_ART, SVC_KW, SVM_Q2_VEC, SVM_Q3_VEC,
                      ensure_dirs)

from benchmark import build_train_sets, pick_threshold, svm_decision  # noqa: E402
from common import f1_binary, load_split  # noqa: E402
from reference_svm import ZBSVModel  # noqa: E402

GATE_F1 = 0.8050
GATE_TOL = 0.0005


def q1_gate(test, results):
    model = ZBSVModel(EXP11_ART / "toxicity_model.zbsv")
    th = float(getattr(model, "threshold", 0.475))
    probs = np.array([model.prob_toxic(t) for t in test["text"]])
    m = f1_binary(test["label"].tolist(), (probs >= th).astype(int).tolist())
    np.save(ART / "q1_int8_test_probs.npy", probs)
    results["q1"] = {"th": th, "n": len(probs), **{k: round(v, 6) for k, v in m.items()}}
    print(f"[Q1] int8 ZBSV test: F1={m['f1']:.4f} (P={m['precision']:.4f} "
          f"R={m['recall']:.4f}) @ th={th}")
    delta = abs(m["f1"] - GATE_F1)
    if delta > GATE_TOL:
        raise SystemExit(f"[Q1 gate failed] |{m['f1']:.4f} - {GATE_F1}| = {delta:.4f} > "
                         f"{GATE_TOL} — stopping before the cause is understood (spec)")
    print(f"[Q1] gate passed (Δ={delta:.4f})")


def fit_svm(vec_kw, tr, val, test, tag, results):
    vec = TfidfVectorizer(**vec_kw)
    clf = LinearSVC(**SVC_KW)
    clf.fit(vec.fit_transform(tr["text"]), tr["label"])
    pv = svm_decision(clf, vec.transform(val["text"]))
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = svm_decision(clf, vec.transform(test["text"]))
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())
    np.save(ART / f"{tag}_test_probs.npy", pt)
    results[tag] = {"n_train": len(tr), "vocab": len(vec.vocabulary_),
                    "th": round(float(th), 3), "val_f1": round(val_f1, 4),
                    **{k: round(v, 6) for k, v in m.items()}}
    print(f"[{tag}] train={len(tr)} th={th:.3f} val={val_f1:.4f} test F1={m['f1']:.4f}")


def main():
    ensure_dirs()
    sets = build_train_sets()
    gold, dist = sets["gold only"], sets["gold+distilled"]
    val, test = load_split("val"), load_split("test")
    results = {}

    q1_gate(test, results)
    fit_svm(SVM_Q2_VEC, dist, val, test, "q2_svm", results)          # expected ≈ 0.8034
    fit_svm(SVM_Q3_VEC, gold, val, test, "q3_svm_gold", results)     # expected ≈ 0.732
    fit_svm(SVM_Q3_VEC, dist, val, test, "q3_svm_dist", results)     # expected ≈ 0.805

    with open(ART / "svm_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("→ saved artifacts/svm_results.json")


if __name__ == "__main__":
    main()
