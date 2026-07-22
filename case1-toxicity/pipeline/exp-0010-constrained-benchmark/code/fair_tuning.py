"""exp-0010 fair re-tuning: every candidate is re-tuned on the distilled data
(38k) under the same budget.

The problem: the fastText configuration (OP) used in the first benchmark was
**tuned on the gold data (11.8k)**. Distillation tripled the data, so a larger
capacity may now be optimal for fastText too, and a comparison in which only
one side is tuned is unfair (the fairness rule in the spec).

→ Both candidates are given the same search budget (12 configurations each),
   their optima are found on val, and those optima are then compared on test.
   Any configuration above the 15MB size budget is disqualified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from common import f1_binary, int8_serialized_bytes, load_split, train_with_retry  # noqa: E402
from transfer_matrix import prob_pos, write_ft  # noqa: E402
from benchmark import build_train_sets, pick_threshold, sklearn_size_bytes, svm_decision  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
BUDGET_MB = 15.0


def main():
    tr = build_train_sets()["gold+distilled"]
    val, test = load_split("val"), load_split("test")
    yv, yt = val["label"].tolist(), test["label"].tolist()

    print("=== fastText re-tuning (distilled data, 38k, on val) — 12 configurations ===")
    ft_rows = []
    for dim in (16, 32):
        for bucket in (250_000, 500_000, 1_000_000):
            for minmax in ((2, 4), (2, 5)):
                cfg = {"dim": dim, "bucket": bucket, "minn": minmax[0], "maxn": minmax[1], "lr": 0.125}
                model, _ = train_with_retry(input=write_ft(tr, DATA / "ft_tune.txt"),
                                            wordNgrams=2, epoch=25, loss="softmax",
                                            thread=1, verbose=0, **cfg)
                pv = prob_pos(model, val["text"])
                th, f1v = pick_threshold(yv, pv)
                mb = int8_serialized_bytes(model) / 2**20
                ok = mb <= BUDGET_MB
                ft_rows.append({**cfg, "val_f1": round(f1v, 4), "th": round(float(th), 3),
                                "size_mb": round(mb, 2), "within_budget": ok, "_model": model})
                print(f"  dim{dim} bucket{bucket//1000}k n{minmax} → val {f1v:.4f}, {mb:.2f}MB"
                      f"{'' if ok else '  ❌ over budget'}")

    print("\n=== TF-IDF(char)+SVM re-tuning (same budget, 12 configurations) ===")
    svm_rows = []
    for ngram in ((2, 4), (2, 5), (2, 6)):
        for max_feat in (200_000, 500_000):
            for C in (0.5, 1.0):
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2,
                                      sublinear_tf=True, max_features=max_feat)
                clf = LinearSVC(C=C, max_iter=5000)
                X = vec.fit_transform(tr["text"])
                clf.fit(X, tr["label"])
                pv = svm_decision(clf, vec.transform(val["text"]))
                th, f1v = pick_threshold(yv, pv)
                mb = sklearn_size_bytes(vec, clf) / 2**20
                ok = mb <= BUDGET_MB
                svm_rows.append({"ngram": ngram, "max_feat": max_feat, "C": C,
                                 "val_f1": round(f1v, 4), "th": round(float(th), 3),
                                 "size_mb": round(mb, 2), "within_budget": ok,
                                 "_vec": vec, "_clf": clf})
                print(f"  ngram{ngram} feat{max_feat//1000}k C{C} → val {f1v:.4f}, {mb:.2f}MB"
                      f"{'' if ok else '  ❌ over budget'}")

    # compare the in-budget optima on test
    ft_best = max((r for r in ft_rows if r["within_budget"]), key=lambda r: r["val_f1"])
    svm_best = max((r for r in svm_rows if r["within_budget"]), key=lambda r: r["val_f1"])

    print("\n=== in-budget optima compared on test (threshold chosen on val) ===")
    pt = prob_pos(ft_best["_model"], test["text"])
    m_ft = f1_binary(yt, (pt >= ft_best["th"]).astype(int).tolist())
    print(f"  fastText  dim{ft_best['dim']} bucket{ft_best['bucket']} n({ft_best['minn']},{ft_best['maxn']}): "
          f"val {ft_best['val_f1']:.4f} → test F1 {m_ft['f1']:.4f} "
          f"(P {m_ft['precision']:.4f} R {m_ft['recall']:.4f}), {ft_best['size_mb']}MB")

    pt = svm_decision(svm_best["_clf"], svm_best["_vec"].transform(test["text"]))
    m_svm = f1_binary(yt, (pt >= svm_best["th"]).astype(int).tolist())
    print(f"  SVM       ngram{svm_best['ngram']} feat{svm_best['max_feat']} C{svm_best['C']}: "
          f"val {svm_best['val_f1']:.4f} → test F1 {m_svm['f1']:.4f} "
          f"(P {m_svm['precision']:.4f} R {m_svm['recall']:.4f}), {svm_best['size_mb']}MB")

    print(f"\n  difference: SVM - fastText = {m_svm['f1'] - m_ft['f1']:+.4f} F1, "
          f"{svm_best['size_mb'] - ft_best['size_mb']:+.2f}MB")

    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in ft_rows]) \
        .to_csv(Path(__file__).parent.parent / "artifacts" / "ft_tuning.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in svm_rows]) \
        .to_csv(Path(__file__).parent.parent / "artifacts" / "svm_tuning.csv", index=False)


if __name__ == "__main__":
    main()
