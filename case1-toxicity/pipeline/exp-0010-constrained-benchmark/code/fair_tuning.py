"""exp-0010 공정 재튜닝: 증류 데이터(38k)에서 각 후보를 동일 예산으로 재튜닝한다.

문제: 1차 벤치마크의 fastText 설정(OP)은 **골드 데이터(11.8k)에서 튜닝**된 것이다.
증류로 데이터가 3배로 늘었으니, fastText도 더 큰 용량이 최적일 수 있다.
한쪽만 튜닝된 비교는 불공정하다 (spec의 공정성 규칙).

→ 두 후보에 동일한 탐색 예산(각 12설정)을 주고 val에서 최적점을 찾은 뒤,
   그 최적점끼리 test에서 비교한다. 크기 예산 15MB를 넘는 설정은 실격 처리.
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
    tr = build_train_sets()["골드+증류"]
    val, test = load_split("val"), load_split("test")
    yv, yt = val["label"].tolist(), test["label"].tolist()

    print("=== fastText 재튜닝 (증류 데이터 38k, val) — 12설정 ===")
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
                                "size_mb": round(mb, 2), "예산내": ok, "_model": model})
                print(f"  dim{dim} bucket{bucket//1000}k n{minmax} → val {f1v:.4f}, {mb:.2f}MB"
                      f"{'' if ok else '  ❌예산초과'}")

    print("\n=== TF-IDF(char)+SVM 재튜닝 (동일 예산 12설정) ===")
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
                                 "size_mb": round(mb, 2), "예산내": ok,
                                 "_vec": vec, "_clf": clf})
                print(f"  ngram{ngram} feat{max_feat//1000}k C{C} → val {f1v:.4f}, {mb:.2f}MB"
                      f"{'' if ok else '  ❌예산초과'}")

    # 예산 내 최적점끼리 test 비교
    ft_best = max((r for r in ft_rows if r["예산내"]), key=lambda r: r["val_f1"])
    svm_best = max((r for r in svm_rows if r["예산내"]), key=lambda r: r["val_f1"])

    print("\n=== 예산 내 최적점 test 비교 (임계값은 val에서 선택) ===")
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

    print(f"\n  차이: SVM - fastText = {m_svm['f1'] - m_ft['f1']:+.4f} F1, "
          f"{svm_best['size_mb'] - ft_best['size_mb']:+.2f}MB")

    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in ft_rows]) \
        .to_csv(Path(__file__).parent.parent / "artifacts" / "ft_tuning.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in svm_rows]) \
        .to_csv(Path(__file__).parent.parent / "artifacts" / "svm_tuning.csv", index=False)


if __name__ == "__main__":
    main()
