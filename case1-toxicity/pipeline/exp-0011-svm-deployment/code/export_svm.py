"""exp-0011 1단계: TF-IDF(char_wb) + LinearSVM 최종 학습 → ZBSV 포맷 내보내기.

동작점 (exp-0010 공정 재튜닝 결과):
  TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=2,
                  sublinear_tf=True, max_features=500_000)
  LinearSVC(C=0.5)
  학습 데이터: ours(골드 11,849) + 교사 의사라벨(신뢰도 ≥0.9, 26,170) = 38,019건

ZBSV 포맷 v1 (little-endian):
  magic   4B  'ZBSV'
  version u32 = 1
  minN, maxN, nTerms : u32 ×3
  sublinearTf u8, useIdf u8, pad u16
  threshold  f32          (val에서 선택된 판정 임계값, 시그모이드 확률 기준)
  intercept  f32
  coefScale  f32          (int8 역양자화 스케일)
  idf        f32 × nTerms
  coef       i8  × nTerms
  vocab      nTerms × { u16 byteLen + utf8 bytes }   # 인덱스 순서
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
EXP10 = Path(__file__).parent.parent.parent / "exp-0010-constrained-benchmark"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))
sys.path.insert(0, str(EXP10 / "code"))

from benchmark import build_train_sets, pick_threshold, svm_decision  # noqa: E402
from common import f1_binary, load_split  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"

VEC_KW = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
              sublinear_tf=True, max_features=500_000)
SVC_KW = dict(C=0.5, max_iter=5000)


def train():
    tr = build_train_sets()["골드+증류"]
    vec = TfidfVectorizer(**VEC_KW)
    clf = LinearSVC(**SVC_KW)
    X = vec.fit_transform(tr["text"])
    clf.fit(X, tr["label"])
    print(f"학습: {len(tr)}건, 어휘 {len(vec.vocabulary_)} n-gram")
    return vec, clf


def main():
    ART.mkdir(exist_ok=True)
    vec, clf = train()
    val, test = load_split("val"), load_split("test")

    pv = svm_decision(clf, vec.transform(val["text"]))
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = svm_decision(clf, vec.transform(test["text"]))
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())
    print(f"val F1 = {val_f1:.4f} (th={th:.3f})")
    print(f"test F1 = {m['f1']:.4f} (P={m['precision']:.4f} R={m['recall']:.4f})")

    # ── ZBSV 직렬화 ──────────────────────────────────────────────
    vocab = vec.vocabulary_                       # term → index
    n = len(vocab)
    terms = [None] * n
    for t, i in vocab.items():
        terms[i] = t
    idf = vec.idf_.astype(np.float32)
    coef = clf.coef_.ravel().astype(np.float32)
    scale = float(np.abs(coef).max() / 127.0) or 1.0
    q = np.clip(np.round(coef / scale), -127, 127).astype(np.int8)

    path = ART / "toxicity_model.zbsv"
    with open(path, "wb") as f:
        f.write(b"ZBSV")
        f.write(struct.pack("<4I", 1, VEC_KW["ngram_range"][0], VEC_KW["ngram_range"][1], n))
        f.write(struct.pack("<BBH", 1 if VEC_KW["sublinear_tf"] else 0, 1, 0))
        f.write(struct.pack("<3f", float(th), float(clf.intercept_[0]), scale))
        f.write(idf.tobytes())
        f.write(q.tobytes())
        for t in terms:
            b = t.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    size = path.stat().st_size
    meta = {"nTerms": n, "minN": VEC_KW["ngram_range"][0], "maxN": VEC_KW["ngram_range"][1],
            "threshold": round(float(th), 4), "intercept": float(clf.intercept_[0]),
            "coefScale": scale, "bytes": size, "mb": round(size / 2**20, 2),
            "val_f1": round(val_f1, 4), "test_f1": round(m["f1"], 4)}
    (ART / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    # 참조 검증용으로 sklearn 확률도 저장
    np.save(ART / "sklearn_test_probs.npy", pt)


if __name__ == "__main__":
    main()
