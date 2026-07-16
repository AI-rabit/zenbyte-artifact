"""exp-0010 제약 조건 하 정량 벤치마크.

후보 6종 × 학습조건 2종(골드만 / 골드+증류) × (F1, 크기, 지연)을 동일 조건에서 측정한다.

공정성 규칙:
  - 모든 후보에 동일 train/val/test, 동일 증류 데이터.
  - 하이퍼파라미터는 후보마다 val에서 동일한 예산으로 탐색 (한쪽만 튜닝하지 않는다).
  - 임계값은 val에서 선택 → test에 1회 적용.
  - 크기는 **동일한 int8 직렬화 규칙**으로 실측 (이론값 금지).
  - 지연은 같은 머신·같은 문장·같은 반복 수로 측정.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
EXP9 = Path(__file__).parent.parent.parent / "exp-0009-distillation"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from baseline_keyword import LEXICON  # noqa: E402
from baseline_keyword import predict as kw_predict  # noqa: E402
from common import f1_binary, int8_serialized_bytes, load_split, train_with_retry  # noqa: E402
from transfer_matrix import OP, prob_pos, write_ft  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"
DATA = Path(__file__).parent.parent / "data"
CONF = 0.9
LATENCY_N = 1000


def build_train_sets():
    ours = load_split("train")[["text", "label"]]
    pool = pd.read_csv(EXP9 / "artifacts" / "pseudo_labels.csv")
    conf = pool[(pool["teacher_prob"] >= CONF) | (pool["teacher_prob"] <= 1 - CONF)]
    pseudo = pd.DataFrame({"text": conf["text"], "label": (conf["teacher_prob"] >= 0.5).astype(int)})
    distilled = pd.concat([ours, pseudo], ignore_index=True)
    return {"골드만": ours, "골드+증류": distilled}


def pick_threshold(y, p):
    return max(((t, f1_binary(y, (p >= t).astype(int).tolist())["f1"])
                for t in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1])


def sklearn_size_bytes(vec, clf) -> int:
    """배포 시 실제로 반입해야 하는 바이트 수 (fastText와 동일한 int8 규칙 적용).

    - 어휘 문자열 (UTF-8 + 길이 prefix)
    - idf 벡터: float32
    - 계수: int8 + scale 1개 (행 1개짜리 선형 모델)
    """
    vocab = vec.vocabulary_
    vocab_bytes = sum(len(t.encode("utf-8")) + 2 for t in vocab)
    idf_bytes = len(vocab) * 4
    # NB는 coef_ 대신 클래스별 로그확률(feature_log_prob_)을 반입해야 한다
    weights = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
    coef_bytes = np.asarray(weights).size * 1 + 4 * np.asarray(weights).shape[0]  # int8 + 행별 scale
    return vocab_bytes + idf_bytes + coef_bytes + 4  # + intercept


def eval_sklearn(name, vec, clf, tr, val, test, decision_fn):
    X_tr = vec.fit_transform(tr["text"])
    clf.fit(X_tr, tr["label"])
    pv = decision_fn(clf, vec.transform(val["text"]))
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = decision_fn(clf, vec.transform(test["text"]))
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())

    # 지연 (문장 1건씩 — 실사용 시나리오)
    texts = test["text"].tolist()[:200]
    for t in texts[:20]:
        decision_fn(clf, vec.transform([t]))  # warm-up
    times = []
    for i in range(LATENCY_N):
        t0 = time.perf_counter()
        decision_fn(clf, vec.transform([texts[i % len(texts)]]))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"model": name, "val_f1": round(val_f1, 4), "th": round(float(th), 3),
            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4), "size_mb": round(sklearn_size_bytes(vec, clf) / 2**20, 2),
            "p50_ms": round(times[len(times)//2], 3), "p95_ms": round(times[int(len(times)*0.95)], 3),
            "vocab": len(vec.vocabulary_)}


def lr_proba(clf, X):
    return clf.predict_proba(X)[:, 1]


def svm_decision(clf, X):
    d = clf.decision_function(X)
    return 1 / (1 + np.exp(-d))  # 시그모이드로 [0,1] 사상 (임계값 튜닝을 위해)


def eval_fasttext(tr, val, test):
    path = write_ft(tr, DATA / "ft_bench.txt")
    model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                loss="softmax", thread=1, verbose=0, **OP)
    pv = prob_pos(model, val["text"])
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = prob_pos(model, test["text"])
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())

    texts = test["text"].tolist()[:200]
    for t in texts[:20]:
        model.predict([t], k=2)  # 단일 문자열 predict는 numpy2 비호환 → 리스트 경로 사용
    times = []
    for i in range(LATENCY_N):
        t0 = time.perf_counter()
        model.predict([texts[i % len(texts)]], k=2)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"model": "fastText", "val_f1": round(val_f1, 4), "th": round(float(th), 3),
            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4), "size_mb": round(int8_serialized_bytes(model) / 2**20, 2),
            "p50_ms": round(times[len(times)//2], 3), "p95_ms": round(times[int(len(times)*0.95)], 3),
            "vocab": len(model.get_words())}


def eval_keyword(test):
    y, pred = test["label"].tolist(), [kw_predict(t) for t in test["text"]]
    m = f1_binary(y, pred)
    texts = test["text"].tolist()[:200]
    times = []
    for i in range(LATENCY_N):
        t0 = time.perf_counter()
        kw_predict(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    size = sum(len(w.encode()) + 2 for w in LEXICON)
    return {"model": "키워드 사전", "val_f1": None, "th": None,
            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4), "size_mb": round(size / 2**20, 4),
            "p50_ms": round(times[len(times)//2], 3), "p95_ms": round(times[int(len(times)*0.95)], 3),
            "vocab": len(LEXICON)}


def main():
    ART.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    val, test = load_split("val"), load_split("test")
    train_sets = build_train_sets()

    rows = []
    kw = eval_keyword(test)
    for cond in train_sets:
        rows.append({"조건": "—", **kw})
        break  # 키워드는 학습이 없으므로 1회만

    for cond, tr in train_sets.items():
        print(f"\n=== 학습 조건: {cond} (n={len(tr)}) ===")

        # TF-IDF(word) + LR
        r = eval_sklearn("TF-IDF(word)+LR",
                         TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=1.0), tr, val, test, lr_proba)
        rows.append({"조건": cond, **r}); print(" ", r)

        # TF-IDF(char) + LR  ← fastText subword의 직접 경쟁자
        r = eval_sklearn("TF-IDF(char2-5)+LR",
                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=2, sublinear_tf=True, max_features=500_000),
                         LogisticRegression(max_iter=2000, C=1.0), tr, val, test, lr_proba)
        rows.append({"조건": cond, **r}); print(" ", r)

        # TF-IDF(char) + LinearSVM
        r = eval_sklearn("TF-IDF(char2-5)+SVM",
                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=2, sublinear_tf=True, max_features=500_000),
                         LinearSVC(C=0.5, max_iter=5000), tr, val, test, svm_decision)
        rows.append({"조건": cond, **r}); print(" ", r)

        # Naive Bayes (char)
        r = eval_sklearn("TF-IDF(char2-5)+NB",
                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=2, sublinear_tf=True, max_features=500_000),
                         MultinomialNB(alpha=0.1), tr, val, test, lr_proba)
        rows.append({"조건": cond, **r}); print(" ", r)

        # fastText
        r = eval_fasttext(tr, val, test)
        rows.append({"조건": cond, **r}); print(" ", r)

    df = pd.DataFrame(rows)
    df.to_csv(ART / "benchmark.csv", index=False)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
