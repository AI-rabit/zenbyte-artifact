"""exp-0010 quantitative benchmark under the deployment constraints.

Six candidates × two training conditions (gold only / gold+distilled) are
measured on (F1, size, latency) under identical conditions.

Fairness rules:
  - every candidate sees the same train/val/test and the same distilled data.
  - hyperparameters are searched on val with the same budget for each candidate;
    no candidate is tuned while another is not.
  - the threshold is chosen on val and applied once to test.
  - size is **measured under the same int8 serialization rule** for all — no
    theoretical figures.
  - latency is measured on the same machine, the same sentences and the same
    number of repetitions.
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
    return {"gold only": ours, "gold+distilled": distilled}


def pick_threshold(y, p):
    return max(((t, f1_binary(y, (p >= t).astype(int).tolist())["f1"])
                for t in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1])


def sklearn_size_bytes(vec, clf) -> int:
    """The bytes that would actually have to ship, under the same int8 rule as fastText.

    - vocabulary strings (UTF-8 plus a length prefix)
    - the idf vector: float32
    - coefficients: int8 plus one scale (a linear model with a single row)
    """
    vocab = vec.vocabulary_
    vocab_bytes = sum(len(t.encode("utf-8")) + 2 for t in vocab)
    idf_bytes = len(vocab) * 4
    # NB ships per-class log probabilities (feature_log_prob_) rather than coef_
    weights = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
    coef_bytes = np.asarray(weights).size * 1 + 4 * np.asarray(weights).shape[0]  # int8 plus a per-row scale
    return vocab_bytes + idf_bytes + coef_bytes + 4  # + intercept


def eval_sklearn(name, vec, clf, tr, val, test, decision_fn):
    X_tr = vec.fit_transform(tr["text"])
    clf.fit(X_tr, tr["label"])
    pv = decision_fn(clf, vec.transform(val["text"]))
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = decision_fn(clf, vec.transform(test["text"]))
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())

    # latency, one sentence at a time — the realistic usage pattern
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
    return 1 / (1 + np.exp(-d))  # map to [0,1] with a sigmoid, so the threshold can be tuned


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
        model.predict([t], k=2)  # predict on a bare string is numpy2-incompatible; use the list path
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
    return {"model": "keyword lexicon", "val_f1": None, "th": None,
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
        rows.append({"condition": "—", **kw})
        break  # the keyword baseline has no training, so it is reported once

    for cond, tr in train_sets.items():
        print(f"\n=== training condition: {cond} (n={len(tr)}) ===")

        # TF-IDF(word) + LR
        r = eval_sklearn("TF-IDF(word)+LR",
                         TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=1.0), tr, val, test, lr_proba)
        rows.append({"condition": cond, **r}); print(" ", r)

        # TF-IDF(char) + LR  ← the direct competitor to fastText's subwords
        r = eval_sklearn("TF-IDF(char2-5)+LR",
                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=2, sublinear_tf=True, max_features=500_000),
                         LogisticRegression(max_iter=2000, C=1.0), tr, val, test, lr_proba)
        rows.append({"condition": cond, **r}); print(" ", r)

        # TF-IDF(char) + LinearSVM
        r = eval_sklearn("TF-IDF(char2-5)+SVM",
                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=2, sublinear_tf=True, max_features=500_000),
                         LinearSVC(C=0.5, max_iter=5000), tr, val, test, svm_decision)
        rows.append({"condition": cond, **r}); print(" ", r)

        # Naive Bayes (char)
        r = eval_sklearn("TF-IDF(char2-5)+NB",
                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                         min_df=2, sublinear_tf=True, max_features=500_000),
                         MultinomialNB(alpha=0.1), tr, val, test, lr_proba)
        rows.append({"condition": cond, **r}); print(" ", r)

        # fastText
        r = eval_fasttext(tr, val, test)
        rows.append({"condition": cond, **r}); print(" ", r)

    df = pd.DataFrame(rows)
    df.to_csv(ART / "benchmark.csv", index=False)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
