"""exp-0002 shared helpers: metrics, int8 serialized-size estimation, jamo
decomposition, and training with retries on NaN.

Note: this fastText build exposes no seed parameter, so training is stochastic
even at thread=1. Consequently (1) NaN divergence is absorbed by retrying, and
(2) the performance of a final candidate is reported as a mean over repeated
runs.
"""
from pathlib import Path

import fasttext
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# Hangul syllable → jamo decomposition (onset / nucleus / coda)
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def jamo_decompose(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_CHO[code // 588])
            out.append(_JUNG[(code % 588) // 28])
            jong = _JONG[code % 28]
            if jong:
                out.append(jong)
        else:
            out.append(ch)
    return "".join(out)


def f1_binary(y_true, y_pred, positive=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def int8_serialized_bytes(model) -> int:
    """Estimate the int8 serialized size for porting to Kotlin (measured, not guessed).

    Input matrix (nwords+bucket, dim) as int8 + a per-row float32 scale
    + output matrix (n_labels, dim) as float32 + the vocabulary strings.
    """
    inp = model.get_input_matrix()          # (nwords + bucket, dim) float32
    out = model.get_output_matrix()         # (n_labels, dim) float32
    vocab_bytes = sum(len(w.encode("utf-8")) + 1 for w in model.get_words())
    return inp.size * 1 + inp.shape[0] * 4 + out.size * 4 + vocab_bytes


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def train_with_retry(max_attempts: int = 6, **kwargs):
    """On NaN divergence, retry at the same lr (training is stochastic), halving lr every second failure. Returns the final lr."""
    lr = kwargs.pop("lr", 0.5)
    last = None
    for attempt in range(max_attempts):
        try:
            model = fasttext.train_supervised(lr=lr, **kwargs)
            return model, lr
        except RuntimeError as e:
            last = e
            if attempt % 2 == 1:
                lr /= 2
    raise RuntimeError(f"NaN persisted after {max_attempts} attempts (final lr={lr}): {last}")
