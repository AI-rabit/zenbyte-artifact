"""exp-0002 공용: 평가 지표, int8 직렬화 크기 산정, 자모 분해, NaN 재시도 학습.

주의: 이 fastText 빌드는 seed 파라미터가 없어 학습이 확률적이다 (thread=1이어도).
따라서 (1) NaN 발산은 재시도로 흡수하고, (2) 최종 후보 성능은 반복 실행 평균으로 보고한다.
"""
from pathlib import Path

import fasttext
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# 한글 음절 → 자모 분해 (초성/중성/종성)
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
    """Kotlin 반입용 int8 직렬화 크기 산정 (실측 기반).

    입력 행렬 (nwords+bucket, dim) int8 + 행별 scale float32
    + 출력 행렬 (n_labels, dim) float32 + 어휘 문자열.
    """
    inp = model.get_input_matrix()          # (nwords + bucket, dim) float32
    out = model.get_output_matrix()         # (n_labels, dim) float32
    vocab_bytes = sum(len(w.encode("utf-8")) + 1 for w in model.get_words())
    return inp.size * 1 + inp.shape[0] * 4 + out.size * 4 + vocab_bytes


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def train_with_retry(max_attempts: int = 6, **kwargs):
    """NaN 발산 시 동일 lr로 재시도(확률적) → 2회 실패마다 lr 반감. 최종 lr 반환."""
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
