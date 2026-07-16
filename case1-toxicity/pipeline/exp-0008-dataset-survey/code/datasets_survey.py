"""exp-0008 데이터셋 로더 + 조사표.

각 데이터셋을 (text, label) 이진 형태로 정규화한다. label 1 = 부적절(욕설/혐오/악플).
라이선스 불명 데이터셋은 로드하지 않는다.
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
DATA2 = EXP2 / "data"
DATA = Path(__file__).parent.parent / "data"


def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()


# ── 조사표 (라이선스는 2026-07-13 확인 기준) ──────────────────────────────
SURVEY = [
    # name, 규모, 라이선스, 라벨 정의, 도메인, 채택
    ("curse (2runo)", 5825, "MIT", "욕설 포함 여부", "온라인 댓글", "✅ 기존 사용"),
    ("hatescore", 11108, "Apache-2.0", "혐오발언/단순악플 vs Clean", "댓글+위키+규칙생성", "✅ 기존 사용"),
    ("unsmile", 18742, "CC-BY-NC-ND", "혐오 카테고리 9종 + 악플/욕설", "온라인 댓글", "⚠️ exp-0002서 역효과"),
    ("APEACH", 11666, "CC-BY-SA-4.0", "혐오표현 여부 (crowd-generated)", "크라우드 생성", "🔬 이번 평가"),
    ("kor-hate-sentence", 36661, "CC-BY-SA-4.0", "hate vs clean", "댓글 (unsmile/kmhas 계열 추정)", "🔬 이번 평가"),
    ("2tle/korean-curse", 1000, "MIT", "욕설 span 주석", "2runo와 동일 문장", "❌ 중복 (신규 데이터 아님)"),
    ("josephnam/korean_toxic", 8448, "불명", "LLM 유해 질의 (안전성)", "합성 프롬프트", "❌ 과제 불일치 + 라이선스 불명"),
    ("jigsaw (영어)", 160000, "CC0", "toxic 6종 멀티라벨", "위키 토론 (영어)", "🔬 참고 (언어 불일치)"),
]


def print_survey():
    df = pd.DataFrame(SURVEY, columns=["데이터셋", "규모", "라이선스", "라벨 정의", "도메인", "판정"])
    print(df.to_string(index=False))


# ── 로더 ──────────────────────────────────────────────────────────────
def load_ours(split: str) -> pd.DataFrame:
    """exp-0002의 병합 데이터 (curse + hatescore). 우리의 기준 분포."""
    return pd.read_csv(DATA2 / f"{split}.csv")[["text", "label"]]


def load_unsmile() -> pd.DataFrame:
    frames = [pd.read_csv(DATA2 / f, sep="\t") for f in
              ("unsmile_train_v1.0.tsv", "unsmile_valid_v1.0.tsv")]
    un = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"text": un["문장"].map(normalize), "label": (un["clean"] == 0).astype(int)})


def load_apeach() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("jason9693/APEACH")
    rows = []
    for split in ds:
        for r in ds[split]:
            rows.append((normalize(r["text"]), int(r["class"])))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_korhate() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("SJ-Donald/kor-hate-sentence")
    rows = []
    for split in ds:
        for r in ds[split]:
            # clean=1 이면 정상, hate=1 이면 혐오
            label = 0 if int(r["clean"]) == 1 else 1
            rows.append((normalize(r["문장"]), label))
    return pd.DataFrame(rows, columns=["text", "label"])


LOADERS = {
    "ours(curse+hatescore)": lambda: load_ours("train"),
    "unsmile": load_unsmile,
    "APEACH": load_apeach,
    "kor-hate-sentence": load_korhate,
}


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    ours_eval = set(load_ours("val")["text"]) | set(load_ours("test")["text"])
    for name, fn in LOADERS.items():
        df = fn()
        df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")
        if name != "ours(curse+hatescore)":
            # 우리 평가셋과 겹치는 문장 제거 (누수 방지)
            before = len(df)
            df = df[~df["text"].isin(ours_eval)]
            if before != len(df):
                print(f"  [{name}] 평가셋 누수 {before - len(df)}건 제거")
        out[name] = df.reset_index(drop=True)
        print(f"  [{name}] {len(df)}건, 양성 {df['label'].mean():.3f}")
    return out


if __name__ == "__main__":
    print("=== exp-0008 데이터셋 조사표 ===")
    print_survey()
    print("\n=== 로드 결과 ===")
    load_all()
