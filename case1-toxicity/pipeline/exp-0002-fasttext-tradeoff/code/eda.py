"""exp-0002 EDA: 데이터셋 2종의 분포·중복·난독화 패턴 확인.

실행: .venv-research 활성화 후 `python eda.py`
입력: ../data/curse_detection.txt (2runo, MIT), ../data/hatescore.csv (Apache-2.0)
출력: stdout 리포트 (log.md에 기록용)
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data"


def load_curse() -> pd.DataFrame:
    rows, bad = [], 0
    for line in (DATA / "curse_detection.txt").read_text(encoding="utf-8").splitlines():
        # 포맷: 문장|라벨  (문장 내 '|' 가능성 → rsplit)
        if "|" not in line:
            bad += 1
            continue
        text, _, label = line.rpartition("|")
        label = label.strip()
        if label not in ("0", "1"):
            bad += 1
            continue
        rows.append((text.strip(), int(label)))
    print(f"[curse] 파싱 실패 라인: {bad}")
    return pd.DataFrame(rows, columns=["text", "label"])


def load_hatescore() -> pd.DataFrame:
    df = pd.read_csv(DATA / "hatescore.csv", index_col=0)
    print(f"[hatescore] 컬럼: {list(df.columns)}")
    print(f"[hatescore] macrolabel 분포:\n{df['macrolabel'].value_counts()}")
    print(f"[hatescore] microlabel 상위:\n{df['microlabel'].value_counts().head(12)}")
    print(f"[hatescore] source 분포:\n{df['source'].value_counts()}")
    # 이진화: 혐오발언 계열 → 1, 그 외(일반문장 등) → 0  (정확한 매핑은 분포 확인 후 확정)
    return df


def report(df: pd.DataFrame, name: str):
    print(f"\n===== {name} =====")
    print(f"행 수: {len(df)}")
    print(f"라벨 분포:\n{df['label'].value_counts(normalize=True).round(3)}")
    dup = df.duplicated(subset="text").sum()
    print(f"완전 중복 문장: {dup}")
    df["len"] = df["text"].str.len()
    print(f"길이: median={df['len'].median():.0f}, p95={df['len'].quantile(0.95):.0f}, max={df['len'].max()}")
    empty = (df["text"].str.strip() == "").sum()
    print(f"빈 문장: {empty}")
    # 난독화 패턴 표본
    obfus = {
        "숫자 삽입형 (시1발 등)": r"[가-힣]\d[가-힣]",
        "자모 단독형 (ㅅㅂ, ㅄ 등)": r"[ㄱ-ㅎㅏ-ㅣ]{2,}",
        "특수문자 삽입형": r"[가-힣][@#$%^&*~\-_.]+[가-힣]",
    }
    pos = df[df["label"] == 1] if "label" in df else df
    for desc, pat in obfus.items():
        hits = pos["text"].str.contains(pat, regex=True).sum()
        print(f"난독화 [{desc}]: label=1 중 {hits}건 ({hits/max(len(pos),1)*100:.1f}%)")


if __name__ == "__main__":
    curse = load_curse()
    report(curse, "2runo Curse-detection (MIT)")

    hs = load_hatescore()
    # 이진 라벨 매핑은 분포 출력을 본 뒤 아래에서 확정한다
    print("\n[hatescore] macrolabel별 표본:")
    for lbl in hs["macrolabel"].unique():
        sample = hs[hs["macrolabel"] == lbl]["comment"].iloc[0]
        print(f"  {lbl}: {str(sample)[:60]}")

    # 두 데이터셋 교차 중복
    hs_texts = set(hs["comment"].astype(str).str.strip())
    cross = curse["text"].isin(hs_texts).sum()
    print(f"\n데이터셋 간 교차 중복: {cross}건")
