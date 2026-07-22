"""exp-0002 EDA: distribution, duplication and obfuscation patterns of the two
datasets.

Run: activate the research virtualenv, then `python eda.py`
Input: ../data/curse_detection.txt (2runo, MIT), ../data/hatescore.csv (Apache-2.0)
Output: a stdout report (to be pasted into the experiment log)
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data"


def load_curse() -> pd.DataFrame:
    rows, bad = [], 0
    for line in (DATA / "curse_detection.txt").read_text(encoding="utf-8").splitlines():
        # format: sentence|label  (a sentence may contain '|', hence rpartition)
        if "|" not in line:
            bad += 1
            continue
        text, _, label = line.rpartition("|")
        label = label.strip()
        if label not in ("0", "1"):
            bad += 1
            continue
        rows.append((text.strip(), int(label)))
    print(f"[curse] unparsable lines: {bad}")
    return pd.DataFrame(rows, columns=["text", "label"])


def load_hatescore() -> pd.DataFrame:
    df = pd.read_csv(DATA / "hatescore.csv", index_col=0)
    print(f"[hatescore] columns: {list(df.columns)}")
    print(f"[hatescore] macrolabel distribution:\n{df['macrolabel'].value_counts()}")
    print(f"[hatescore] top microlabels:\n{df['microlabel'].value_counts().head(12)}")
    print(f"[hatescore] source distribution:\n{df['source'].value_counts()}")
    # binarization: hate-speech family → 1, everything else (ordinary sentences, ...) → 0
    # (the exact mapping is fixed once the distribution above has been inspected)
    return df


def report(df: pd.DataFrame, name: str):
    print(f"\n===== {name} =====")
    print(f"rows: {len(df)}")
    print(f"label distribution:\n{df['label'].value_counts(normalize=True).round(3)}")
    dup = df.duplicated(subset="text").sum()
    print(f"exact duplicate sentences: {dup}")
    df["len"] = df["text"].str.len()
    print(f"length: median={df['len'].median():.0f}, p95={df['len'].quantile(0.95):.0f}, max={df['len'].max()}")
    empty = (df["text"].str.strip() == "").sum()
    print(f"blank sentences: {empty}")
    # sample of obfuscation patterns
    obfus = {
        "digit infix (e.g. 시1발)": r"[가-힣]\d[가-힣]",
        "bare jamo (e.g. ㅅㅂ, ㅄ)": r"[ㄱ-ㅎㅏ-ㅣ]{2,}",
        "punctuation infix": r"[가-힣][@#$%^&*~\-_.]+[가-힣]",
    }
    pos = df[df["label"] == 1] if "label" in df else df
    for desc, pat in obfus.items():
        hits = pos["text"].str.contains(pat, regex=True).sum()
        print(f"obfuscation [{desc}]: {hits} of the label=1 rows ({hits/max(len(pos),1)*100:.1f}%)")


if __name__ == "__main__":
    curse = load_curse()
    report(curse, "2runo Curse-detection (MIT)")

    hs = load_hatescore()
    # the binary label mapping is fixed below, after inspecting the printed distribution
    print("\n[hatescore] a sample per macrolabel:")
    for lbl in hs["macrolabel"].unique():
        sample = hs[hs["macrolabel"] == lbl]["comment"].iloc[0]
        print(f"  {lbl}: {str(sample)[:60]}")

    # cross-dataset duplication
    hs_texts = set(hs["comment"].astype(str).str.strip())
    cross = curse["text"].isin(hs_texts).sum()
    print(f"\ncross-dataset duplicates: {cross}")
