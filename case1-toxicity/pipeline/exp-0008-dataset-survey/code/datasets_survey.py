"""exp-0008 dataset loaders and survey table.

Every dataset is normalized to a binary (text, label) form, where label 1 =
inappropriate (profanity / hate / abuse). Datasets whose license is unclear are
not loaded.

Note: the Korean strings in the loaders below ("문장") are column names in the
source datasets and must stay as they are.
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


# ── survey table (license status as checked on 2026-07-13) ────────────────
SURVEY = [
    # name, size, license, label definition, domain, verdict
    ("curse (2runo)", 5825, "MIT", "contains profanity", "online comments", "✅ already in use"),
    ("hatescore", 11108, "Apache-2.0", "hate speech / plain abuse vs Clean", "comments + wiki + rule-generated", "✅ already in use"),
    ("unsmile", 18742, "CC-BY-NC-ND", "9 hate categories + abuse/profanity", "online comments", "⚠️ hurt performance in exp-0002"),
    ("APEACH", 11666, "CC-BY-SA-4.0", "is hate speech (crowd-generated)", "crowd-generated", "🔬 evaluated here"),
    ("kor-hate-sentence", 36661, "CC-BY-SA-4.0", "hate vs clean", "comments (likely unsmile/kmhas lineage)", "🔬 evaluated here"),
    ("2tle/korean-curse", 1000, "MIT", "profanity span annotation", "same sentences as 2runo", "❌ duplicate (no new data)"),
    ("josephnam/korean_toxic", 8448, "unclear", "harmful LLM queries (safety)", "synthetic prompts", "❌ different task + unclear license"),
    ("jigsaw (English)", 160000, "CC0", "6 toxic labels, multi-label", "wiki talk pages (English)", "🔬 reference only (language mismatch)"),
]


def print_survey():
    df = pd.DataFrame(SURVEY, columns=["dataset", "size", "license", "label definition", "domain", "verdict"])
    print(df.to_string(index=False))


# ── loaders ──────────────────────────────────────────────────────────────
def load_ours(split: str) -> pd.DataFrame:
    """The merged data from exp-0002 (curse + hatescore) — our own reference distribution."""
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
            # clean=1 means ordinary, hate=1 means hateful
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
            # drop sentences overlapping our own evaluation splits (no leakage)
            before = len(df)
            df = df[~df["text"].isin(ours_eval)]
            if before != len(df):
                print(f"  [{name}] removed {before - len(df)} rows leaking into our evaluation splits")
        out[name] = df.reset_index(drop=True)
        print(f"  [{name}] {len(df)} rows, positive {df['label'].mean():.3f}")
    return out


if __name__ == "__main__":
    print("=== exp-0008 dataset survey ===")
    print_survey()
    print("\n=== load results ===")
    load_all()
