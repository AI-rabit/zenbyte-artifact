"""exp-0002 data preparation: merge → normalize → deduplicate → stratified
split → fastText format.

Label definition (binary): 1 = inappropriate (profanity / hate / abuse), 0 = ordinary
- curse_detection: original labels kept as-is (1 = profanity)
- hatescore: hate speech and plain abuse → 1, Clean → 0

Split: train 70 / val 10 / test 20 (stratified, seed=42). The test split stays
sealed until the final report.
Output: ../data/{train,val,test}.txt (fastText) and ../data/{train,val,test}.csv (shared)
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent.parent / "data"
SEED = 42


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_all() -> pd.DataFrame:
    rows = []
    for line in (DATA / "curse_detection.txt").read_text(encoding="utf-8").splitlines():
        text, _, label = line.rpartition("|")
        if label.strip() in ("0", "1"):
            rows.append((normalize(text), int(label.strip()), "curse"))
    hs = pd.read_csv(DATA / "hatescore.csv", index_col=0)
    for _, r in hs.iterrows():
        label = 0 if r["macrolabel"] == "Clean" else 1
        rows.append((normalize(r["comment"]), label, f"hatescore/{r['source']}"))
    df = pd.DataFrame(rows, columns=["text", "label", "origin"])
    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    print(f"merged {before} → {len(df)} after deduplication")
    print(f"label distribution: {df['label'].value_counts().to_dict()} (positive rate {df['label'].mean():.3f})")
    print(f"source distribution:\n{df['origin'].value_counts()}")
    return df


def main():
    df = load_all()
    trainval, test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.125, stratify=trainval["label"], random_state=SEED)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(DATA / f"{name}.csv", index=False)
        with open(DATA / f"{name}.txt", "w", encoding="utf-8") as f:
            for _, r in part.iterrows():
                f.write(f"__label__{r['label']} {r['text']}\n")
        print(f"{name}: {len(part)} rows (positive {part['label'].mean():.3f})")


if __name__ == "__main__":
    main()
