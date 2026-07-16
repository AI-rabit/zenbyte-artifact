"""exp-0002 데이터 준비: 병합 → 정규화 → 중복 제거 → 층화 분할 → fastText 포맷.

라벨 정의 (이진): 1 = 부적절(욕설/혐오/악플), 0 = 정상
- curse_detection: 원 라벨 그대로 (1=욕설)
- hatescore: 혐오발언/단순 악플 → 1, Clean → 0

분할: train 70 / val 10 / test 20 (층화, seed=42). test는 최종 보고 전까지 봉인.
출력: ../data/{train,val,test}.txt (fastText), ../data/{train,val,test}.csv (공용)
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
    print(f"병합 {before} → 중복 제거 후 {len(df)}")
    print(f"라벨 분포: {df['label'].value_counts().to_dict()} (양성 비율 {df['label'].mean():.3f})")
    print(f"출처 분포:\n{df['origin'].value_counts()}")
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
        print(f"{name}: {len(part)}건 (양성 {part['label'].mean():.3f})")


if __name__ == "__main__":
    main()
