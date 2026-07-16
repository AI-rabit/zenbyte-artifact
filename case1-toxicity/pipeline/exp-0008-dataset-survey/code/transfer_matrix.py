"""exp-0008 교차 데이터셋 전이 행렬.

각 데이터셋으로 fastText(동작점 설정)를 학습하고, 모든 데이터셋의 held-out에서 F1을 측정한다.
대각선 = in-domain 성능(그 데이터셋 자체의 난이도), 비대각선 = 라벨 정의·도메인 정합성.

핵심 질문: "train on X → eval on ours"가 낮으면, X를 증강해도 우리 성능은 오르지 않는다.
이것이 exp-0002의 unsmile 증강 실패에 대한 정량적 설명이 된다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))

from common import f1_binary, train_with_retry  # noqa: E402
from datasets_survey import DATA, load_all, load_ours  # noqa: E402

OP = {"dim": 16, "bucket": 500_000, "minn": 2, "maxn": 5, "lr": 0.125}
SEED = 42


def write_ft(df: pd.DataFrame, path: Path) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"__label__{r['label']} {r['text']}\n")
    return str(path)


def prob_pos(model, texts):
    labels, probs = model.predict(list(texts), k=2)
    out = []
    for ls, ps in zip(labels, probs):
        d = dict(zip(ls, ps))
        out.append(d.get("__label__1", 0.0))
    return np.array(out)


def best_f1(y, p) -> float:
    """임계값을 최적화한 F1 (데이터셋 간 양성 비율 차이를 보정)."""
    return max(f1_binary(y, (p >= th).astype(int).tolist())["f1"]
               for th in np.arange(0.05, 0.95, 0.025))


def main():
    DATA.mkdir(exist_ok=True)
    print("데이터 로드…")
    data = load_all()
    # 우리 val을 별도 평가 축으로 추가 (증강 의사결정의 실제 기준)
    ours_val = load_ours("val")

    # 각 데이터셋을 train/holdout으로 분할
    splits = {}
    for name, df in data.items():
        tr, ho = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
        splits[name] = (tr.reset_index(drop=True), ho.reset_index(drop=True))

    names = list(splits.keys())
    rows = []
    for train_name in names:
        tr, _ = splits[train_name]
        path = write_ft(tr, DATA / f"tm_{train_name.replace('/', '_')}.txt")
        model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                    loss="softmax", thread=1, verbose=0, **OP)
        row = {"train": train_name, "n_train": len(tr)}
        for eval_name in names:
            _, ho = splits[eval_name]
            row[eval_name] = round(best_f1(ho["label"].tolist(), prob_pos(model, ho["text"])), 3)
        row["→ ours(val)"] = round(best_f1(ours_val["label"].tolist(),
                                           prob_pos(model, ours_val["text"])), 3)
        rows.append(row)
        print(f"  {train_name} 학습 완료")

    df = pd.DataFrame(rows)
    print("\n=== 교차 전이 행렬 (fastText 동작점, 임계값 최적화 F1) ===")
    print("행=학습 데이터, 열=평가 데이터\n")
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).parent.parent / "artifacts" / "transfer_matrix.csv", index=False)

    print("\n--- 해석 ---")
    for r in rows:
        indomain = r[r["train"]]
        to_ours = r["→ ours(val)"]
        print(f"{r['train']:24s}: in-domain {indomain:.3f} → ours {to_ours:.3f} "
              f"(격차 {to_ours - indomain:+.3f})")


if __name__ == "__main__":
    main()
