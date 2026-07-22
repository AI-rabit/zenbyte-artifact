"""exp-0008 cross-dataset transfer matrix.

fastText (at the operating-point configuration) is trained on each dataset and
scored by F1 on the held-out portion of every dataset. The diagonal is
in-domain performance (how hard that dataset is in itself); the off-diagonal
measures how well label definitions and domains agree.

The question that matters: if "train on X → eval on ours" is low, then
augmenting with X cannot raise our performance. That is the quantitative
explanation for why the unsmile augmentation failed in exp-0002.
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
    """F1 with the threshold optimized, which corrects for differing positive rates across datasets."""
    return max(f1_binary(y, (p >= th).astype(int).tolist())["f1"]
               for th in np.arange(0.05, 0.95, 0.025))


def main():
    DATA.mkdir(exist_ok=True)
    print("loading data…")
    data = load_all()
    # add our own val as a separate evaluation axis — it is the criterion the
    # augmentation decision actually rests on
    ours_val = load_ours("val")

    # split each dataset into train and holdout
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
        print(f"  {train_name} trained")

    df = pd.DataFrame(rows)
    print("\n=== cross-transfer matrix (fastText operating point, threshold-optimized F1) ===")
    print("rows = training data, columns = evaluation data\n")
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).parent.parent / "artifacts" / "transfer_matrix.csv", index=False)

    print("\n--- interpretation ---")
    for r in rows:
        indomain = r[r["train"]]
        to_ours = r["→ ours(val)"]
        print(f"{r['train']:24s}: in-domain {indomain:.3f} → ours {to_ours:.3f} "
              f"(gap {to_ours - indomain:+.3f})")


if __name__ == "__main__":
    main()
