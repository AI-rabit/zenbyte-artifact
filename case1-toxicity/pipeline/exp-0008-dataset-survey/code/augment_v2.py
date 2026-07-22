"""exp-0008 test of H2: does selective augmentation that respects label
agreement improve F1?

What the transfer matrix revealed: the labels of the external datasets are
**broader** than ours (ours centre on profanity and insult; theirs extend to
discriminatory or biased statements that contain no profanity at all).
Indiscriminate augmentation therefore destroys precision.

This script compares several arms:
  A. baseline (ours only)                       — the reference
  B. augment with everything (ours + APEACH + kor-hate) — the naive version,
     expected to fail; it is the control
  C. label-aligned augmentation: of the external positives, add **only those
     containing profanity vocabulary** — a principled filter that takes just the
     portion matching our own label definition
  D. negatives only: add the external clean rows only, on the hypothesis that
     the label-definition mismatch lies on the positive side, so negatives can
     be reused safely

Each arm is repeated 3 times and scored by threshold-optimized F1 on val (test
is opened only once, at the end).
"""
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(Path(__file__).parent))

from baseline_keyword import LEXICON  # noqa: E402  (the profanity lexicon, used by the label-alignment filter)
from common import f1_binary, train_with_retry  # noqa: E402
from datasets_survey import DATA, load_all, load_ours  # noqa: E402
from transfer_matrix import OP, best_f1, prob_pos, write_ft  # noqa: E402

REPEATS = 3


def has_profanity(text: str) -> bool:
    return any(term in text for term in LEXICON)


def run_arm(name: str, train_df: pd.DataFrame, val: pd.DataFrame) -> tuple[float, float]:
    path = write_ft(train_df, DATA / f"arm_{name}.txt")
    f1s = []
    for _ in range(REPEATS):
        model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                    loss="softmax", thread=1, verbose=0, **OP)
        f1s.append(best_f1(val["label"].tolist(), prob_pos(model, val["text"])))
    mean, std = statistics.mean(f1s), statistics.stdev(f1s)
    print(f"{name:32s} n={len(train_df):6d} positive={train_df['label'].mean():.3f} "
          f"→ val F1 = {mean:.4f} ± {std:.4f}")
    return mean, std


def main():
    data = load_all()
    ours = data["ours(curse+hatescore)"]
    val = load_ours("val")
    external = pd.concat([data["APEACH"], data["kor-hate-sentence"]], ignore_index=True)
    external = external.drop_duplicates(subset="text")
    external = external[~external["text"].isin(set(ours["text"]))]

    print("\n=== H2: selective augmentation arms (val, averaged over 3 runs) ===\n")

    # A. baseline
    run_arm("A. baseline (ours only)", ours, val)

    # B. naive augmentation with everything
    run_arm("B. augment with all (+APEACH+korhate)",
            pd.concat([ours, external], ignore_index=True), val)

    # C. label-aligned: external positives containing profanity, plus all external negatives
    ext_pos_aligned = external[(external["label"] == 1) & external["text"].map(has_profanity)]
    ext_neg = external[external["label"] == 0]
    print(f"   (label-alignment filter: of {(external['label']==1).sum()} external positives, "
          f"only the {len(ext_pos_aligned)} containing profanity are kept)")
    run_arm("C. label-aligned (filtered positives + negatives)",
            pd.concat([ours, ext_pos_aligned, ext_neg], ignore_index=True), val)

    # D. negatives only
    run_arm("D. negatives only", pd.concat([ours, ext_neg], ignore_index=True), val)

    # E. aligned positives only (no negatives)
    run_arm("E. aligned positives only", pd.concat([ours, ext_pos_aligned], ignore_index=True), val)


if __name__ == "__main__":
    main()
