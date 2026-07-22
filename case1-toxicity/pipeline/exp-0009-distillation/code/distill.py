"""exp-0009 stage 2: the distillation experiment and its attribution controls
(scored on val; test stays sealed).

The arms (all at the fastText operating point, averaged over 3 repeats):
  A. baseline               : ours only
  B. teacher distillation, all      : ours + pool (teacher hard labels)
  C. teacher distillation, confident: ours + pool (only samples the teacher is
     confident about) — swept over the confidence threshold
  D. **self-training control**: ours + pool labelled by the student fastText
     itself                                    ← the core of the H2 attribution test
  E. **original-label control**: ours + pool with the external original labels
                                                ← re-confirms exp-0008
  F. pool only (teacher labels): the pool without ours

If D produces a gain comparable to B, then attributing the improvement to "the
teacher's knowledge" is wrong — it would be a plain self-training effect.
E re-confirms that the label definition was the problem, holding the text fixed
and changing only the labels.
"""
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from common import f1_binary, load_split, train_with_retry  # noqa: E402
from transfer_matrix import OP, best_f1, prob_pos, write_ft  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"
DATA = Path(__file__).parent.parent / "data"
REPEATS = 3


def run(name: str, train_df: pd.DataFrame, val: pd.DataFrame) -> float:
    DATA.mkdir(exist_ok=True)
    path = write_ft(train_df[["text", "label"]], DATA / f"{name.replace(' ', '_')}.txt")
    f1s = []
    for _ in range(REPEATS):
        model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                    loss="softmax", thread=1, verbose=0, **OP)
        f1s.append(best_f1(val["label"].tolist(), prob_pos(model, val["text"])))
    m, s = statistics.mean(f1s), statistics.stdev(f1s)
    print(f"  {name:34s} n={len(train_df):6d} positive={train_df['label'].mean():.3f} "
          f"→ val F1 = {m:.4f} ± {s:.4f}")
    return m


def student_pseudo_labels(ours: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    """The student (fastText) labels the pool itself — the self-training control."""
    path = write_ft(ours[["text", "label"]], DATA / "student_teacher.txt")
    model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                loss="softmax", thread=1, verbose=0, **OP)
    return prob_pos(model, pool["text"])


def main():
    ours = load_split("train")[["text", "label"]]
    val = load_split("val")
    pool = pd.read_csv(ART / "pseudo_labels.csv")

    print(f"pool of {len(pool)} rows, teacher positive rate {(pool['teacher_prob'] >= 0.5).mean():.3f}\n")

    print("=== baseline and controls ===")
    base = run("A. baseline (ours only)", ours, val)

    # E. original-label control (same text, only the labels are the external originals)
    e_df = pd.concat([ours, pool[["text", "orig_label"]].rename(columns={"orig_label": "label"})],
                     ignore_index=True)
    e = run("E. original-label control (external labels)", e_df, val)

    # D. self-training control (the student does the labelling)
    sp = student_pseudo_labels(ours, pool)
    d_df = pd.concat([ours, pd.DataFrame({"text": pool["text"], "label": (sp >= 0.5).astype(int)})],
                     ignore_index=True)
    d = run("D. self-training control (student labels)", d_df, val)

    print("\n=== teacher distillation ===")
    # B. all of the pool
    b_df = pd.concat([ours, pd.DataFrame({"text": pool["text"],
                                          "label": (pool["teacher_prob"] >= 0.5).astype(int)})],
                     ignore_index=True)
    b = run("B. teacher distillation (all)", b_df, val)

    # C. sweep over the confidence filter
    print("\n=== C. confidence-filter sweep (only samples the teacher is confident about) ===")
    best_c, best_margin = -1.0, None
    for margin in (0.6, 0.7, 0.8, 0.9, 0.95):
        conf = pool[(pool["teacher_prob"] >= margin) | (pool["teacher_prob"] <= 1 - margin)]
        c_df = pd.concat([ours, pd.DataFrame({"text": conf["text"],
                                              "label": (conf["teacher_prob"] >= 0.5).astype(int)})],
                         ignore_index=True)
        c = run(f"C. teacher distillation (confidence ≥{margin})", c_df, val)
        if c > best_c:
            best_c, best_margin = c, margin

    # F. pool only
    f_df = pd.DataFrame({"text": pool["text"], "label": (pool["teacher_prob"] >= 0.5).astype(int)})
    f = run("F. pool only (ours excluded)", f_df, val)

    print("\n=== summary (val) ===")
    print(f"  A baseline                 : {base:.4f}")
    print(f"  E original-label control   : {e:.4f}  ({e - base:+.4f})")
    print(f"  D self-training control    : {d:.4f}  ({d - base:+.4f})   ← attribution test")
    print(f"  B teacher distillation, all: {b:.4f}  ({b - base:+.4f})")
    print(f"  C teacher distillation, best: {best_c:.4f}  ({best_c - base:+.4f})  @ confidence {best_margin}")
    print(f"  F pool only                : {f:.4f}  ({f - base:+.4f})")
    print()
    if best_c > base and best_c - d > 0.01:
        print("→ H2 supported: the distillation gain clearly exceeds the self-training control, so it can be attributed to knowledge transferred from the teacher")
    elif best_c > base:
        print("→ H2 uncertain: there is an improvement, but it is close to the self-training control — the attribution needs re-examination")
    else:
        print("→ H1 rejected: teacher distillation does not beat the baseline")


if __name__ == "__main__":
    main()
