"""exp-0008 decisive diagnostic: the learning curve.

Question: "would F1 rise if we obtained more data that matches our own label
definition exactly?"

The transfer matrix showed that other people's data cannot be used. So what
about more data under the same definition? The curve is drawn by training on
25/50/75/100% of train.
  - still rising at the right-hand end → acquiring data is the answer, and
    labelling more under the same criteria is worth the investment.
  - flat → the data axis is saturated. The bottleneck is **model capacity**, and
    the remaining levers are distillation or a change of architecture.

A comparison worth running: train KcELECTRA on the same subsets — does a
higher-capacity model use the data better? Because of GPU cost, the fastText
curve is drawn first and this is extended only if needed.
"""
import statistics
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(Path(__file__).parent))

from common import train_with_retry  # noqa: E402
from datasets_survey import DATA, load_ours  # noqa: E402
from transfer_matrix import OP, best_f1, prob_pos, write_ft  # noqa: E402

REPEATS = 3
SEED = 42


def main():
    train = load_ours("train")
    val = load_ours("val")

    print("=== learning curve (ours, fastText operating point, val F1 averaged over 3 runs) ===\n")
    print("frac    n_train  n_positive |   val F1")
    results = []
    for frac in (0.25, 0.50, 0.75, 1.00):
        if frac < 1.0:
            sub, _ = train_test_split(train, train_size=frac, stratify=train["label"],
                                      random_state=SEED)
        else:
            sub = train
        path = write_ft(sub, DATA / f"lc_{int(frac*100)}.txt")
        f1s = []
        for _ in range(REPEATS):
            model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                        loss="softmax", thread=1, verbose=0, **OP)
            f1s.append(best_f1(val["label"].tolist(), prob_pos(model, val["text"])))
        m, s = statistics.mean(f1s), statistics.stdev(f1s)
        results.append((frac, len(sub), int(sub["label"].sum()), m, s))
        print(f"{frac:4.0%} {len(sub):9d} {int(sub['label'].sum()):10d} | {m:.4f} ± {s:.4f}")

    print("\n--- interpretation ---")
    d_last = results[-1][3] - results[-2][3]
    print(f"slope over the 75% → 100% segment: {d_last:+.4f} F1 per +{results[-1][1]-results[-2][1]} rows")
    if abs(d_last) < 0.01:
        print("→ the curve is flat. Doubling the data should gain under +0.01 — **the data axis is saturated**.")
        print("   the bottleneck is model capacity (fastText); the remaining levers are knowledge distillation or a change of architecture.")
    else:
        print("→ the curve is still rising. Acquiring more data under the same label definition is worthwhile.")


if __name__ == "__main__":
    main()
