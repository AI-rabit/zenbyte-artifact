# A — Cross-dataset transfer matrix (full)

Supplements paper §5.3, which reports only the row/column summary
(ours→external 0.70–0.85; external→ours 0.55–0.60).

Setup: a fastText model at the operating-point configuration is trained on each
dataset and evaluated on every dataset's held-out split, with the decision
threshold optimized per cell (best-F1). Identical preprocessing throughout.
Script: `case1-toxicity/pipeline/exp-0008-dataset-survey/code/transfer_matrix.py`.

| train ↓ / eval → | ours | UnSmile | APEACH | kor-hate-sentence | → ours (val) |
|---|---|---|---|---|---|
| **ours** (curse + HateScore) | 0.772 | 0.852 | 0.696 | 0.806 | **0.770** |
| UnSmile | 0.553 | 0.894 | 0.742 | 0.871 | **0.551** |
| APEACH | 0.569 | 0.868 | 0.803 | 0.846 | **0.550** |
| kor-hate-sentence | 0.713 | 0.976 | 0.891 | 0.847 | **0.603** |

Transfer flows in one direction only. Models trained on our data score
0.70–0.85 on the external corpora; models trained on external corpora score
0.55–0.60 on ours. The pattern is an inclusion relation between label
definitions: ours (profanity/direct insult, narrow) ⊂ external
(profanity-free discrimination and bias included, broad). A broad-definition
model flags profanity-free discriminatory sentences; our evaluation counts
those as negatives; precision collapses.

Leakage note: kor-hate-sentence is a compilation with lineage back to our gold
sources; 1,764 sentences overlapping our evaluation sets were detected and
removed before any use.

## The five failed augmentation arms

Validation F1, mean of 3 runs; baseline = gold data only.

| Arm | Train size | Positive rate | val F1 | Δ vs baseline |
|---|---|---|---|---|
| A. baseline (ours only) | 11,849 | 0.176 | **0.7828** | — |
| B. full augmentation (+APEACH +kor-hate) | 46,154 | 0.529 | 0.7710 | −0.012 |
| C. label-aligned positives + negatives | 30,508 | 0.287 | 0.7737 | −0.009 |
| D. negatives only | 23,823 | 0.087 | 0.7794 | −0.003 |
| E. aligned positives only | 18,534 | 0.473 | 0.7617 | −0.021 |

All five arms fail — including D (negatives only), designed on the assumption
that label disagreement lives only in the positive class. Negative-class
*domain* (crowd-generated prose, other communities' registers) also shifts the
distribution. A profanity-lexicon filter over external positives (22,331 →
6,685 kept) does not rescue arms C/E: a lexicon approximates a label
definition; it does not reproduce one.

## Learning curve (same label definition)

| Fraction of our training data | Sentences | val F1 |
|---|---|---|
| 25% | 2,962 | 0.6837 |
| 50% | 5,924 | 0.7500 |
| 75% | 8,886 | 0.7647 |
| 100% | 11,849 | **0.7828** |

The curve is still rising at 100% (+0.018 F1 per ~3k sentences in the final
segment). Capacity was never the binding constraint; data carrying *our* label
definition was — and that data cannot be collected in a zero-persistence
system. This diagnosis is what motivates teacher relabeling (paper §5.4,
`docs/F-attribution.md`).
