# F — Attribution control (all arms)

Full version of paper §5.4 / Table 2. Scripts:
`case1-toxicity/pipeline/exp-0009-distillation/code/` (`teacher.py`, `distill.py`).

## Leakage guards (enforced in code, not prose)

- Teacher (KcELECTRA) trains on **our training split only** (val/test never
  touched); teacher val F1 = 0.858.
- Unlabeled pool: text of UnSmile + APEACH + kor-hate-sentence, original
  labels discarded. 64,994 → deduplicated 38,359 → purged of every sentence
  overlapping our train/val/test → **34,305**.
- Two set-intersection assertions (`pool ∩ val = ∅`, `pool ∩ test = ∅`) abort
  the pipeline on violation.

## Label-definition shift, observed directly

On the identical 34,305 texts: original external labels are 65.1% positive;
the teacher's relabeling under our definition is 48.1% positive; label
agreement 78.1% — **22% of the pool changes labels**. The transfer-matrix
inclusion relation (`docs/A-transfer-matrix.md`) is here observed on the same
sentences rather than inferred from transfer scores.

## All six arms (val F1, mean of 3 runs)

| Arm | Train size | val F1 | Δ vs baseline |
|---|---|---|---|
| A. baseline (gold only) | 11,849 | 0.7828 | — |
| E. + pool with **original** labels | 46,154 | 0.7710 | −0.012 |
| D. + pool with the **student's own** labels (self-training) | 46,154 | 0.7719 | −0.011 |
| B. + pool with **teacher** labels (unfiltered) | 46,154 | 0.8013 | +0.019 |
| **C. + pool with teacher labels (confidence ≥ 0.9)** | **38,019** | **0.8174** | **+0.035** |
| F. pool only (gold excluded) | 34,305 | 0.7641 | −0.019 |

Arms A/E/D/B use the same texts, differing only in labels — the attribution is
therefore to the labeling function, not to text volume: the original-label arm
loses, the self-training arm loses, the teacher arm gains. Arm B shows the
gain does not depend on the confidence filter; arm F shows pseudo-labels
supplement gold labels rather than replace them.

## Confidence-filter sweep (non-monotonic)

| Confidence threshold | Kept samples | val F1 |
|---|---|---|
| 0.6 | 44,770 | 0.8090 |
| 0.7 | 43,177 | 0.8052 |
| 0.8 | 41,175 | 0.8137 |
| **0.9** | **38,019** | **0.8174** ← adopted |
| 0.95 | 34,846 | 0.8109 |

Discarding the teacher's uncertain samples (~8k) helps up to 0.9; at 0.95 the
sample-count loss outweighs the noise removal.

## Sealed-test outcome

Applied to the deployment candidate of the time (fastText): test F1
0.744 → 0.788 (+0.044), with the gain concentrated in recall
(0.687 → 0.758) — the transferred ability is recognizing insults that use no
profanity vocabulary. The same distilled data moves the final SVM to test F1
0.805 (`docs/C-parameter-sweeps.md`, §C.3).
