# E — Equivalence procedure (Kotlin ≡ scikit-learn)

The paper claims the pure-Kotlin inference engine is *proven* equivalent to
the training stack, not assumed (§5.6). This is the procedure; it was first
established for a fastText port and re-used for the deployed SVM.

## Protocol (two stages, one bridge)

```
scikit-learn (training stack, float)
      │  stage 1: max |Δp| on the validation split (1,693 sentences)
      ▼
Python reference implementation (mirrors the exact intended semantics, incl. int8)
      │  stage 2: max |Δp| on 1,000 held-out test sentences (JUnit)
      ▼
Kotlin engine reading the shipped ZBSV artifact
```

The Python reference (`pipeline/exp-0011-svm-deployment/code/reference_svm.py`)
exists so that algorithmic errors are isolated *before* the JVM is involved:
token-level comparison against the library is possible in Python, painful in
Kotlin.

## Results for the deployed model

| Check | Criterion | Measured | Verdict |
|---|---|---|---|
| Stage 1: reference vs. scikit-learn (val, 1,693) | max Δp < 1e-4 | **9.68e-08** | pass |
| Stage 2: Kotlin vs. reference int8 (test, 1,000) | max Δp < 1e-4 | pass (JUnit) | pass |
| Quantization loss (val F1) | ≤ 0.005 | −0.0022 (0.8285 → 0.8263) | pass |
| Decision agreement at threshold 0.475 | ≥ 99% | reference 99.65% / **Kotlin 100%** (0 flips / 1,000) | pass |
| JVM latency (1,000 runs) | reference only | p95 = 76 µs | — |

The shipped int8 artifact measures test F1 0.805, marginally above the float
model's 0.8034 — quantization happened to help on test; the paper reports the
deployed number throughout, as a matter of reporting the shipped artifact
rather than the lab artifact.

Inputs for independent re-verification are included:
`case1-toxicity/equivalence/test_vectors.json` (sentences with expected
probabilities) and `sklearn_test_probs.npy` (reference probabilities).

## Reproduction subtleties worth recording

- The single scikit-learn subtlety in re-implementing `char_wb`:
  `_white_spaces = re.compile(r"\s\s+")` collapses only runs of **two or more**
  whitespace characters — single spaces pass through. Missing this produces
  small, hard-to-localize probability drift.
- The earlier fastText port (retired when the deployed model changed;
  `pipeline/exp-0003-kotlin-inference/`) had exactly one class of trap:
  integer sign extension — the `int8_t` cast inside FNV-1a hashing and the
  `int32_t → uint64_t` promotion in word-bigram hashing. Its first attempt
  failed at max Δp = 0.14; the reference-first protocol localized both sites.
  The SVM engine, by contrast, passed on the first attempt — there is nothing
  hashed or bit-twiddled to get wrong, which the paper counts as a portability
  advantage of the classical model under criterion C4.
