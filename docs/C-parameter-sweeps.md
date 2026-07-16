# C — Parameter sweeps

Full sweep data behind three summarized results: the fastText plateau
(paper §5.2), the fairness benchmark (§5.5), and the server detector tuning
(§6.2).

## C.1 fastText size–accuracy study (79 configurations, 3 stages)

Scripts: `case1-toxicity/pipeline/exp-0002-fasttext-tradeoff/code/`
(`train.py`, `stage2.py`, `stage3.py`; per-run results in the scripts' CSV output).

- Stage 1 (grid, 54 settings) + stage 2 (targeted: epochs, loss, n-gram
  orders, oversampling) under the original 5 MB budget: **plateau at
  val F1 0.74–0.78**; no configuration reaches 0.80.
- Stage 3 (budget re-derived to 15 MB, new size band searched): best new-band
  val F1 0.783 — the plateau does not move with the budget.
- Confirmed operating point (sealed test, one shot): **test F1 0.744 at
  10.79 MB int8** (P 0.810, R 0.687), configuration
  `dim16 · bucket 500k · minn2/maxn5 · wordNgrams2 · epoch25 · threshold 0.375 (val-chosen)`.
  3 repeated runs: test F1 std 0.000 for this configuration.
- Structure of the plateau: without character n-grams, every configuration
  scores val F1 0.46–0.51 — below a 43-entry curse lexicon (0.576); with them,
  0.72–0.77. Embedding dim saturates at 16; buckets saturate at 500k
  (3.6 MB → 10.8 MB buys < +0.02 F1).
- Unconstrained reference on identical splits: KcELECTRA, test F1 0.853
  (P 0.877, R 0.830), ~420 MB — ineligible for deployment (C2/C4).

## C.2 Constrained benchmark: every candidate × both training conditions

Script: `case1-toxicity/pipeline/exp-0010-constrained-benchmark/code/benchmark.py`.
Identical test set and procedure (threshold chosen on val, test opened once);
identical int8 serialization rule for size; latency measured in Python, same
machine, same 1,000 sentences.

| Training | Model | test F1 | P | R | int8 size | p95 latency |
|---|---|---|---|---|---|---|
| — | curse lexicon (43 entries) | 0.568 | 0.784 | 0.445 | 0.4 KB | 0.003 ms |
| gold only | TF-IDF (word) + LR | 0.564 | 0.535 | 0.597 | 0.17 MB | 0.36 ms |
| gold only | TF-IDF (char 2–5) + LR | 0.711 | 0.763 | 0.666 | 1.06 MB | 0.41 ms |
| gold only | TF-IDF (char 2–5) + SVM | 0.732 | 0.794 | 0.679 | 1.06 MB | 0.41 ms |
| gold only | TF-IDF (char 2–5) + NB | 0.699 | 0.700 | 0.698 | 1.13 MB | 0.78 ms |
| gold only | fastText | **0.744** | 0.810 | 0.687 | 10.79 MB | 0.015 ms |
| gold + distill | TF-IDF (word) + LR | 0.590 | 0.670 | 0.528 | 0.69 MB | 0.29 ms |
| gold + distill | TF-IDF (char 2–5) + LR | 0.767 | 0.777 | 0.756 | 4.03 MB | 0.46 ms |
| gold + distill | TF-IDF (char 2–5) + SVM | **0.805** | 0.806 | 0.803 | 4.03 MB | 0.47 ms |
| gold + distill | TF-IDF (char 2–5) + NB | 0.736 | 0.817 | 0.669 | 4.27 MB | 1.87 ms |
| gold + distill | fastText | 0.788 | 0.820 | 0.758 | 14.04 MB | 0.010 ms |
| (reference) | KcELECTRA | 0.853 | 0.877 | 0.830 | ~420 MB | — (violates C2/C4) |

Ranking reversal: on gold data alone, fastText (0.744) beats the SVM (0.732);
grant every candidate the same distilled data and the SVM (0.805) beats
fastText (0.788). Latency caveat: fastText infers in C++, sklearn in
Python/scipy — the comparison across that column is not
implementation-fair, but both finalists sit below 1/100 of the 50 ms budget,
so the constraint verdict (C3) is unaffected.

## C.3 Final fair retuning (equal budget: 12 configurations each, distilled data)

Script: `case1-toxicity/pipeline/exp-0010-constrained-benchmark/code/fair_tuning.py`.
The first-pass fastText setting had been tuned on gold data; both finalists
were re-tuned from scratch on the distilled data with an identical budget.

| Candidate | Best configuration (val-chosen) | val F1 | test F1 | int8 size |
|---|---|---|---|---|
| fastText | dim16 / bucket 250k / n-gram (2,4) | 0.8262 | 0.7841 | 9.27 MB |
| **TF-IDF (char_wb 2–4) + SVM** | max_features 500k / C = 0.5 | 0.8285 | **0.8034** | **3.08 MB** |

- fastText configurations exceeding the 15 MB budget were disqualified
  outright (dim32/bucket ≥ 250k and bucket 1M: 15.2–40.9 MB) — the budget
  binds fastText before accuracy does.
- Generalization gap: fastText val 0.826 → test 0.784 (−0.042); SVM val
  0.8285 → test 0.8034 (−0.025). Interpretation: hash collisions (hundreds of
  thousands of n-grams into 250k buckets) drive val overfitting.
- The shipped artifact is the int8 quantization of the SVM finalist; it
  measures test F1 0.805 (quantization happened to help on test;
  see `docs/E-equivalence.md`).

## C.4 Server detector sweep (EWMA + CUSUM)

Runnable directly: `cd case2-anomaly && go test -run 'TestParameterSweep|TestRefinedSweep' -v`.
Synthetic traffic, 200 normal keys × 1 h, base rate 2 events/s; deterministic seed.

Excerpt (full grid in the test output):

| α | K | H | False-positive rate | burst (20×) | ramp (→10× / 60 s) | low-and-slow (2× sustained) |
|---|---|---|---|---|---|---|
| 0.3 | 0.5 | 5 | 93.0% | 0 s | 11 s | 0 s |
| 0.05 | 0.5 | 12 | 1.5% | 0 s | 17 s | missed |
| **0.1** | **0.5** | **10** | **0.0%** | **0 s** | **20 s** | **missed** ← adopted |
| 0.2 | 0.5 | 10 | 0.0% | 0 s | missed | missed |
| 0.1 | 0.75 | 10 | 0.0% | 0 s | missed | missed |

α (baseline adaptation speed) is the dominant variable for ramp detection: at
α ≥ 0.2 the baseline absorbs a gradual ramp as the new normal in real time.
H controls false positives (H ≥ 8 is the suppression boundary). At the adopted
setting, measured false positives over 500 keys × 1 h: 0.2%.

Fixed-threshold baseline under the same traffic (with diurnal variation):

| Method | FP rate | burst | ramp | low-and-slow |
|---|---|---|---|---|
| fixed 2× base | 100% | instant | instant | missed |
| fixed 3× base | 74% | instant | instant | missed |
| fixed 5× base | 0% | instant | delayed | missed |
| **EWMA+CUSUM (adopted)** | **0.2%** | instant | 20 s | missed |

Low-and-slow (2× sustained) is missed by every row — including a
long-term-profiling row that cannot exist here, because it requires the
longitudinal history a zero-persistence server does not keep. The
non-detection is itself pinned by a test so that it cannot silently become
stale documentation.
