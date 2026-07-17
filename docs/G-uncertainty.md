# G. Post-hoc uncertainty quantification (exp-0012 / exp-0013)

The paper's headline comparisons were re-examined after the fact under (i) bootstrap
resampling of the sealed test set, (ii) repeated retraining of the student models,
(iii) teacher retraining across seeds, and (iv) seed sweeps of the traffic
simulation. Pre-registered specs fixed all repetition counts, seeds, and verdict
criteria before execution; every run is reported (no selection). Where the
re-examination weakened a statement, the paper reports the weaker form
(see its Limitations section).

Bootstrap: B = 10,000, percentile CIs, seed 20260718, shared resample indices for
all paired comparisons (test n = 3,386).

## G.1 Deployed model (exp-0012)

| Quantity | Point | Uncertainty |
|---|---|---|
| Deployed int8 test F1 | 0.8050 | 95% CI [0.780, 0.829] |
| Distillation gain, SVM (paired) | +0.0728 | 95% CI [+0.051, +0.096] |
| Ranking SVM > fastText (this teacher's labels) | 0.8034 vs 0.7841 | SVM ahead in 10/10 fastText retrains; paired-bootstrap superiority 97.75% |
| Attribution ordering (teacher > baseline > {orig labels, self-training}) | 0.8174 / 0.7828 / 0.7710 / 0.7719 | direction stable across 10 retrains per arm |

Incidental finding: in this environment fastText training (thread=1) is fully
deterministic when each run gets a fresh process; the "nondeterminism" recorded in
earlier experiments traces to in-process repeated training (persistent NaN
divergence) and the learning-rate-halving retry it triggers. All original recorded
numbers reproduced exactly.

## G.2 Teacher-seed sweep (exp-0013, seeds 42/43/44)

Gold-only student baseline: test F1 0.7341.

| Teacher seed | Teacher test F1 | Student (SVM) test F1 | Distillation gain | SVM − fastText margin |
|---|---|---|---|---|
| 42 | 0.8507 | 0.7905 | +0.0564 | −0.0016 |
| 43 | 0.8710 | 0.7962 | +0.0622 | +0.0096 |
| 44 | 0.8647 | 0.7944 | +0.0603 | +0.0017 |

- The prescription's gain reproduces under every teacher realization (3/3, per-seed
  paired 95% CIs all exclude 0).
- The accuracy margin of the SVM-vs-fastText comparison does **not** reproduce
  reliably under fresh teachers (per-seed paired superiority 43–85%); the paper
  therefore ties that margin to the deployed teacher's labels. The size advantage
  (3.08 MB vs 9.27 MB) is teacher-invariant.
- The unconstrained reference (0.853 in the paper) lands at 0.851–0.871 across
  seeds; the paper's framing of it as a lower bound stands.

## G.3 Traffic-simulation seed sweep (exp-0013, 30 seeds)

Protocol of the recorded operating point (α = 0.1, K = 0.5σ, H = 10σ; 500 normal
keys × 3600 ticks, base 2.0/s; seeds 1000+i / 2000+i, i = 0..29). Harness:
`case2-anomaly/exp0013_uncertainty_test.go`, gated behind `ZENBYTE_EXP0013_OUT`
(skipped in a plain `go test ./...`).

| Quantity | Paper | Across 30 seeds |
|---|---|---|
| False-positive rate | 0.2% | mean 0.047% (95% CI 0.009–0.084%), max 0.4% — all ≤ 5% |
| Burst detection delay | first tick | 0 s in 30/30 |
| Ramp detection delay | 20 s | median 22 s, range 12–37 s (3/30 exceed 30 s) |
| Low-and-slow (2× sustained) | undetected | undetected in 30/30 |

## Reproducing

CPU only for exp-0012 (`runs_svm.py` → `runs_fasttext.py` → `analyze.py`); the
teacher sweep in exp-0013 (`teacher_seeds.py`) needs a GPU. The Go sweep:

```
cd case2-anomaly
ZENBYTE_EXP0013_OUT=$PWD/seed_sweep.csv go test -run TestExp0013 -v
```
