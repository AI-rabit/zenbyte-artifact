# Reproduction guide

## What runs with zero setup

```
cd case2-anomaly
go test ./... -v          # all simulations, sweeps, and the 100k-key load test
go test -bench=Observe    # hot-path cost (paper: 41 ns/op)
```

Go ≥ 1.24, standard library only. This reproduces the executable content of
paper §6 (Table 4, Figure 3's underlying measurement, the poisoning defense,
restart recovery, and the pinned low-and-slow non-detection).

## What can be inspected without running anything

- `case1-toxicity/model/toxicity_model.zbsv` — the deployed artifact
  (format: `docs/D-zbsv-format.md`; metadata: `model/model_meta.json`).
- `case1-toxicity/equivalence/` — test vectors + reference probabilities.
- `case1-toxicity/inference-kotlin/TfidfSvmClassifier.kt` — the complete
  on-device inference engine (131 lines, no dependencies).
- `case1-toxicity/compliance/` — the three-layer zero-persistence proof.
- `docs/A–F` — full result tables.

## Reproducing the client-side pipeline

Environment: Python 3.10; `pip install scikit-learn numpy pandas fasttext
datasets`. The teacher additionally needs `torch transformers` (GPU
recommended for fine-tuning; everything downstream of the teacher's labels is
CPU-friendly).

Data: fetch the gold datasets per `data/README.md` and place them under
`case1-toxicity/pipeline/exp-0002-fasttext-tradeoff/data/` as
`curse_detection.txt` and `hatescore.csv`. External pools are pulled from
HuggingFace by the scripts themselves.

Execution order (paths relative to `case1-toxicity/pipeline/`; scripts locate
each other by these folder names — keep them):

1. `exp-0002-fasttext-tradeoff/code/prepare.py` — merge, normalize,
   deduplicate, stratified split (seed 42; test sealed).
2. `exp-0002-fasttext-tradeoff/code/` — `train.py`, `stage2.py`, `stage3.py`
   (the 79-configuration study), `baseline_keyword.py`, `kcelectra.py`
   (the unconstrained reference), `final_eval.py`.
3. `exp-0008-dataset-survey/code/` — `datasets_survey.py`,
   `transfer_matrix.py`, `augment_v2.py` (the five failed arms),
   `learning_curve.py`.
4. `exp-0009-distillation/code/` — `teacher.py` (trains the teacher on the
   gold train split only), `distill.py` (pool construction with leakage
   assertions + all attribution arms), `final_eval.py`.
5. `exp-0010-constrained-benchmark/code/` — `benchmark.py` (every candidate ×
   both conditions), `fair_tuning.py` (equal-budget retuning of the finalists).
6. `exp-0011-svm-deployment/code/` — `export_svm.py` (writes the ZBSV
   artifact), `reference_svm.py` + `equivalence_svm.py` (stage-1 equivalence),
   `calibrate_threshold.py` (the table in `docs/B-calibration.md`).

Determinism notes: scikit-learn results are deterministic given the seed;
fastText does not expose a seed, so fastText numbers are reported as the mean
of 3 runs (the recorded operating point happens to converge deterministically,
std 0.000). The sealed test split is opened once per experiment, at its
recorded operating point — preserving that discipline is part of reproducing
the result.

## Reproducing the compliance proof

The static audit and the two JUnit tests run inside the app's Gradle project,
which is not published while the product is in closed beta; the audit script,
the audited sources it checks, and both test files are included verbatim under
`case1-toxicity/compliance/` for inspection (see its README for what each
layer asserts and how the script fails loudly when an audit target is
missing).
