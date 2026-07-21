# Artifact: The price of absence — deploying AI under zero-persistence constraints

Anonymized artifact repository accompanying a PoPETs 2027.2 submission.

Zenbyte is a zero-persistence messenger: no database, no logs, no push
infrastructure, no external API calls — from device to server, a message exists
only in volatile memory. This artifact contains the code, models, test vectors,
and verification suites behind the paper's two case studies, plus supplementary
result tables (the full versions of data the paper summarizes in one line).

Everything here was produced by the authors; nothing contains or derives from
user data — the system's architecture makes such data impossible to collect,
which is the property the paper studies.

## Layout

```
case1-toxicity/            Case study 1: on-device toxicity warning (paper §5)
  pipeline/                training & evaluation pipeline, one folder per experiment
  model/                   the deployed model artifact (ZBSV v1, int8, 3.08 MB)
  equivalence/             test vectors + reference probabilities for the equivalence proof
  inference-kotlin/        the pure-Kotlin inference engine that ships in the app
  compliance/              zero-persistence compliance tests (static / network / disk)
case2-anomaly/             Case study 2: server traffic anomaly detection (paper §6)
                           self-contained Go module — `go test ./...` runs everything
docs/                      supplementary result tables (A–F, see below)
data/                      dataset sources & licenses (data is referenced, not redistributed)
REPRODUCE.md               environment, data acquisition, execution order
```

## Claim → verification map

| Paper claim | Where to verify |
|---|---|
| Deployed model: test F1 0.805 at 3.08 MB, int8 (§5.6) | `case1-toxicity/model/` + `pipeline/exp-0011-svm-deployment/` |
| Fairness control reverses model ranking (§5.5, T3) | `pipeline/exp-0010-constrained-benchmark/` + `docs/C-parameter-sweeps.md` |
| Attribution: gain comes from teacher labels, not more text (§5.4, T2) | `pipeline/exp-0009-distillation/` + `docs/F-attribution.md` |
| Label-definition transfer asymmetry (§5.3) | `pipeline/exp-0008-dataset-survey/` + `docs/A-transfer-matrix.md` |
| Kotlin ≡ scikit-learn, max Δp < 1e-4, 0 decision flips (§5.6) | `case1-toxicity/equivalence/` + `inference-kotlin/` + `docs/E-equivalence.md` |
| Zero-persistence proven at 3 layers: static / network / disk (§5.6) | `case1-toxicity/compliance/` — `bash zero_persistence_audit.sh` runs the static layer |
| Warning-rate calibration at the deployed threshold (§5.6) | `docs/B-calibration.md` + `pipeline/exp-0011-svm-deployment/code/calibrate_threshold.py` |
| Memory ceiling: 100k keys → tracked state flat at 10k, heap +2.1 MB (§6.2, F3) | `case2-anomaly/` — `go test -run TestMemoryBoundUnderKeyFlood -v` |
| Baseline freeze under alarm (poisoning defense) (§6.2) | `case2-anomaly/` — `go test -run TestBaselinePoisoning -v` |
| Low-and-slow non-detection, pinned by a test (§6.5) | `case2-anomaly/simulation_test.go` (the sweep asserts the miss) |
| Cold-start blind window covered by absolute ceiling (§6.3) | `case2-anomaly/` — `go test -run TestAbsoluteCeiling` |
| Hot-path cost 41 ns/op (§6.2) | `case2-anomaly/` — `go test -bench=Observe` |

## Quickest verification (no setup)

The server case study is fully self-contained (standard library only):

```
cd case2-anomaly && go test ./... -v
```

This runs the parameter sweeps, the 100,000-key memory-bound load test, the
baseline-poisoning defense, the restart-recovery and absolute-ceiling tests,
and the low-and-slow non-detection assertion — the executable form of paper
Table 4 and Figure 3.

The client pipeline requires the public datasets (see `data/README.md` and
`REPRODUCE.md`). The deployed model, its test vectors, and the reference
probabilities are included, so the shipped artifact itself can be inspected
without any download.

## Supplementary tables (`docs/`)

- **A — Cross-dataset transfer matrix**: the full 4×4 matrix behind §5.3, plus
  the five failed augmentation arms and the learning curve.
- **B — Calibration**: warning rate / recall / precision / false-warning rate
  across thresholds for the deployed model.
- **C — Parameter sweeps**: the 79-configuration fastText study, the fair
  retuning benchmark of every surviving candidate, and the server-side
  (α, K, H) sweep with the fixed-threshold baseline.
- **D — ZBSV format**: byte layout of the deployed model file.
- **E — Equivalence procedure**: the two-stage protocol proving the Kotlin
  re-implementation equivalent to scikit-learn.
- **F — Attribution control**: all six training arms and the confidence-filter
  sweep behind §5.4.

## License

Code and documentation in this repository: MIT (see `LICENSE`).
Datasets are not redistributed here; see `data/README.md` for sources and licenses.
