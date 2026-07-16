# Case study 1: on-device toxicity warning

Paper §5. The deployed result: **test F1 0.805 at 3.08 MB (int8), +0.88 MB
APK, p95 76 µs on the JVM** — 94% of an unconstrained 420 MB transformer's F1
(0.853) at 1/136 of its size, with the message never leaving the device.

| Folder | Contents | Paper |
|---|---|---|
| `pipeline/exp-0002-fasttext-tradeoff/` | 79-configuration size–accuracy study; budget re-derivation; hypothesis rejected as recorded | §5.2 |
| `pipeline/exp-0003-kotlin-inference/` | the earlier fastText port + equivalence protocol (retired; procedure reused) | §5.6 |
| `pipeline/exp-0008-dataset-survey/` | dataset survey, transfer matrix, failed augmentation arms, learning curve | §5.3 |
| `pipeline/exp-0009-distillation/` | teacher training, pool construction with leakage assertions, attribution arms | §5.4 |
| `pipeline/exp-0010-constrained-benchmark/` | every candidate × both conditions; equal-budget fair retuning | §5.5 |
| `pipeline/exp-0011-svm-deployment/` | ZBSV export, equivalence verification, threshold calibration | §5.6 |
| `model/` | the shipped artifact + metadata | §5.6 |
| `equivalence/` | test vectors + reference probabilities | §5.6 |
| `inference-kotlin/` | the complete production inference engine (pure Kotlin, no dependencies) | §5.6 |
| `compliance/` | the three-layer zero-persistence proof | §5.6 |

Pipeline scripts locate each other through the `exp-NNNN-*` folder names —
keep the layout intact. Data setup and execution order: `../REPRODUCE.md`.
