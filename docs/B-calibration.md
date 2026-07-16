# B — Calibration of the deployed model

Supplements paper §5.6, which reports only the deployed operating point.

Measured on the validation split (1,693 sentences) by loading the deployed
int8 ZBSV artifact directly — not the float training model. Script:
`case1-toxicity/pipeline/exp-0011-svm-deployment/code/calibrate_threshold.py`.

Validity check (pre-defined): the table's val F1 at the deployed threshold must
match the int8 model's recorded val F1 (0.8263) within ±0.001.
Result: **Δ 0.0000, PASS**.

| Threshold | Warning rate (all messages) | Recall | Precision | F1 | False-warning rate (clean sentences) |
|---|---|---|---|---|---|
| 0.30 | 41.6% | 98.3% | 41.6% | 0.585 | 29.5% |
| 0.375 | 27.8% | 93.0% | 58.8% | 0.720 | 13.9% |
| 0.425 | 21.9% | 88.6% | 71.2% | 0.789 | 7.7% |
| **0.475 (deployed)** | **17.4%** | **82.2%** | **83.1%** | **0.826** | **3.6%** |
| 0.55 | 13.5% | 69.1% | 90.4% | 0.783 | 1.6% |
| 0.70 | 7.7% | 43.3% | 98.5% | 0.601 | 0.1% |

Reading: at the deployed threshold the warning fires on 17.4% of messages and
falsely on 3.6% of clean sentences — acceptable for a dismissible advisory
(the user can always send anyway), and plainly insufficient for enforcement,
which is not what the feature does.

Process note recorded as a lesson: the same probability threshold produces
entirely different user-facing rates when the model changes. The calibration
table must be re-derived on every model swap — this table exists because an
earlier version of it silently referred to a previous model.
