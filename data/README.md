# Datasets: sources and licenses

No dataset is redistributed in this repository. Gold datasets carry permissive
licenses but are fetched from their canonical sources to preserve provenance;
the distillation-pool texts carry share-alike / non-commercial licenses and
are therefore referenced, never copied (see the paper's Limitations section).

## Gold data (training/evaluation)

| Dataset | Size used | License | Source |
|---|---|---|---|
| Curse-detection-data ("2runo") | 5,825 sentences | MIT | github.com/2runo/Curse-detection-data |
| HateScore | 11,108 sentences | Apache-2.0 | github.com/sgunderscore/hatescore-korean-hate-speech (paper: arXiv:2204.03262) |

Merged, NFC-normalized, deduplicated → 16,928 sentences (17.6% positive);
stratified 70/10/20 split, seed 42; the 3,386-sentence test split stays sealed
until each experiment's operating point is fixed. Preparation:
`case1-toxicity/pipeline/exp-0002-fasttext-tradeoff/code/prepare.py`, which
expects `curse_detection.txt` (lines of `text|label`) and `hatescore.csv`
under that experiment's `data/` directory.

## Distillation pool (text only — original labels are discarded by design)

| Dataset | License | Source |
|---|---|---|
| Korean UnSmile | CC-BY-NC-ND | github.com/smilegate-ai/korean_unsmile_dataset |
| APEACH | CC-BY-SA-4.0 | HuggingFace: jason9693/APEACH |
| kor-hate-sentence | CC-BY-SA-4.0 | HuggingFace: SJ-Donald/kor-hate-sentence |

Pool construction (`exp-0009-distillation/code/distill.py`): 64,994 texts →
deduplicated 38,359 → purged of any sentence overlapping our train/val/test →
34,305. kor-hate-sentence overlaps our gold sources by lineage; 1,764
overlapping sentences are removed before any use.

License note, stated as in the paper: the pool's *texts* are acceptable for
this research artifact; a commercial deployment would rebuild the pool from
permissively licensed text. The labels used for training are the teacher's
own output and carry no external license.

## Teacher model

KcELECTRA (MIT) — github.com/Beomi/KcELECTRA. Trained here only on our gold
training split; never shipped.
