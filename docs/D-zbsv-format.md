# D — ZBSV format v1 (the deployed model file)

The artifact `case1-toxicity/model/toxicity_model.zbsv` (3,231,173 bytes) is
the exact file shipped inside the app's assets. Byte layout (little-endian):

```
magic 'ZBSV' | version (u8)
minN (u8), maxN (u8), nTerms (u32)          # char_wb n-gram orders, vocabulary size
sublinearTf (u8), useIdf (u8)
threshold (f32)                             # deployed decision threshold (0.475)
intercept (f32)                             # SVM intercept
coefScale (f32)                             # global max-abs int8 scale (single scalar)
idf (f32 × nTerms)
coef (int8 × nTerms)
vocab (nTerms × { u16 length | UTF-8 bytes })
```

- Writer: `case1-toxicity/pipeline/exp-0011-svm-deployment/code/export_svm.py`
- Reader: `case1-toxicity/inference-kotlin/TfidfSvmClassifier.kt`
- Metadata snapshot: `case1-toxicity/model/model_meta.json`

Design notes:

- Coefficients use **global max-abs int8 quantization** — one scale for the
  whole vector (simpler than fastText's per-row scaling, and lossless in
  effect here: val F1 −0.0022; the shipped int8 model measures test F1 0.805
  vs. the float model's 0.8034).
- Vocabulary strings dominate the size: 209,870 n-grams × ~11 B ≈ 2.3 MB of
  the 3.08 MB total.
- The int8 weights and vocabulary are low-entropy: deflate inside the APK
  compresses the asset 3.5×, so the user-visible download cost is
  **+0.88 MB** — the number the paper argues is the budget one should measure.
