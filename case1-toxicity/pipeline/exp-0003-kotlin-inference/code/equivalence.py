"""exp-0003: verify equivalence 1, measure the quantization loss, and emit test
vectors for Kotlin.

- equivalence 1: the reference implementation (fed the original float weights)
  against the fasttext library — the maximum probability error over all of val
- quantization loss: val F1 along the int8 path and the decision agreement rate
  (th=0.375)
- test vectors: int8 probabilities for 1,000 test-set sentences, as input to the
  Kotlin JUnit suite
"""
import json
import sys
from pathlib import Path

import fasttext
import numpy as np

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))
from common import f1_binary, load_split          # noqa: E402
from threshold import prob_positive               # noqa: E402
from reference import ART, ZBFTModel              # noqa: E402

fasttext.FastText.eprint = lambda x: None
TH = 0.375


def main():
    ft = fasttext.load_model(str(EXP2 / "artifacts" / "operating_point.bin"))
    ref = ZBFTModel(ART / "toxicity_model.zbft")

    # equivalence 1: inject the original float weights, excluding quantization effects
    ref_float = ZBFTModel(ART / "toxicity_model.zbft")
    ref_float.matrix = ft.get_input_matrix()
    ref_float.output = ft.get_output_matrix().astype(np.float32)

    val = load_split("val")
    texts = val["text"].tolist()
    y = val["label"].tolist()

    p_ft = prob_positive(ft, texts)
    p_rf = np.array([ref_float.prob_positive(t) for t in texts])
    # fasttext can return probabilities marginally outside [0,1], so clip before comparing
    diff_float = np.abs(np.clip(p_ft, 0, 1) - np.clip(p_rf, 0, 1))
    print(f"equivalence 1 (float, n={len(texts)}): max|Δp|={diff_float.max():.2e}, mean={diff_float.mean():.2e}")

    p_i8 = np.array([ref.prob_positive(t) for t in texts])
    f1_ft = f1_binary(y, (np.clip(p_ft, 0, 1) >= TH).astype(int).tolist())["f1"]
    f1_i8 = f1_binary(y, (p_i8 >= TH).astype(int).tolist())["f1"]
    agree = float(((np.clip(p_ft, 0, 1) >= TH) == (p_i8 >= TH)).mean())
    print(f"quantization: val F1 {f1_ft:.4f} → {f1_i8:.4f} (Δ={f1_i8-f1_ft:+.4f}), decision agreement={agree:.4f}")

    # Kotlin test vectors (1,000 test-set sentences, int8 reference probabilities)
    test = load_split("test").head(1000)
    vectors = [{"text": t, "p1": round(float(ref.prob_positive(t)), 6)} for t in test["text"]]
    out = ART / "test_vectors.json"
    out.write_text(json.dumps(vectors, ensure_ascii=False))
    print(f"{len(vectors)} test vectors → {out}")


if __name__ == "__main__":
    main()
