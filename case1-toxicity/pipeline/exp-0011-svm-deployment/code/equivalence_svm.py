"""exp-0011 3단계: 동치성 1 (Python 참조 vs sklearn) + 양자화 손실 + Kotlin 테스트 벡터.

exp-0003과 동일한 검증 절차를 SVM 경로에 적용한다.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP10 = Path(__file__).parent.parent.parent / "exp-0010-constrained-benchmark"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP10 / "code"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import svm_decision  # noqa: E402
from common import f1_binary, load_split  # noqa: E402
from export_svm import train  # noqa: E402
from reference_svm import ART, ZBSVModel  # noqa: E402


def main():
    vec, clf = train()  # 동일 시드·동일 설정이므로 export와 같은 모델
    ref = ZBSVModel(ART / "toxicity_model.zbsv")
    th = ref.threshold

    val = load_split("val")
    texts = val["text"].tolist()
    y = val["label"].tolist()

    # 동치성 1-a: float 계수로 참조 구현 vs sklearn (양자화 효과 배제)
    ref_float = ZBSVModel(ART / "toxicity_model.zbsv")
    ref_float.coef = clf.coef_.ravel().astype(np.float32)

    p_sk = svm_decision(clf, vec.transform(texts))
    p_rf = np.array([1 / (1 + np.exp(-ref_float.decision(t))) for t in texts])
    d = np.abs(p_sk - p_rf)
    print(f"동치성1 (float, n={len(texts)}): max|Δp| = {d.max():.2e}, mean = {d.mean():.2e}")

    # 양자화 손실
    p_i8 = np.array([ref.prob_toxic(t) for t in texts])
    f1_sk = f1_binary(y, (p_sk >= th).astype(int).tolist())["f1"]
    f1_i8 = f1_binary(y, (p_i8 >= th).astype(int).tolist())["f1"]
    agree = float(((p_sk >= th) == (p_i8 >= th)).mean())
    print(f"양자화: val F1 {f1_sk:.4f} → {f1_i8:.4f} (Δ={f1_i8 - f1_sk:+.4f}), 판정 일치율 {agree:.4f}")

    # Kotlin 테스트 벡터 (test 1,000문장, int8 참조 확률)
    test = load_split("test").head(1000)
    vectors = [{"text": t, "p1": round(float(ref.prob_toxic(t)), 6)} for t in test["text"]]
    (ART / "test_vectors.json").write_text(json.dumps(vectors, ensure_ascii=False))
    print(f"테스트 벡터 {len(vectors)}건 → artifacts/test_vectors.json")


if __name__ == "__main__":
    main()
