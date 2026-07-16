"""exp-0008 H2 검증: 라벨 정합성을 고려한 선별 증강이 F1을 개선하는가.

전이 행렬에서 드러난 사실: 외부 데이터셋의 라벨은 우리보다 **넓다**
(우리 = 욕설/모욕 중심, 그들 = 비속어 없는 차별·편향 발언까지 포함).
따라서 단순 증강은 정밀도를 무너뜨린다.

이 스크립트는 4개 팔(arm)을 비교한다:
  A. 기존 (ours only)                          — 기준선
  B. 전량 증강 (ours + APEACH + kor-hate)      — 순진한 증강 (실패 예상, 대조군)
  C. 라벨 정렬 증강: 외부 양성 중 **욕설 어휘를 포함한 것만** 추가
     (우리 라벨 정의에 맞는 부분만 뽑아 쓰는 원칙적 필터)
  D. 음성만 증강: 외부 음성(clean)만 추가 — 라벨 정의 차이는 양성 쪽에 있으므로
     음성은 안전하게 재사용 가능하다는 가설

각 팔 3회 반복, val에서 임계값 최적화 F1 (test는 최종 1회만 개봉).
"""
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(Path(__file__).parent))

from baseline_keyword import LEXICON  # noqa: E402  (욕설 사전 — 라벨 정렬 필터에 사용)
from common import f1_binary, train_with_retry  # noqa: E402
from datasets_survey import DATA, load_all, load_ours  # noqa: E402
from transfer_matrix import OP, best_f1, prob_pos, write_ft  # noqa: E402

REPEATS = 3


def has_profanity(text: str) -> bool:
    return any(term in text for term in LEXICON)


def run_arm(name: str, train_df: pd.DataFrame, val: pd.DataFrame) -> tuple[float, float]:
    path = write_ft(train_df, DATA / f"arm_{name}.txt")
    f1s = []
    for _ in range(REPEATS):
        model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                    loss="softmax", thread=1, verbose=0, **OP)
        f1s.append(best_f1(val["label"].tolist(), prob_pos(model, val["text"])))
    mean, std = statistics.mean(f1s), statistics.stdev(f1s)
    print(f"{name:32s} n={len(train_df):6d} 양성={train_df['label'].mean():.3f} "
          f"→ val F1 = {mean:.4f} ± {std:.4f}")
    return mean, std


def main():
    data = load_all()
    ours = data["ours(curse+hatescore)"]
    val = load_ours("val")
    external = pd.concat([data["APEACH"], data["kor-hate-sentence"]], ignore_index=True)
    external = external.drop_duplicates(subset="text")
    external = external[~external["text"].isin(set(ours["text"]))]

    print("\n=== H2 검증: 선별 증강 4개 팔 (val, 3회 반복 평균) ===\n")

    # A. 기준선
    run_arm("A. 기존 (ours only)", ours, val)

    # B. 순진한 전량 증강
    run_arm("B. 전량 증강 (+APEACH+korhate)",
            pd.concat([ours, external], ignore_index=True), val)

    # C. 라벨 정렬 증강: 외부 양성 중 욕설 어휘 포함분만 + 외부 음성 전량
    ext_pos_aligned = external[(external["label"] == 1) & external["text"].map(has_profanity)]
    ext_neg = external[external["label"] == 0]
    print(f"   (라벨 정렬 필터: 외부 양성 {(external['label']==1).sum()}건 중 "
          f"욕설 포함 {len(ext_pos_aligned)}건만 채택)")
    run_arm("C. 라벨정렬 증강 (양성 필터+음성)",
            pd.concat([ours, ext_pos_aligned, ext_neg], ignore_index=True), val)

    # D. 음성만 증강
    run_arm("D. 음성만 증강", pd.concat([ours, ext_neg], ignore_index=True), val)

    # E. 라벨정렬 양성만 (음성 없이)
    run_arm("E. 정렬양성만 증강", pd.concat([ours, ext_pos_aligned], ignore_index=True), val)


if __name__ == "__main__":
    main()
