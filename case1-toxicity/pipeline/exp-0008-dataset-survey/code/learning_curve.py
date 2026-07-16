"""exp-0008 결정적 진단: 학습 곡선.

질문: "우리 라벨 정의와 완벽히 일치하는 데이터를 더 구하면 F1이 오르는가?"

전이 행렬은 "남의 데이터는 못 쓴다"를 보였다. 그렇다면 같은 정의의 데이터를 더 모으면?
train의 25/50/75/100%로 학습해 곡선을 그린다.
  - 곡선이 끝에서 우상향 → 데이터 확보가 답. 같은 기준으로 라벨링 투자할 가치 있음.
  - 곡선이 평탄 → 데이터는 포화. 병목은 **모델 용량**이며, 남은 레버는 증류/아키텍처뿐.

비교선: 같은 부분집합으로 KcELECTRA를 학습하면? (용량이 큰 모델은 데이터를 더 잘 쓰는가)
→ GPU 비용 때문에 fastText 곡선만 먼저 그리고, 필요 시 확장.
"""
import statistics
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(Path(__file__).parent))

from common import train_with_retry  # noqa: E402
from datasets_survey import DATA, load_ours  # noqa: E402
from transfer_matrix import OP, best_f1, prob_pos, write_ft  # noqa: E402

REPEATS = 3
SEED = 42


def main():
    train = load_ours("train")
    val = load_ours("val")

    print("=== 학습 곡선 (ours, fastText 동작점, val F1 3회 평균) ===\n")
    print("비율   학습건수   양성건수 |   val F1")
    results = []
    for frac in (0.25, 0.50, 0.75, 1.00):
        if frac < 1.0:
            sub, _ = train_test_split(train, train_size=frac, stratify=train["label"],
                                      random_state=SEED)
        else:
            sub = train
        path = write_ft(sub, DATA / f"lc_{int(frac*100)}.txt")
        f1s = []
        for _ in range(REPEATS):
            model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                        loss="softmax", thread=1, verbose=0, **OP)
            f1s.append(best_f1(val["label"].tolist(), prob_pos(model, val["text"])))
        m, s = statistics.mean(f1s), statistics.stdev(f1s)
        results.append((frac, len(sub), int(sub["label"].sum()), m, s))
        print(f"{frac:4.0%} {len(sub):9d} {int(sub['label'].sum()):10d} | {m:.4f} ± {s:.4f}")

    print("\n--- 해석 ---")
    d_last = results[-1][3] - results[-2][3]
    print(f"75% → 100% 구간 기울기: {d_last:+.4f} F1 / +{results[-1][1]-results[-2][1]}건")
    if abs(d_last) < 0.01:
        print("→ 곡선 평탄. 데이터를 2배로 늘려도 +0.01 미만 예상 — **데이터 축 포화**.")
        print("   병목은 모델 용량(fastText)이며, 남은 레버는 지식 증류 또는 아키텍처 변경.")
    else:
        print("→ 곡선이 아직 상승 중. 같은 라벨 정의로 데이터를 더 확보할 가치가 있음.")


if __name__ == "__main__":
    main()
