"""exp-0009 2단계: 증류 실험 + 귀인 대조군 (val 기준, test 봉인 유지).

팔 구성 (모두 fastText 동작점 설정, 3회 반복 평균):
  A. 기준선          : ours only
  B. 교사 증류(전량)  : ours + 풀(교사 하드 라벨)
  C. 교사 증류(신뢰도): ours + 풀(교사 확신 표본만) — 임계 스윕
  D. **자기학습 대조**: ours + 풀(학생 fastText 자신이 라벨링)   ← H2 귀인 검증의 핵심
  E. **원라벨 대조**  : ours + 풀(외부 원래 라벨)                ← exp-0008 재확인
  F. 풀만 (교사 라벨) : ours 없이 풀만

D가 B와 비슷한 이득을 낸다면 "교사의 지식" 귀인은 틀린 것이다 (단순 self-training 효과).
E는 라벨 정의가 문제였음을 같은 텍스트로 재확인한다 (텍스트 통제, 라벨만 변경).
"""
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from common import f1_binary, load_split, train_with_retry  # noqa: E402
from transfer_matrix import OP, best_f1, prob_pos, write_ft  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"
DATA = Path(__file__).parent.parent / "data"
REPEATS = 3


def run(name: str, train_df: pd.DataFrame, val: pd.DataFrame) -> float:
    DATA.mkdir(exist_ok=True)
    path = write_ft(train_df[["text", "label"]], DATA / f"{name.replace(' ', '_')}.txt")
    f1s = []
    for _ in range(REPEATS):
        model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                    loss="softmax", thread=1, verbose=0, **OP)
        f1s.append(best_f1(val["label"].tolist(), prob_pos(model, val["text"])))
    m, s = statistics.mean(f1s), statistics.stdev(f1s)
    print(f"  {name:34s} n={len(train_df):6d} 양성={train_df['label'].mean():.3f} "
          f"→ val F1 = {m:.4f} ± {s:.4f}")
    return m


def student_pseudo_labels(ours: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    """학생(fastText) 자신이 풀에 라벨을 붙인다 — 자기학습 대조군."""
    path = write_ft(ours[["text", "label"]], DATA / "student_teacher.txt")
    model, _ = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                loss="softmax", thread=1, verbose=0, **OP)
    return prob_pos(model, pool["text"])


def main():
    ours = load_split("train")[["text", "label"]]
    val = load_split("val")
    pool = pd.read_csv(ART / "pseudo_labels.csv")

    print(f"풀 {len(pool)}건, 교사 양성률 {(pool['teacher_prob'] >= 0.5).mean():.3f}\n")

    print("=== 기준선 및 대조군 ===")
    base = run("A. 기준선 (ours only)", ours, val)

    # E. 원라벨 대조 (텍스트 동일, 라벨만 외부 원본)
    e_df = pd.concat([ours, pool[["text", "orig_label"]].rename(columns={"orig_label": "label"})],
                     ignore_index=True)
    e = run("E. 원라벨 대조 (외부 라벨)", e_df, val)

    # D. 자기학습 대조 (학생이 라벨링)
    sp = student_pseudo_labels(ours, pool)
    d_df = pd.concat([ours, pd.DataFrame({"text": pool["text"], "label": (sp >= 0.5).astype(int)})],
                     ignore_index=True)
    d = run("D. 자기학습 대조 (학생 라벨)", d_df, val)

    print("\n=== 교사 증류 ===")
    # B. 전량
    b_df = pd.concat([ours, pd.DataFrame({"text": pool["text"],
                                          "label": (pool["teacher_prob"] >= 0.5).astype(int)})],
                     ignore_index=True)
    b = run("B. 교사 증류 (전량)", b_df, val)

    # C. 신뢰도 필터 스윕
    print("\n=== C. 신뢰도 필터 스윕 (교사가 확신하는 표본만) ===")
    best_c, best_margin = -1.0, None
    for margin in (0.6, 0.7, 0.8, 0.9, 0.95):
        conf = pool[(pool["teacher_prob"] >= margin) | (pool["teacher_prob"] <= 1 - margin)]
        c_df = pd.concat([ours, pd.DataFrame({"text": conf["text"],
                                              "label": (conf["teacher_prob"] >= 0.5).astype(int)})],
                         ignore_index=True)
        c = run(f"C. 교사 증류 (신뢰도 ≥{margin})", c_df, val)
        if c > best_c:
            best_c, best_margin = c, margin

    # F. 풀만
    f_df = pd.DataFrame({"text": pool["text"], "label": (pool["teacher_prob"] >= 0.5).astype(int)})
    f = run("F. 풀만 (ours 제외)", f_df, val)

    print("\n=== 요약 (val) ===")
    print(f"  A 기준선          : {base:.4f}")
    print(f"  E 원라벨 대조      : {e:.4f}  ({e - base:+.4f})")
    print(f"  D 자기학습 대조    : {d:.4f}  ({d - base:+.4f})   ← 귀인 검증")
    print(f"  B 교사 증류(전량)  : {b:.4f}  ({b - base:+.4f})")
    print(f"  C 교사 증류(최적)  : {best_c:.4f}  ({best_c - base:+.4f})  @ 신뢰도 {best_margin}")
    print(f"  F 풀만            : {f:.4f}  ({f - base:+.4f})")
    print()
    if best_c > base and best_c - d > 0.01:
        print("→ H2 지지: 교사 증류의 이득이 자기학습 대조군을 유의하게 상회 (교사 지식 전달로 귀인 가능)")
    elif best_c > base:
        print("→ H2 주의: 개선은 있으나 자기학습 대조군과 차이가 작음 — 귀인 재검토 필요")
    else:
        print("→ H1 기각: 교사 증류가 기준선을 넘지 못함")


if __name__ == "__main__":
    main()
