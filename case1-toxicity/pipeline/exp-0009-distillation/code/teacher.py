"""exp-0009 1단계: 교사(KcELECTRA) 학습 + 무라벨 풀 재라벨링.

엄격성 요건:
  - 교사는 **우리 train만** 본다. val/test는 절대 미사용 (누수 차단).
  - 무라벨 풀은 외부 텍스트만. **원 라벨은 즉시 폐기**하고, 우리 val/test와 겹치는 문장을 제거.
  - 누수 검증을 코드로 수행하고 실패 시 중단한다.

산출물: artifacts/pseudo_labels.csv  (text, teacher_prob, orig_label[대조군용])
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
EXP8 = Path(__file__).parent.parent.parent / "exp-0008-dataset-survey"
sys.path.insert(0, str(EXP2 / "code"))
sys.path.insert(0, str(EXP8 / "code"))

from common import f1_binary, load_split  # noqa: E402
from datasets_survey import load_all  # noqa: E402
from kcelectra import BATCH, LR, MAXLEN, MODEL, SEED, TextDS  # noqa: E402

ART = Path(__file__).parent.parent / "artifacts"
EPOCHS = 2  # exp-0002에서 val 최고였던 지점


def build_pool() -> pd.DataFrame:
    """외부 텍스트만 모은 무라벨 풀. 원 라벨은 대조군 분석용으로만 보관한다."""
    data = load_all()
    ours_train = set(load_split("train")["text"])
    ours_val = set(load_split("val")["text"])
    ours_test = set(load_split("test")["text"])

    frames = [data[k].assign(source=k) for k in ("unsmile", "APEACH", "kor-hate-sentence")]
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")

    # 누수 차단: 우리 데이터(train 포함)와 겹치는 문장 전부 제거
    before = len(pool)
    pool = pool[~pool["text"].isin(ours_train | ours_val | ours_test)].reset_index(drop=True)
    print(f"무라벨 풀: {before} → 누수 제거 후 {len(pool)}건")

    # 누수 검증 (실패 시 중단)
    assert not pool["text"].isin(ours_val).any(), "❌ 풀에 val 문장이 남아있음"
    assert not pool["text"].isin(ours_test).any(), "❌ 풀에 test 문장이 남아있음"
    print("✅ 누수 검증 통과 (풀 ∩ val = ∅, 풀 ∩ test = ∅)")

    return pool.rename(columns={"label": "orig_label"})[["text", "orig_label", "source"]]


def train_teacher():
    """우리 train만으로 교사 학습. val은 성능 확인용으로만 본다(파라미터는 exp-0002에서 고정)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

    train_dl = DataLoader(TextDS(load_split("train"), tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, len(train_dl) * EPOCHS)

    model.train()
    for ep in range(EPOCHS):
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            if i % 150 == 0:
                print(f"  ep{ep} {i}/{len(train_dl)} loss={out.loss.item():.4f}", flush=True)
    return model, tok


@torch.no_grad()
def predict_probs(model, tok, texts: list[str]) -> np.ndarray:
    model.eval()
    df = pd.DataFrame({"text": texts, "label": [0] * len(texts)})
    dl = DataLoader(TextDS(df, tok), batch_size=128)
    out = []
    for batch in dl:
        batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        out.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
    return np.array(out)


def main():
    ART.mkdir(exist_ok=True)
    pool = build_pool()

    print("\n교사 학습 (우리 train만)…")
    model, tok = train_teacher()

    # 교사 성능 확인 (val — 파라미터 선택에는 쓰지 않음, 기록용)
    val = load_split("val")
    pv = predict_probs(model, tok, val["text"].tolist())
    yv = val["label"].tolist()
    best = max(((th, f1_binary(yv, [int(p >= th) for p in pv]))
                for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
    print(f"교사 val F1 = {best[1]['f1']:.4f} (th={best[0]:.3f})")

    print(f"\n무라벨 풀 {len(pool)}건 재라벨링…")
    pool["teacher_prob"] = predict_probs(model, tok, pool["text"].tolist())

    # 교사가 우리 정의로 본 풀의 양성 비율 vs 원 라벨의 양성 비율 (라벨 정의 차이의 직접 증거)
    for th in (0.5,):
        teacher_pos = (pool["teacher_prob"] >= th).mean()
        orig_pos = pool["orig_label"].mean()
        agree = ((pool["teacher_prob"] >= th).astype(int) == pool["orig_label"]).mean()
        print(f"\n[라벨 정의 차이] 원 라벨 양성률 {orig_pos:.3f} vs 교사(우리 정의) 양성률 {teacher_pos:.3f}")
        print(f"                 원 라벨과의 일치율: {agree:.3f}")

    pool.to_csv(ART / "pseudo_labels.csv", index=False)
    print(f"\n저장: {ART / 'pseudo_labels.csv'} ({len(pool)}건)")


if __name__ == "__main__":
    main()
