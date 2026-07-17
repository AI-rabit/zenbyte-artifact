"""exp-0013 Phase B: 교사 seed 3종({42,43,44}) — 상한선 분산 + 캐스케이드(증류 이득의 교사-분산).

seed당: 교사 학습(우리 train만, 2ep — exp-0009 레시피 그대로) → val/test 평가
→ 풀 34,305건 재라벨(텍스트는 exp-0009 pseudo_labels.csv 그대로 — 텍스트 통제)
→ ≥0.9 필터 → 학생 SVM(배포 동작점, 결정적) → test 평가
→ 같은 증류 데이터로 fastText(Q7, 프로세스 격리 1런).

골드-only 학생(char_wb(2,4), 결정적)은 공통 기준선으로 1회 학습.
누수 assert(풀 ∩ val/test = ∅)를 재확인하고 위반 시 중단한다 (원프로토콜).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common13 import (ART, CONF, DATA, EXP9_ART, SVC_KW, SVM_VEC,
                      TEACHER_EPOCHS, TEACHER_SEEDS, append_csv, ensure_dirs)

from benchmark import pick_threshold, svm_decision  # noqa: E402
from common import f1_binary, load_split  # noqa: E402
from kcelectra import BATCH, LR, MODEL, TextDS  # noqa: E402
from transfer_matrix import write_ft  # noqa: E402

HERE = Path(__file__).parent


def train_teacher(train_df, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    dl = DataLoader(TextDS(train_df, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, len(dl) * TEACHER_EPOCHS)
    model.train()
    for ep in range(TEACHER_EPOCHS):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            if i % 150 == 0:
                print(f"  seed{seed} ep{ep} {i}/{len(dl)} loss={out.loss.item():.4f}", flush=True)
    return model, tok


@torch.no_grad()
def predict_probs(model, tok, texts):
    model.eval()
    df = pd.DataFrame({"text": list(texts), "label": [0] * len(texts)})
    dl = DataLoader(TextDS(df, tok), batch_size=128)
    out = []
    for batch in dl:
        batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
        out.extend(torch.softmax(model(**batch).logits, -1)[:, 1].cpu().tolist())
    return np.array(out)


def fit_student(tr, val, test, tag):
    vec = TfidfVectorizer(**SVM_VEC)
    clf = LinearSVC(**SVC_KW)
    clf.fit(vec.fit_transform(tr["text"]), tr["label"])
    pv = svm_decision(clf, vec.transform(val["text"]))
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = svm_decision(clf, vec.transform(test["text"]))
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())
    np.save(ART / f"{tag}_test_probs.npy", pt)
    return {"th": round(float(th), 3), "val_f1": round(val_f1, 4),
            **{k: round(v, 6) for k, v in m.items()}}


def run_fasttext_isolated(train_path, prefix, val, test):
    r = subprocess.run([sys.executable, str(HERE / "ft_worker13.py"), train_path, prefix],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{prefix}: fastText 워커 실패 — {r.stderr.strip().splitlines()[-1]}")
    pv = np.load(ART / f"{prefix}_val_probs.npy")
    th, val_f1 = pick_threshold(val["label"].tolist(), pv)
    pt = np.load(ART / f"{prefix}_test_probs.npy")
    m = f1_binary(test["label"].tolist(), (pt >= th).astype(int).tolist())
    meta = json.loads((ART / f"{prefix}_meta.json").read_text(encoding="utf-8"))
    return {"th": round(float(th), 3), "val_f1": round(val_f1, 4),
            **{k: round(v, 6) for k, v in m.items()}, "final_lr": meta["final_lr"]}


def main():
    ensure_dirs()
    ours = load_split("train")[["text", "label"]]
    val, test = load_split("val"), load_split("test")
    pool = pd.read_csv(EXP9_ART / "pseudo_labels.csv")[["text", "orig_label", "source"]]

    # 누수 재확인 (원프로토콜의 assert 승계 — 실패 시 중단)
    assert not pool["text"].isin(set(val["text"])).any(), "❌ 풀에 val 문장"
    assert not pool["text"].isin(set(test["text"])).any(), "❌ 풀에 test 문장"
    print(f"✅ 누수 재확인 통과 (풀 {len(pool)}건 ∩ val/test = ∅)", flush=True)

    # 공통 기준선: 골드-only 학생 (결정적, 1회)
    gold_metrics = fit_student(ours, val, test, "student_goldonly")
    append_csv(ART / "cascade.csv", {"seed": "gold-only", "role": "student_svm", **gold_metrics})
    print(f"[기준선] 골드-only 학생 test F1={gold_metrics['f1']:.4f}", flush=True)

    for seed in TEACHER_SEEDS:
        t0 = time.time()
        print(f"\n=== 교사 seed {seed} ===", flush=True)
        model, tok = train_teacher(ours, seed)

        pv = predict_probs(model, tok, val["text"].tolist())
        th, tv = pick_threshold(val["label"].tolist(), pv)
        pt = predict_probs(model, tok, test["text"].tolist())
        mt = f1_binary(test["label"].tolist(), [int(p >= th) for p in pt])
        np.save(ART / f"teacher_seed{seed}_test_probs.npy", pt)

        probs = predict_probs(model, tok, pool["text"].tolist())
        np.save(ART / f"teacher_seed{seed}_pool_probs.npy", probs)
        pos_rate = float((probs >= 0.5).mean())
        conf_mask = (probs >= CONF) | (probs <= 1 - CONF)
        append_csv(ART / "teachers.csv",
                   {"seed": seed, "val_f1": round(tv, 4), "val_th": round(float(th), 3),
                    **{f"test_{k}": round(v, 6) for k, v in mt.items()},
                    "pool_pos_rate": round(pos_rate, 4), "n_conf09": int(conf_mask.sum()),
                    "gpu_sec": round(time.time() - t0, 1)})
        print(f"[교사 seed{seed}] val={tv:.4f} test F1={mt['f1']:.4f} "
              f"양성률={pos_rate:.3f} ≥0.9통과={int(conf_mask.sum())}", flush=True)

        del model
        torch.cuda.empty_cache()

        conf = pool[conf_mask].assign(label=(probs[conf_mask] >= 0.5).astype(int))
        dist = pd.concat([ours, conf[["text", "label"]]], ignore_index=True)

        sm = fit_student(dist, val, test, f"student_seed{seed}")
        append_csv(ART / "cascade.csv", {"seed": seed, "role": "student_svm",
                                         "n_train": len(dist), **sm})
        print(f"[학생 seed{seed}] test F1={sm['f1']:.4f} (골드-only 대비 "
              f"{sm['f1'] - gold_metrics['f1']:+.4f})", flush=True)

        ft_path = write_ft(dist[["text", "label"]], DATA / f"ft_dist_seed{seed}.txt")
        fm = run_fasttext_isolated(str(ft_path), f"ft_seed{seed}", val, test)
        append_csv(ART / "cascade.csv", {"seed": seed, "role": "student_fasttext",
                                         "n_train": len(dist), **fm})
        print(f"[fastText seed{seed}] test F1={fm['f1']:.4f} (SVM 대비 "
              f"{sm['f1'] - fm['f1']:+.4f})", flush=True)

    print("\n→ Phase B 전 실행 완료", flush=True)


if __name__ == "__main__":
    main()
