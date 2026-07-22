"""exp-0013 Phase B: three teacher seeds ({42,43,44}) — variance of the ceiling,
and the cascade (how the distillation gain varies with the teacher).

Per seed: train the teacher (our train split only, 2 epochs — the exp-0009
recipe unchanged) → evaluate on val/test
→ relabel the 34,305-row pool (the text is exactly that of exp-0009
  pseudo_labels.csv, so the text is held fixed)
→ filter at ≥0.9 → student SVM (the deployed operating point, deterministic)
  → evaluate on test
→ fastText on the same distilled data (Q7, one run, process-isolated).

The gold-only student (char_wb(2,4), deterministic) is trained once as the
common baseline.
The leakage assertions (pool ∩ val/test = ∅) are re-checked, and a violation
stops the run, as in the original protocol.
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
        raise RuntimeError(f"{prefix}: fastText worker failed — {r.stderr.strip().splitlines()[-1]}")
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

    # re-check for leakage (the original protocol's assertions; failure stops the run)
    assert not pool["text"].isin(set(val["text"])).any(), "❌ val sentences in the pool"
    assert not pool["text"].isin(set(test["text"])).any(), "❌ test sentences in the pool"
    print(f"✅ leakage re-check passed (pool of {len(pool)} ∩ val/test = ∅)", flush=True)

    # common baseline: the gold-only student (deterministic, trained once)
    gold_metrics = fit_student(ours, val, test, "student_goldonly")
    append_csv(ART / "cascade.csv", {"seed": "gold-only", "role": "student_svm", **gold_metrics})
    print(f"[baseline] gold-only student test F1={gold_metrics['f1']:.4f}", flush=True)

    for seed in TEACHER_SEEDS:
        t0 = time.time()
        print(f"\n=== teacher seed {seed} ===", flush=True)
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
        print(f"[teacher seed{seed}] val={tv:.4f} test F1={mt['f1']:.4f} "
              f"positive_rate={pos_rate:.3f} passed_0.9={int(conf_mask.sum())}", flush=True)

        del model
        torch.cuda.empty_cache()

        conf = pool[conf_mask].assign(label=(probs[conf_mask] >= 0.5).astype(int))
        dist = pd.concat([ours, conf[["text", "label"]]], ignore_index=True)

        sm = fit_student(dist, val, test, f"student_seed{seed}")
        append_csv(ART / "cascade.csv", {"seed": seed, "role": "student_svm",
                                         "n_train": len(dist), **sm})
        print(f"[student seed{seed}] test F1={sm['f1']:.4f} (vs gold-only "
              f"{sm['f1'] - gold_metrics['f1']:+.4f})", flush=True)

        ft_path = write_ft(dist[["text", "label"]], DATA / f"ft_dist_seed{seed}.txt")
        fm = run_fasttext_isolated(str(ft_path), f"ft_seed{seed}", val, test)
        append_csv(ART / "cascade.csv", {"seed": seed, "role": "student_fasttext",
                                         "n_train": len(dist), **fm})
        print(f"[fastText seed{seed}] test F1={fm['f1']:.4f} (vs SVM "
              f"{sm['f1'] - fm['f1']:+.4f})", flush=True)

    print("\n→ Phase B complete", flush=True)


if __name__ == "__main__":
    main()
