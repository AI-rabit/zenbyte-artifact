"""exp-0002 accuracy ceiling: fine-tuning KcELECTRA (GPU; not a deployment
candidate).

Purpose: measure a transformer's F1 on the same data and the same val split, to
obtain a reference that settles whether the 0.80 target is limited by the data
or by the model. The resulting model is never brought into the app — deployment
of a transformer was already ruled out in exp-0001.
"""
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ARTIFACTS, f1_binary, load_split

MODEL = "beomi/KcELECTRA-base-v2022"
EPOCHS = 3
BATCH = 32
LR = 3e-5
MAXLEN = 128
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class TextDS(Dataset):
    def __init__(self, df, tok):
        self.enc = tok(df["text"].tolist(), truncation=True, max_length=MAXLEN,
                       padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(df["label"].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}


def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

    train_dl = DataLoader(TextDS(load_split("train"), tok), batch_size=BATCH, shuffle=True)
    val = load_split("val")
    val_dl = DataLoader(TextDS(val, tok), batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = len(train_dl) * EPOCHS
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, steps)

    for ep in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            if i % 100 == 0:
                print(f"ep{ep} step{i}/{len(train_dl)} loss={out.loss.item():.4f}", flush=True)

        model.eval()
        probs = []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
                probs.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
        y = val["label"].tolist()
        argmax_f1 = f1_binary(y, [int(p >= 0.5) for p in probs])
        best = max(((th, f1_binary(y, [int(p >= th) for p in probs]))
                    for th in np.arange(0.05, 0.95, 0.025)), key=lambda x: x[1]["f1"])
        print(json.dumps({"epoch": ep, "argmax_f1": round(argmax_f1["f1"], 4),
                          "best_th": round(float(best[0]), 3), "f1": round(best[1]["f1"], 4),
                          "precision": round(best[1]["precision"], 3),
                          "recall": round(best[1]["recall"], 3)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
