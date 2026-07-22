"""exp-0011 stage 2: the ZBSV reference implementation in Python, reproducing
the sklearn pipeline.

What is reproduced (following the sklearn sources):
  1. preprocess: lowercase → text.lower()
  2. _white_spaces = re.compile(r"\\s\\s+") collapses **only runs of two or more**
     spaces into a single space
  3. split() separates words, and each word is padded to " "+w+" "
  4. for n = minN..maxN, char n-grams are taken with a sliding window.
     If the padded word is shorter than n (offset stays 0 and the while loop is
     never entered), the whole word is counted once and the **n loop ends**
  5. CountVectorizer: the occurrence count of each in-vocabulary n-gram
  6. sublinear_tf: tf = 1 + ln(count)
  7. multiply by idf (smooth_idf, as learned by the vectorizer)
  8. L2 normalization
  9. LinearSVC: decision = w·x + b, turned into a probability by sigmoid(decision)

Equivalence 2 (the Kotlin port) only holds if this reference first matches
sklearn, which is equivalence 1.
"""
import math
import re
import struct
from pathlib import Path

import numpy as np

ART = Path(__file__).parent.parent / "artifacts"
_WHITE_SPACES = re.compile(r"\s\s+")


class ZBSVModel:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            assert f.read(4) == b"ZBSV"
            ver, self.min_n, self.max_n, self.n_terms = struct.unpack("<4I", f.read(16))
            self.sublinear, self.use_idf, _ = struct.unpack("<BBH", f.read(4))
            self.threshold, self.intercept, self.coef_scale = struct.unpack("<3f", f.read(12))
            self.idf = np.frombuffer(f.read(self.n_terms * 4), dtype=np.float32)
            q = np.frombuffer(f.read(self.n_terms), dtype=np.int8)
            self.coef = q.astype(np.float32) * self.coef_scale
            self.vocab = {}
            for i in range(self.n_terms):
                (blen,) = struct.unpack("<H", f.read(2))
                self.vocab[f.read(blen).decode("utf-8")] = i

    def ngrams(self, text: str) -> list[str]:
        text = _WHITE_SPACES.sub(" ", text.lower())
        out = []
        for w in text.split():
            w = " " + w + " "
            w_len = len(w)
            for n in range(self.min_n, self.max_n + 1):
                offset = 0
                out.append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    out.append(w[offset:offset + n])
                if offset == 0:  # a short word is counted once and the n loop ends
                    break
        return out

    def decision(self, text: str) -> float:
        counts: dict[int, int] = {}
        for ng in self.ngrams(text):
            idx = self.vocab.get(ng, -1)
            if idx >= 0:
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return float(self.intercept)

        vals, idxs = [], []
        for idx, c in counts.items():
            tf = 1.0 + math.log(c) if self.sublinear else float(c)
            vals.append(tf * float(self.idf[idx]))
            idxs.append(idx)
        norm = math.sqrt(sum(v * v for v in vals))
        if norm == 0:
            return float(self.intercept)

        acc = 0.0
        for idx, v in zip(idxs, vals):
            acc += (v / norm) * float(self.coef[idx])
        return acc + float(self.intercept)

    def prob_toxic(self, text: str) -> float:
        return 1.0 / (1.0 + math.exp(-self.decision(text)))

    def is_toxic(self, text: str) -> bool:
        return self.prob_toxic(text) >= self.threshold


if __name__ == "__main__":
    m = ZBSVModel(ART / "toxicity_model.zbsv")
    for t in ("ㅅㅂ 뭐래", "좋은 아침입니다", "병신같은 소리 하지마"):
        print(f"{m.prob_toxic(t):.4f}  {t}")
