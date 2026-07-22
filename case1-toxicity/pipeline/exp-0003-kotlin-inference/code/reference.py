"""exp-0003 Python reference implementation: reproduces fastText supervised
inference from the ZBFT format.

It follows the inference path of the fastText C++ sources (dictionary.cc,
model.cc) exactly:
  1. whitespace tokenization plus a sentence-final EOS ('</s>')
  2. in-vocabulary word: [word id] + its char n-gram ids / OOV word: char n-gram
     ids only
     - char n-grams run over '<'+word+'>' by UTF-8 character, for minn..maxn
     - EOS has no char n-grams (its word id only)
  3. word bigrams (wordNgrams=2): h = uint64(hash(w_i)) * 116049371 + hash(w_{i+1});
     id = nwords + (h % bucket). The hash is 32-bit FNV-1a, XOR-ing each byte
     after sign-extending it as int8.
     The detail that matters: in C++ the hashes are held in a vector<int32_t>
     and then promoted to uint64_t, so any hash with its top bit set is
     **sign-extended** (0xFFFFFFFF........). Reproducing that is what makes the
     implementations equivalent.
  4. hidden = the mean of all selected rows → softmax(output @ hidden)

Equivalence 2 (the Kotlin port) only holds if this reference first matches the
fasttext library itself, which is equivalence 1.
"""
import json
import struct
from pathlib import Path

import numpy as np

ART = Path(__file__).parent.parent / "artifacts"
BOW, EOW, EOS = "<", ">", "</s>"


def fnv1a(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8"):
        signed = b - 256 if b > 127 else b            # int8_t sign extension
        h = (h ^ (signed & 0xFFFFFFFF)) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class ZBFTModel:
    def __init__(self, path: Path, dequant: bool = True):
        with open(path, "rb") as f:
            assert f.read(4) == b"ZBFT"
            ver, self.dim, self.nwords, self.bucket, self.minn, self.maxn, self.wn = \
                struct.unpack("<7I", f.read(28))
            (self.threshold,) = struct.unpack("<f", f.read(4))
            self.output = np.frombuffer(f.read(2 * self.dim * 4), dtype=np.float32).reshape(2, self.dim)
            rows = self.nwords + self.bucket
            self.scales = np.frombuffer(f.read(rows * 4), dtype=np.float32)
            q = np.frombuffer(f.read(rows * self.dim), dtype=np.int8).reshape(rows, self.dim)
            self.vocab = {}
            for i in range(self.nwords):
                (n,) = struct.unpack("<H", f.read(2))
                self.vocab[f.read(n).decode("utf-8")] = i
        # dequantized matrix (this reference dequantizes everything up front for
        # convenience; the Kotlin port dequantizes lazily, row by row)
        self.matrix = q.astype(np.float32) * self.scales[:, None]

    def char_ngrams(self, word: str) -> list[int]:
        ids = []
        w = BOW + word + EOW
        chars = list(w)
        for i in range(len(chars)):
            for n in range(1, self.maxn + 1):
                j = i + n
                if j > len(chars):
                    break
                if n >= self.minn and not (n == 1 and (i == 0 or j == len(chars))):
                    ng = "".join(chars[i:j])
                    ids.append(self.nwords + fnv1a(ng) % self.bucket)
        return ids

    def line_ids(self, text: str) -> list[int]:
        tokens = text.split() + [EOS]
        ids, hashes = [], []
        for tok in tokens:
            wid = self.vocab.get(tok, -1)
            if wid >= 0:
                ids.append(wid)
                if tok != EOS and self.maxn > 0:
                    ids.extend(self.char_ngrams(tok))
            elif tok != EOS:
                ids.extend(self.char_ngrams(tok))
            hashes.append(fnv1a(tok))
        def sext64(u32: int) -> int:
            """Reproduces the int32_t → uint64_t sign extension (as in fastText's addWordNgrams)."""
            return (u32 - (1 << 32) if u32 >= (1 << 31) else u32) & 0xFFFFFFFFFFFFFFFF

        for i in range(len(hashes)):                   # word n-grams
            h = sext64(hashes[i])
            for j in range(i + 1, min(i + self.wn, len(hashes))):
                h = (h * 116049371 + sext64(hashes[j])) & 0xFFFFFFFFFFFFFFFF
                ids.append(self.nwords + h % self.bucket)
        return ids

    def prob_positive(self, text: str) -> float:
        ids = self.line_ids(text)
        hidden = self.matrix[ids].mean(axis=0)
        logits = self.output @ hidden
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())


if __name__ == "__main__":
    m = ZBFTModel(ART / "toxicity_model.zbft")
    for t in ("ㅅㅂ 뭐래", "좋은 아침입니다"):
        print(t, "→ P(toxic) =", round(m.prob_positive(t), 4))
