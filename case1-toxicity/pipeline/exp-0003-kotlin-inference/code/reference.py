"""exp-0003 Python 참조 구현: fastText supervised 추론을 ZBFT 포맷에서 재현.

fastText C++ 소스(dictionary.cc, model.cc)의 추론 경로를 그대로 따른다:
  1. 공백 토큰화 + 문말 EOS('</s>')
  2. in-vocab 단어: [단어 id] + char n-gram id들 / OOV 단어: char n-gram id들만
     - char n-gram은 '<'+word+'>'에 대해 UTF-8 문자 단위로 minn..maxn
     - EOS는 char n-gram 없음 (단어 id만)
  3. 단어 bigram (wordNgrams=2): h = uint64(hash(w_i)) * 116049371 + hash(w_{i+1});
     id = nwords + (h % bucket)   ※ hash는 FNV-1a 32bit, 바이트를 int8로 부호확장 후 XOR
     ※※ 핵심 디테일: C++에서 해시가 vector<int32_t>에 저장됐다가 uint64_t로 승격되므로
        상위비트가 1인 해시는 **부호 확장**된다 (0xFFFFFFFF........). 이를 재현해야 동치.
  4. hidden = 모든 id 행의 평균 → softmax(output @ hidden)

이 참조 구현이 fasttext 라이브러리와 일치해야(동치성 1) Kotlin 포팅(동치성 2)이 성립한다.
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
        signed = b - 256 if b > 127 else b            # int8_t 부호확장
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
        # 역양자화 행렬 (참조 구현은 편의상 전체 역양자화; Kotlin은 행 단위 지연 역양자화)
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
            """int32_t → uint64_t 부호 확장 재현 (fastText addWordNgrams와 동일)."""
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
        print(t, "→ P(독성) =", round(m.prob_positive(t), 4))
