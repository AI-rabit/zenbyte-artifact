"""exp-0003 가중치 추출: operating_point.bin → ZBFT 바이너리 포맷 (Kotlin 반입용).

포맷 v1 (little-endian):
  magic   4B  'ZBFT'
  version u32 = 1
  dim, nwords, bucket, minn, maxn, wordNgrams : u32 ×6
  threshold f32  (val에서 선택된 판정 임계값)
  output  f32 × (2·dim)          # row0=__label__0, row1=__label__1
  scales  f32 × (nwords+bucket)  # 행별 int8 역양자화 스케일
  matrix  i8  × ((nwords+bucket)·dim)
  vocab   nwords × { u16 len + utf8 bytes }  # id 순서
"""
import json
import struct
import sys
from pathlib import Path

import fasttext
import numpy as np

EXP2 = Path(__file__).parent.parent.parent / "exp-0002-fasttext-tradeoff"
sys.path.insert(0, str(EXP2 / "code"))

OUT = Path(__file__).parent.parent / "artifacts"
THRESHOLD = 0.375

fasttext.FastText.eprint = lambda x: None


def main():
    OUT.mkdir(exist_ok=True)
    m = fasttext.load_model(str(EXP2 / "artifacts" / "operating_point.bin"))
    a = m.f.getArgs()
    words = m.get_words()
    inp = m.get_input_matrix()            # (nwords+bucket, dim) f32
    out = m.get_output_matrix().astype(np.float32)  # (2, dim)
    assert m.get_labels() == ["__label__0", "__label__1"]

    # 행별 int8 양자화
    scales = np.abs(inp).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.round(inp / scales[:, None]), -127, 127).astype(np.int8)

    path = OUT / "toxicity_model.zbft"
    with open(path, "wb") as f:
        f.write(b"ZBFT")
        f.write(struct.pack("<7I", 1, a.dim, len(words), a.bucket, a.minn, a.maxn, a.wordNgrams))
        f.write(struct.pack("<f", THRESHOLD))
        f.write(out.tobytes())
        f.write(scales.astype(np.float32).tobytes())
        f.write(q.tobytes())
        for w in words:
            b = w.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    meta = {"dim": a.dim, "nwords": len(words), "bucket": a.bucket, "minn": a.minn,
            "maxn": a.maxn, "wordNgrams": a.wordNgrams, "threshold": THRESHOLD,
            "bytes": path.stat().st_size, "mb": round(path.stat().st_size / 2**20, 2)}
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
