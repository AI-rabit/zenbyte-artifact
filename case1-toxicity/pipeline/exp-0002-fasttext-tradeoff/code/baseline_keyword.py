"""exp-0002 naive baseline: substring matching against a profanity lexicon.

The lexicon is a minimal hand-built list (common profanity stems plus their
jamo abbreviations) and is kept deliberately simple — its only job is to answer
"how much better than dictionary matching is fastText?".

The entries are Korean profanity: they are the baseline itself, so they are
left exactly as they are.
"""
from common import f1_binary, load_split

LEXICON = [
    # common stems (substring matching, so some variants are covered too)
    "씨발", "시발", "씨빨", "시빨", "씨팔", "병신", "븅신", "빙신", "개새끼", "개색기", "개세끼",
    "새끼", "존나", "졸라", "좆", "닥쳐", "꺼져", "지랄", "염병", "미친놈", "미친년", "또라이",
    "등신", "멍청이", "호구", "걸레", "창녀", "한남", "김치녀", "된장녀", "맘충", "급식충",
    "틀딱", "짱깨", "쪽바리", "흑형", "빨갱이",
    # jamo abbreviations
    "ㅅㅂ", "ㅆㅂ", "ㅄ", "ㅂㅅ", "ㅈㄹ", "ㅗ",
]


def predict(text: str) -> int:
    return 1 if any(term in text for term in LEXICON) else 0


def main():
    for split in ("val",):  # test stays sealed
        df = load_split(split)
        y_pred = [predict(t) for t in df["text"]]
        m = f1_binary(df["label"].tolist(), y_pred)
        print(f"[{split}] lexicon of {len(LEXICON)} entries: "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
