"""exp-0002 바보 베이스라인: 욕설 사전 부분문자열 매칭.

사전은 수작업 최소 구성(대표 욕설 어근 + 자모 축약형). 의도적으로 단순하게 유지한다 —
이 베이스라인의 역할은 "fastText가 사전 매칭보다 얼마나 나은가"의 기준선이다.
"""
from common import f1_binary, load_split

LEXICON = [
    # 대표 어근 (부분문자열 매칭이므로 변형 일부 포괄)
    "씨발", "시발", "씨빨", "시빨", "씨팔", "병신", "븅신", "빙신", "개새끼", "개색기", "개세끼",
    "새끼", "존나", "졸라", "좆", "닥쳐", "꺼져", "지랄", "염병", "미친놈", "미친년", "또라이",
    "등신", "멍청이", "호구", "걸레", "창녀", "한남", "김치녀", "된장녀", "맘충", "급식충",
    "틀딱", "짱깨", "쪽바리", "흑형", "빨갱이",
    # 자모 축약형
    "ㅅㅂ", "ㅆㅂ", "ㅄ", "ㅂㅅ", "ㅈㄹ", "ㅗ",
]


def predict(text: str) -> int:
    return 1 if any(term in text for term in LEXICON) else 0


def main():
    for split in ("val",):  # test는 봉인 유지
        df = load_split(split)
        y_pred = [predict(t) for t in df["text"]]
        m = f1_binary(df["label"].tolist(), y_pred)
        print(f"[{split}] 사전 {len(LEXICON)}항목: "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
