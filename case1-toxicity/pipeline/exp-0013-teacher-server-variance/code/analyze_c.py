"""exp-0013 Phase C 분석: seed_sweep.csv → Q8~Q11 판정 (spec 사전 등록 기준 기계 적용)."""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common13 import ART

EXPECTED_SEEDS = 30


def t_ci(v, alpha=0.05):
    v = np.asarray(v, float)
    m = v.mean()
    if len(v) < 2 or v.std(ddof=1) == 0:
        return round(float(m), 6), [round(float(m), 6)] * 2
    se = v.std(ddof=1) / np.sqrt(len(v))
    t = stats.t.ppf(1 - alpha / 2, len(v) - 1)
    return round(float(m), 6), [round(float(m - t * se), 6), round(float(m + t * se), 6)]


def main():
    df = pd.read_csv(ART / "seed_sweep.csv")
    assert len(df) == EXPECTED_SEEDS, f"seed 수 {len(df)} ≠ 계획 {EXPECTED_SEEDS}"
    out = {"n_seeds": len(df)}

    fp_mean, fp_ci = t_ci(df["fp_rate"])
    out["Q8_fp"] = {"원기록": 0.002, "mean": fp_mean, "ci95": fp_ci,
                    "max": float(df["fp_rate"].max()),
                    "seeds_over_5pct": int((df["fp_rate"] > 0.05).sum()),
                    "판정": "유지" if (df["fp_rate"] <= 0.05).all() else "약화"}

    b = df["burst_delay"]
    out["Q9_burst"] = {"원기록": "즉시(≤3s)", "min": int(b.min()), "max": int(b.max()),
                       "미탐": int((b < 0).sum()),
                       "판정": "유지" if ((b >= 0) & (b <= 3)).all() else "약화"}

    r = df["ramp_delay"]
    out["Q10_ramp"] = {"원기록": "20s(≤30s)", "min": int(r.min()), "max": int(r.max()),
                       "median": float(r.median()), "미탐": int((r < 0).sum()),
                       "판정": "유지" if ((r >= 0) & (r <= 30)).all() else "약화"}

    ls = df["lowslow_delay"]
    detected = int((ls >= 0).sum())
    out["Q11_lowslow"] = {"원기록": "구조적 미탐", "탐지된 seed 수": detected,
                          "판정": "유지" if detected == 0 else
                          ("약화" if detected <= 1 else "기각"),
                          "메모": ("" if detected == 0 else
                                   "⚠️ 논문 '구조적 미탐' 표현의 수정 소요 후보 — 동결 절차 별도")}

    with open(ART / "analysis_c.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
