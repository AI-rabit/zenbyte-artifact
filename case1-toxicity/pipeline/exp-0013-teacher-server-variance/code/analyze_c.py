"""exp-0013 Phase C analysis: seed_sweep.csv → verdicts for Q8–Q11, applying the
spec's pre-registered decision rules mechanically."""
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
    assert len(df) == EXPECTED_SEEDS, f"seed count {len(df)} ≠ planned {EXPECTED_SEEDS}"
    out = {"n_seeds": len(df)}

    fp_mean, fp_ci = t_ci(df["fp_rate"])
    out["Q8_fp"] = {"originally_recorded": 0.002, "mean": fp_mean, "ci95": fp_ci,
                    "max": float(df["fp_rate"].max()),
                    "seeds_over_5pct": int((df["fp_rate"] > 0.05).sum()),
                    "verdict": "supported" if (df["fp_rate"] <= 0.05).all() else "weakened"}

    b = df["burst_delay"]
    out["Q9_burst"] = {"originally_recorded": "immediate (≤3s)", "min": int(b.min()), "max": int(b.max()),
                       "missed": int((b < 0).sum()),
                       "verdict": "supported" if ((b >= 0) & (b <= 3)).all() else "weakened"}

    r = df["ramp_delay"]
    out["Q10_ramp"] = {"originally_recorded": "20s (≤30s)", "min": int(r.min()), "max": int(r.max()),
                       "median": float(r.median()), "missed": int((r < 0).sum()),
                       "verdict": "supported" if ((r >= 0) & (r <= 30)).all() else "weakened"}

    ls = df["lowslow_delay"]
    detected = int((ls >= 0).sum())
    out["Q11_lowslow"] = {"originally_recorded": "structurally undetectable", "seeds_detected": detected,
                          "verdict": "supported" if detected == 0 else
                          ("weakened" if detected <= 1 else "rejected"),
                          "note": ("" if detected == 0 else
                                   "⚠️ candidate for revising the paper's wording on 'structurally undetectable' — the freeze procedure is handled separately")}

    with open(ART / "analysis_c.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
