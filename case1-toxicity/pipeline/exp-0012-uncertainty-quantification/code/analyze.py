"""exp-0012 Phase A (3/3): analysis — reads only the saved outputs and computes
the confidence intervals, superiority rates and verdicts.

Pre-registered (spec): bootstrap seed=20260718, B=10,000, percentile CI.
Paired comparisons share the same resample indices; the whole analysis uses one
index matrix. The decision rules are applied mechanically, exactly as fixed in
the spec's decision-rule section before any of this was run.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common12 import ART, B, BOOT_SEED, R

from common import load_split  # noqa: E402


def f1_boot(y, pred, idx):
    """F1 vector over the resample index matrix (idx: B×n)."""
    tp = ((y == 1) & (pred == 1)).astype(np.int8)
    fp = ((y == 0) & (pred == 1)).astype(np.int8)
    fn = ((y == 1) & (pred == 0)).astype(np.int8)
    TP = tp[idx].sum(axis=1).astype(np.float64)
    FP = fp[idx].sum(axis=1).astype(np.float64)
    FN = fn[idx].sum(axis=1).astype(np.float64)
    denom = 2 * TP + FP + FN
    return np.where(denom > 0, 2 * TP / denom, 0.0)


def ci(v, lo=2.5, hi=97.5):
    return [round(float(np.percentile(v, lo)), 4), round(float(np.percentile(v, hi)), 4)]


def welch_ci(a, b, alpha=0.05):
    """Welch 95% CI for the difference of means (a−b).

    If both variances are zero (deterministic training, as observed in practice)
    the CI degenerates to a point. In that case [d, d] is returned, and the
    decision rule "the CI excludes 0" becomes equivalent to d≠0.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        return round(float(d), 4), [round(float(d), 4), round(float(d), 4)]
    se = np.sqrt(va / len(a) + vb / len(b))
    df = se**4 / (((va / len(a))**2 / (len(a) - 1)) + ((vb / len(b))**2 / (len(b) - 1))
                  or np.finfo(float).tiny)
    t = stats.t.ppf(1 - alpha / 2, df)
    return round(float(d), 4), [round(float(d - t * se), 4), round(float(d + t * se), 4)]


def preds_from(tag, th):
    return (np.load(ART / f"{tag}_test_probs.npy") >= th).astype(np.int8)


def main():
    test = load_split("test")
    y = test["label"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(B, n), dtype=np.int32)

    svm = json.load(open(ART / "svm_results.json", encoding="utf-8"))
    out = {"seed": BOOT_SEED, "B": B, "n_test": int(n)}

    # ── Q1: CI for what is actually deployed (int8) ──────────────
    p1 = preds_from("q1_int8", svm["q1"]["th"])
    b1 = f1_boot(y, p1, idx)
    out["Q1"] = {"point": svm["q1"]["f1"], "ci95": ci(b1)}

    # ── Q2: the rank reversal ────────────────────────────────────
    svm_pred = preds_from("q2_svm", svm["q2_svm"]["th"])
    b_svm = f1_boot(y, svm_pred, idx)
    runs2 = pd.read_csv(ART / "runs_q2_ft.csv")
    assert len(runs2) == R, f"Q2 run count {len(runs2)} ≠ planned {R}"
    sup_rates, wins = [], 0
    for _, r in runs2.iterrows():
        ft_pred = preds_from(f"q2_ft_run{int(r['run'])}", r["th"])
        b_ft = f1_boot(y, ft_pred, idx)
        sup_rates.append(round(float((b_svm > b_ft).mean()), 4))
        wins += int(svm["q2_svm"]["f1"] > r["f1"])
    med_sup = float(np.median(sup_rates))
    verdict2 = ("supported" if wins == R and med_sup >= 0.95
                else ("rejected" if svm["q2_svm"]["f1"] < runs2["f1"].mean() else "weakened"))
    out["Q2"] = {"svm_point": svm["q2_svm"]["f1"],
                 "ft_runs_f1": [round(v, 4) for v in runs2["f1"]],
                 "ft_mean": round(float(runs2["f1"].mean()), 4),
                 "ft_std": round(float(runs2["f1"].std(ddof=1)), 4),
                 "wins": f"{wins}/{R}", "sup_rates": sup_rates,
                 "sup_median": round(med_sup, 4), "verdict": verdict2}

    # ── Q3 (SVM): the distillation gain, paired bootstrap ────────
    g = preds_from("q3_svm_gold", svm["q3_svm_gold"]["th"])
    d = preds_from("q3_svm_dist", svm["q3_svm_dist"]["th"])
    delta = f1_boot(y, d, idx) - f1_boot(y, g, idx)
    point = round(svm["q3_svm_dist"]["f1"] - svm["q3_svm_gold"]["f1"], 4)
    ci3s = ci(delta)
    out["Q3_svm"] = {"point": point, "ci95": ci3s,
                     "verdict": "supported" if ci3s[0] > 0 else ("rejected" if point < 0 else "weakened")}

    # ── Q3 (fastText): Welch at the run level ────────────────────
    rg = pd.read_csv(ART / "runs_q3_ft_gold.csv")
    rd = pd.read_csv(ART / "runs_q3_ft_dist.csv")
    assert len(rg) == R and len(rd) == R, "Q3 fastText run counts do not match"
    d3, ci3f = welch_ci(rd["f1"], rg["f1"])
    out["Q3_ft"] = {"gold_mean": round(float(rg["f1"].mean()), 4),
                    "gold_std": round(float(rg["f1"].std(ddof=1)), 4),
                    "dist_mean": round(float(rd["f1"].mean()), 4),
                    "dist_std": round(float(rd["f1"].std(ddof=1)), 4),
                    "delta": d3, "ci95": ci3f,
                    "verdict": "supported" if ci3f[0] > 0 else ("rejected" if d3 < 0 else "weakened")}

    # ── Q4: the attribution ordering, Welch ──────────────────────
    r4 = pd.read_csv(ART / "runs_q4.csv")
    arms = {a: r4[r4["arm"] == a]["val_best_f1"].to_numpy() for a in r4["arm"].unique()}
    for a, v in arms.items():
        assert len(v) == R, f"Q4 {a} run count {len(v)} ≠ {R}"
    q4 = {}
    ok = True
    for other in ("A_baseline", "E_origlabel", "D_selftrain"):
        dlt, c = welch_ci(arms["C_teacher09"], arms[other])
        q4[f"C-{other[0]}"] = {"delta": dlt, "ci95": c}
        ok &= c[0] > 0
    directions = all(arms["C_teacher09"].mean() > arms[o].mean()
                     for o in ("A_baseline", "E_origlabel", "D_selftrain"))
    q4["arm_mean_sd"] = {a: f"{v.mean():.4f}±{v.std(ddof=1):.4f}" for a, v in arms.items()}
    q4["secondary_E_lt_A_D_lt_A"] = {"E<A": bool(arms["E_origlabel"].mean() < arms["A_baseline"].mean()),
                            "D<A": bool(arms["D_selftrain"].mean() < arms["A_baseline"].mean())}
    q4["verdict"] = "supported" if ok else ("rejected" if not directions else "weakened")
    out["Q4"] = q4

    with open(ART / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("→ saved artifacts/analysis.json")


if __name__ == "__main__":
    main()
