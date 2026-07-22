"""exp-0012 Phase A (2/3): repeated fastText training — 71 runs in total, fixed
in advance by the spec.

  Q2 : the fairly re-tuned configuration (dim16/bucket250k/n(2,4)), distilled
       data — R=10
  Q3g: OP (dim16/bucket500k/n(2,5)), gold — R=10
  Q3d: OP, gold+distilled — R=10           (the same protocol as exp-0009 final_eval)
  Q4 : OP, the four attribution arms (A/E/D/C0.9) × R=10, val best_f1
       (the distill.py protocol; the student labels are generated once)

Execution structure (revised 2026-07-18): repeating training inside one process
was observed to produce NaN persistently (in the first run: run0 and run1
succeeded, then run2 hit NaN on all six lr halvings), so **each run is isolated
in a fresh ft_worker.py process**. If the worker's own train_with_retry (6
attempts, halving lr) fails outright, the process is restarted up to 3 times,
and every restart count is recorded.
The partial output of that first execution is preserved as
runs_q2_ft.aborted-inprocess.csv rather than deleted — everything is reported,
selected or not.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common12 import (ART, DATA, EXP9_ART, FT_OP_CFG, FT_Q2_CFG, R,
                      append_csv, ensure_dirs)

from benchmark import build_train_sets, pick_threshold  # noqa: E402
from common import f1_binary, load_split  # noqa: E402
from transfer_matrix import best_f1, write_ft  # noqa: E402

HERE = Path(__file__).parent
MAX_PROC_ATTEMPTS = 3


def train_isolated(input_path: str, cfg: dict, out_prefix: str, evals: list[str]) -> dict:
    """One training run in a fresh process. Returns meta(final_lr, sec, proc_attempts)."""
    job = {"input": input_path, "cfg": cfg, "out_prefix": out_prefix, "evals": evals}
    job_path = ART / f"{out_prefix}_job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    last = None
    for attempt in range(1, MAX_PROC_ATTEMPTS + 1):
        r = subprocess.run([sys.executable, str(HERE / "ft_worker.py"), str(job_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            meta = json.loads((ART / f"{out_prefix}_meta.json").read_text(encoding="utf-8"))
            meta["proc_attempts"] = attempt
            job_path.unlink()
            return meta
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f"exit {r.returncode}"
        print(f"  ! {out_prefix}: process attempt {attempt} failed — {last}", flush=True)
    raise RuntimeError(f"{out_prefix}: all {MAX_PROC_ATTEMPTS} process attempts failed ({last})")


def run_test_block(tag, path, cfg, yv, yt):
    """pick_threshold on val, then applied to test — the original protocol behind the test claims. R runs, each isolated in its own process."""
    for i in range(R):
        prefix = f"{tag}_run{i}"
        meta = train_isolated(path, cfg, prefix, ["val", "test"])
        pv = np.load(ART / f"{prefix}_val_probs.npy")
        th, val_f1 = pick_threshold(yv, pv)
        pt = np.load(ART / f"{prefix}_test_probs.npy")
        m = f1_binary(yt, (pt >= th).astype(int).tolist())
        row = {"run": i, "th": round(float(th), 3), "val_f1": round(val_f1, 4),
               **{k: round(v, 6) for k, v in m.items()},
               "final_lr": meta["final_lr"], "proc_attempts": meta["proc_attempts"],
               "sec": meta["sec"]}
        append_csv(ART / f"runs_{tag}.csv", row)
        print(f"[{tag}] run{i}: th={th:.3f} val={val_f1:.4f} test F1={m['f1']:.4f} "
              f"(lr={meta['final_lr']}, proc={meta['proc_attempts']}, {meta['sec']}s)", flush=True)


def q4_arms(ours, pool, val):
    """The distill.py protocol: metric = best_f1(val), student labels generated once."""
    arms = {"A_baseline": ours}

    e_df = pd.concat([ours, pool[["text", "orig_label"]].rename(columns={"orig_label": "label"})],
                     ignore_index=True)
    arms["E_origlabel"] = e_df

    sp_path = ART / "q4_student_val_pool_probs.npy"
    if sp_path.exists():
        sp = np.load(sp_path)
        print("[Q4] reusing the previously generated student labels", flush=True)
    else:
        # Generating the student labels would also want process isolation, but the
        # pool text cannot be passed as a temporary split, so a single training run
        # happens here instead of in the worker. Being this process's first
        # training, it is not affected by the contamination described above.
        from common import train_with_retry  # noqa: E402
        from transfer_matrix import prob_pos  # noqa: E402
        path = write_ft(ours[["text", "label"]], DATA / "q4_student_teacher.txt")
        model, lr = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                     loss="softmax", thread=1, verbose=0, **FT_OP_CFG)
        sp = prob_pos(model, pool["text"])
        np.save(sp_path, sp)
        print(f"[Q4] student labels generated (lr={lr})", flush=True)
    d_df = pd.concat([ours, pd.DataFrame({"text": pool["text"],
                                          "label": (sp >= 0.5).astype(int)})], ignore_index=True)
    arms["D_selftrain"] = d_df

    conf = pool[(pool["teacher_prob"] >= 0.9) | (pool["teacher_prob"] <= 0.1)]
    c_df = pd.concat([ours, pd.DataFrame({"text": conf["text"],
                                          "label": (conf["teacher_prob"] >= 0.5).astype(int)})],
                     ignore_index=True)
    arms["C_teacher09"] = c_df

    yv = val["label"].tolist()
    for arm, df in arms.items():
        path = write_ft(df[["text", "label"]], DATA / f"q4_{arm}.txt")
        for i in range(R):
            prefix = f"q4_{arm}_run{i}"
            meta = train_isolated(path, FT_OP_CFG, prefix, ["val"])
            f1v = best_f1(yv, np.load(ART / f"{prefix}_val_probs.npy"))
            append_csv(ART / "runs_q4.csv",
                       {"arm": arm, "run": i, "n_train": len(df),
                        "val_best_f1": round(f1v, 6), "final_lr": meta["final_lr"],
                        "proc_attempts": meta["proc_attempts"], "sec": meta["sec"]})
            print(f"[Q4:{arm}] run{i}: val best_f1={f1v:.4f} "
                  f"(lr={meta['final_lr']}, proc={meta['proc_attempts']}, {meta['sec']}s)", flush=True)


def main():
    ensure_dirs()
    sections = set(sys.argv[1:]) or {"q2", "q3", "q4"}
    sets = build_train_sets()
    gold, dist = sets["gold only"], sets["gold+distilled"]
    val, test = load_split("val"), load_split("test")
    yv, yt = val["label"].tolist(), test["label"].tolist()

    if "q2" in sections:
        path = write_ft(dist, DATA / "ft_q2_dist.txt")
        run_test_block("q2_ft", path, FT_Q2_CFG, yv, yt)
    if "q3" in sections:
        path_g = write_ft(gold, DATA / "ft_q3_gold.txt")
        run_test_block("q3_ft_gold", path_g, FT_OP_CFG, yv, yt)
        path_d = write_ft(dist, DATA / "ft_q3_dist.txt")
        run_test_block("q3_ft_dist", path_d, FT_OP_CFG, yv, yt)
    if "q4" in sections:
        ours = load_split("train")[["text", "label"]]
        pool = pd.read_csv(EXP9_ART / "pseudo_labels.csv")
        q4_arms(ours, pool, val)

    print("→ all fastText runs complete", flush=True)


if __name__ == "__main__":
    main()
