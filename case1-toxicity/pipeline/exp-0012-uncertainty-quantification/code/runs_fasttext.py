"""exp-0012 Phase A (2/3): fastText 반복 학습 — 총 71런, 사전 고정 (spec).

  Q2 : 공정 재튜닝 설정(dim16/bucket250k/n(2,4)), 증류 데이터 — R=10
  Q3g: OP(dim16/bucket500k/n(2,5)), 골드 — R=10
  Q3d: OP, 골드+증류 — R=10                (exp-0009 final_eval과 동일 프로토콜)
  Q4 : OP, 귀인 4팔(A/E/D/C0.9) × R=10, val best_f1 (distill.py 프로토콜, 학생 라벨 1회 생성)

실행 구조 (2026-07-18 수정): 같은 프로세스에서 학습을 반복하면 NaN이 지속 발생하는
현상이 관측되어(1차 실행: run0·run1 성공 → run2가 lr 반감 6회 전부 NaN), **런마다
ft_worker.py를 새 프로세스로 격리** 실행한다. 워커 내부의 train_with_retry(6회, lr 반감)가
전부 실패하면 프로세스 수준에서 최대 3회 재시작하며, 모든 재시작 횟수를 기록한다.
1차 실행의 부분 산출물은 runs_q2_ft.aborted-inprocess.csv로 보존(삭제 아님 — 무선별 보고).
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
    """새 프로세스에서 1런 학습. 반환: meta(final_lr, sec, proc_attempts)."""
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
        print(f"  ! {out_prefix}: 프로세스 시도 {attempt} 실패 — {last}", flush=True)
    raise RuntimeError(f"{out_prefix}: 프로세스 {MAX_PROC_ATTEMPTS}회 전부 실패 ({last})")


def run_test_block(tag, path, cfg, yv, yt):
    """val pick_threshold → test 적용 (test 주장의 원프로토콜). R런, 런마다 프로세스 격리."""
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
    """distill.py 프로토콜: metric = best_f1(val), 학생 라벨은 1회 생성."""
    arms = {"A_baseline": ours}

    e_df = pd.concat([ours, pool[["text", "orig_label"]].rename(columns={"orig_label": "label"})],
                     ignore_index=True)
    arms["E_origlabel"] = e_df

    sp_path = ART / "q4_student_val_pool_probs.npy"
    if sp_path.exists():
        sp = np.load(sp_path)
        print("[Q4] 학생 라벨 재사용 (기존 생성분)", flush=True)
    else:
        # 학생 라벨 생성도 격리 프로세스로: 풀 텍스트를 임시 split처럼 다룰 수 없으므로
        # 워커 대신 여기서 직접 1회 학습한다 (프로세스 첫 학습이므로 오염 없음).
        from common import train_with_retry  # noqa: E402
        from transfer_matrix import prob_pos  # noqa: E402
        path = write_ft(ours[["text", "label"]], DATA / "q4_student_teacher.txt")
        model, lr = train_with_retry(input=path, wordNgrams=2, epoch=25,
                                     loss="softmax", thread=1, verbose=0, **FT_OP_CFG)
        sp = prob_pos(model, pool["text"])
        np.save(sp_path, sp)
        print(f"[Q4] 학생 라벨 생성 완료 (lr={lr})", flush=True)
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

    print("→ fastText 전 런 완료", flush=True)


if __name__ == "__main__":
    main()
