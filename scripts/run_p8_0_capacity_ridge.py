"""scripts/run_p8_0_capacity_ridge.py -- P8.0 Capacity ridge extension.
=============================================================================

Rationale
---------
P7.1 K-block at 1 seed showed capacity (width) dominates every other lever:

    P6.4 (5-seed mean) R@20 = 0.09566, NDCG@20 = 0.04382
    k1_e192            R@20 = 0.10096   NDCG@20 = 0.04606
    k2_e256            R@20 = 0.10234   NDCG@20 = 0.04686   (+7.0%/+7.0%)
    k3_L4  (deeper)    R@20 = 0.09515   NDCG@20 = 0.04371   (regression)

Width (embed_size) helped; depth (UI_layers=4) hurt.  Head-decay observed
in the WandB CSV (ep39 -> ep84: 0.15299 -> 0.14339) suggests the model
still has room to trade a bit of capacity into either (a) more width,
(b) wider CF projection (weight_size), or (c) a heavier regs to hold
Head longer.  P8.0 probes the width plateau; P8.1 probes head-preservers.

Grid design (5 variants x 1 seed x 100 epoch  ~=  3 - 4 h wall)
---------------------------------------------------------------

Anchor:  regs=4.8e-4, logq=0.651, UI_layers=3, enable_simgcl=0,
         rsfp=off (base cache), lr=0.000250995, batch_size=1024.

    p8_e320           embed_size=320  UI_layers=3  weight_size=[64,64,64]
    p8_e384           embed_size=384  UI_layers=3  weight_size=[64,64,64]
    p8_e512           embed_size=512  UI_layers=3  weight_size=[64,64,64]
    p8_e256_ws128     embed_size=256  UI_layers=3  weight_size=[128,128,128]
                        -- widens CF projection instead of the base emb;
                           tests if bottleneck sits at CF weight matrices
    p8_e256_L4        embed_size=256  UI_layers=4  weight_size=[64,64,64]
                        -- re-test depth WITH width; kills or confirms
                           the "depth hurts at any width" hypothesis

Usage
-----
::

    python scripts/run_p8_0_capacity_ridge.py \\
        --seeds 23946202 \\
        --epoch 100 \\
        --output ./results/p8_0_capacity_ridge_clothing.json \\
        --dry_run 0

Note
----
p8_e512 may approach the 32 GB VRAM budget of RTX 5090 when combined
with TAMER + MACP + AMP.  If OOM, retry with `--only_tags` excluding it
and lower `simgcl_batch_size_*` inside the driver (SimGCL is off in this
grid, so those flags are inert).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


BASELINE_CLI = {
    "dataset": "Clothing",
    "core": 5,
    "seed": None,
    "epoch": 100,
    "batch_size": 1024,
    "lr": 0.000250995,
    "clip_grad_norm": 1.0,
    "embed_size": 128,
    "weight_size": "[64,64,64]",
    "topk": 10,
    "cf_model": "LightGCN",
    "norm_type": "sym",
    "UI_layers": 3,
    "User_layers": 2,
    "Item_layers": 2,
    "user_loss_ratio": 0.03,
    "item_loss_ratio": 0.07,
    "temperature": 0.3,
    "learnable_tau": 0,
    "Ks": "[10,20]",
    "test_flag": "part",
    "use_gpu_eval": 1,
    "eval_every": 5,
    "eval_last_epochs": 60,
    "early_stopping_patience": 30,
    "early_stopping_min_epochs": 0,
    "early_stopping_min_delta": 1e-4,
    "early_stopping_mode": "max",
    "early_stopping_restore_best": 1,
    "early_stopping_monitor": "val_recall@20",
    "use_reduce_lr": 1,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 3,
    "reduce_lr_min": 1e-6,
    "regs": 4.8e-4,
    "damps_apc": 0,
    "damps_avrf": 0,
    "damps_imcf": 1,
    "damps_permutation_fft": 0,
    "damps_soft_routing": 1,
    "damps_momentum": 1,
    "damps_data_driven_prior": 1,
    "damps_num_categories": 10,
    "damps_warmup_epochs": 10,
    "enable_logq": 1,
    "logq_mode": "laplace",
    "logq_beta": 1.0,
    "logq_clip": 5.0,
    "logq_scale": 0.651,
    "enable_simgcl": 0,
    "simgcl_eps": 0.329,
    "lambda_view": 0.1,
    "simgcl_batch_size_user": 4096,
    "simgcl_batch_size_item": 4096,
    "branchA_view_every_k": 2,
    "branchA_bcl_batchn": 1,
    "branchA_view_bsz": 2048,
    "branchA_bcl_bsz": 2048,
    "enable_nrdmc_lite": 1,
    "nrdmc_lite_layers": 2,
    "enable_ptv": 0,
    "n_prototypes": 0,
    "lambda_ptv": 0.0,
    "enable_align": 0,
    "lambda_align": 0.0,
    "align_temperature": 0.2,
    "use_macp": 1,
    "macp_mode": "replace_pca",
    "macp_alpha_p": 0.0,
    "macp_alpha_z": 0.0,
    "macp_image_mode": "replace_pca",
    "macp_image_alpha_p": 0.0,
    "macp_image_alpha_z": 0.0,
    "macp_verbose": 1,
    "enable_tamer": 1,
    "alpha_interest": 0.50,
    "pop_inverse_eta": 0.0,
    "rebuild_R": 5,
    "faiss_threshold": 60000,
    "knn_chunk_size": 4096,
    "faiss_use_gpu": 1,
    "knn_efsearch": 64,
    "use_amp": 1,
    "use_torch_compile": 1,
    "torch_compile_mode": "default",
    "torch_compile_dynamic": 0,
    "use_gpu_sample": 1,
    "use_cuda_graph": 0,
    "asc_gate_mode": "raw",
    "asc_warmup_epochs": 0,
    "asc_reg_l2": 0.0,
    "asc_reg_target": 0.3,
    "ablation_target": "",
    "use_wandb": 1,
    "wandb_project": "damps-mmhcl-clothing",
    "wandb_entity": "baitapck51cc-uet",
    "wandb_group": "p8_0_capacity_ridge",
}


def build_grid(base_cache: str) -> list[dict]:
    grid = [
        {"tag": "p8_e320", "block": "P8_0_width",
         "overrides": {"embed_size": 320, "UI_layers": 3,
                       "weight_size": "[64,64,64]",
                       "tamer_interest_cache": base_cache}},
        {"tag": "p8_e384", "block": "P8_0_width",
         "overrides": {"embed_size": 384, "UI_layers": 3,
                       "weight_size": "[64,64,64]",
                       "tamer_interest_cache": base_cache}},
        {"tag": "p8_e512", "block": "P8_0_width",
         "overrides": {"embed_size": 512, "UI_layers": 3,
                       "weight_size": "[64,64,64]",
                       "tamer_interest_cache": base_cache}},
        {"tag": "p8_e256_ws128", "block": "P8_0_cf_width",
         "overrides": {"embed_size": 256, "UI_layers": 3,
                       "weight_size": "[128,128,128]",
                       "tamer_interest_cache": base_cache}},
        {"tag": "p8_e256_L4", "block": "P8_0_depth_at_width",
         "overrides": {"embed_size": 256, "UI_layers": 4,
                       "weight_size": "[64,64,64]",
                       "tamer_interest_cache": base_cache}},
    ]
    return grid


def build_cli(python_exe: str, main_py: Path, variant: dict,
              seed: int, epoch: int) -> list[str]:
    cfg = dict(BASELINE_CLI)
    cfg["seed"] = seed
    cfg["epoch"] = epoch
    cfg.update(variant["overrides"])
    cfg["wandb_run_name"] = f"{variant['tag']}_seed{seed}"
    cfg["wandb_tags"] = ",".join([
        "p8", "p8_0_capacity_ridge", variant["block"], variant["tag"],
        f"embed{cfg['embed_size']}",
        f"L{cfg['UI_layers']}",
        f"ws{cfg['weight_size'].replace('[','').replace(']','').replace(',','_')}",
        "nrdmc_lite", "tamer", "alpha050", "r20_monitor",
    ])
    cmd = [python_exe, str(main_py)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = int(v)
        cmd += [f"--{k}", str(v)]
    return cmd


def _parse_run_output(out: str) -> dict:
    result: dict = {}
    for key, pat in [
        ("best_test_recall20", r"BEST_Test_Recall@20:\s*([\d.]+)"),
        ("best_test_ndcg20",   r"BEST_Test_NDCG@20:\s*([\d.]+)"),
        ("best_val_recall20",  r"BEST_Val_Recall@20:\s*([\d.]+)"),
        ("best_val_ndcg20",    r"BEST_Val_NDCG@20:\s*([\d.]+)"),
        ("best_epoch",         r"BEST_Val_Recall_Peak_Epoch:\s*(\d+)"),
    ]:
        m = re.search(pat, out)
        if m:
            v = m.group(1)
            result[key] = int(v) if key == "best_epoch" else float(v)
    for name, pat in [
        ("best_val_head_recall20", r"BEST_Recall@20_Head=([\d.]+)"),
        ("best_val_mid_recall20",  r"BEST_Recall@20_Mid=([\d.]+)"),
        ("best_val_tail_recall20", r"BEST_Recall@20_Tail=([\d.]+)"),
        ("best_test_head_recall20", r"BEST_Test_Recall@20_Head=([\d.]+)"),
        ("best_test_mid_recall20",  r"BEST_Test_Recall@20_Mid=([\d.]+)"),
        ("best_test_tail_recall20", r"BEST_Test_Recall@20_Tail=([\d.]+)"),
    ]:
        m = re.search(pat, out)
        if m:
            result[name] = float(m.group(1))
    return result


def _resolve_main(main_arg: Path) -> Path:
    if main_arg.is_absolute() and main_arg.is_file():
        return main_arg
    cwd = Path.cwd()
    here = Path(__file__).resolve()
    repo = here.parent.parent if here.parent.name == "scripts" else cwd
    candidates = [
        cwd / main_arg,
        cwd / "main_tercile.py",
        cwd / "src" / "main_tercile.py",
        repo / "src" / "main_tercile.py",
        cwd / "MMHCL_DAMPS_Project" / "main_tercile.py",
        here.parent.parent / "main_tercile.py",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"Could not locate main_tercile.py. Tried: {main_arg}")


def _run_one(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[int, str, float]:
    if dry_run:
        print("[dry_run] " + " ".join(shlex.quote(c) for c in cmd))
        return 0, "", 0.0
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[grid] running (tail): "
          f"{' '.join(shlex.quote(c) for c in cmd[-24:])}")
    print(f"[grid] log: {log_path}")
    with log_path.open("wb") as fh:
        cwd = str(Path(cmd[1]).resolve().parent) if len(cmd) > 1 else None
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
        )
        exit_code = proc.wait()
    wall = time.time() - t0
    out = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[grid] exit={exit_code}  wall={wall/60.0:.1f} min")
    return exit_code, out, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[23946202])
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path, default=Path("main_tercile.py"))
    ap.add_argument("--output", type=Path,
                    default=Path("./results/p8_0_capacity_ridge_clothing.json"))
    ap.add_argument("--log_dir", type=Path,
                    default=Path("./results/_p8_0_capacity_ridge_logs"))
    ap.add_argument("--base_cache", type=Path,
                    default=Path("results/interest_tree_clothing.npz"))
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--only_tags", type=str, nargs="*", default=None)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    main_py = _resolve_main(args.main)
    print(f"[grid] main_py: {main_py}")

    grid = build_grid(str(args.base_cache))
    if args.only_tags:
        grid = [v for v in grid if v["tag"] in set(args.only_tags)]

    total_runs = len(grid) * len(args.seeds)
    print(f"[grid] total runs: {total_runs} "
          f"({len(grid)} variants x {len(args.seeds)} seeds x "
          f"{args.epoch} epoch)")

    rows: list[dict] = []
    run_idx = 0

    for variant in grid:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'='*72}\n"
                  f"[grid] {run_idx}/{total_runs}  "
                  f"tag={variant['tag']}  seed={seed}\n{'='*72}")
            cmd = build_cli(args.python, main_py, variant, seed, args.epoch)
            log_path = args.log_dir / f"{variant['tag']}_seed{seed}.log"
            exit_code, out, wall = _run_one(cmd, log_path, bool(args.dry_run))
            parsed = _parse_run_output(out) if not args.dry_run else {}
            row = {
                "tag": variant["tag"],
                "block": variant["block"],
                "seed": seed,
                "epoch_cap": args.epoch,
                "wall_min": wall / 60.0,
                "exit": exit_code,
                **variant["overrides"],
                **parsed,
            }
            rows.append(row)
            with args.output.open("w", encoding="utf-8") as fh:
                json.dump({"rows": rows, "runs_completed": run_idx,
                           "total_runs": total_runs}, fh, indent=2)

    if args.dry_run:
        print("[dry_run] summary skipped.")
        return

    p6_4_ref = {"r20": 0.09566, "ndcg20": 0.04382}
    k2_ref   = {"r20": 0.10234, "ndcg20": 0.04686}  # P7.1 K-block winner

    per_tag: dict = {}
    for r in rows:
        per_tag.setdefault(r["tag"], []).append(r)

    def _mean(rs, k):
        vs = [r[k] for r in rs if k in r]
        return sum(vs) / len(vs) if vs else None

    ranked = []
    for tag, rs in per_tag.items():
        r20 = _mean(rs, "best_test_recall20")
        nd  = _mean(rs, "best_test_ndcg20")
        h = (_mean(rs, "best_test_head_recall20")
             or _mean(rs, "best_val_head_recall20"))
        m = (_mean(rs, "best_test_mid_recall20")
             or _mean(rs, "best_val_mid_recall20"))
        t = (_mean(rs, "best_test_tail_recall20")
             or _mean(rs, "best_val_tail_recall20"))
        ranked.append({
            "tag": tag,
            "block": rs[0]["block"],
            "n_seeds": len(rs),
            "recall20_mean": r20,
            "ndcg20_mean": nd,
            "head_mean": h, "mid_mean": m, "tail_mean": t,
            "delta_vs_p6_4_r20": (r20 - p6_4_ref["r20"]) if r20 else None,
            "delta_vs_k2_r20":   (r20 - k2_ref["r20"])   if r20 else None,
            "delta_vs_p6_4_ndcg": (nd - p6_4_ref["ndcg20"]) if nd else None,
            "delta_vs_k2_ndcg":   (nd - k2_ref["ndcg20"])   if nd else None,
        })
    ranked.sort(key=lambda r: (r["recall20_mean"] or 0), reverse=True)

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows, "ranked": ranked,
            "p6_4_reference": p6_4_ref,
            "k2_e256_p71_reference": k2_ref,
            "runs_completed": len(rows), "total_runs": total_runs,
        }, fh, indent=2)

    print("\n=== P8.0 Capacity ridge (ranked by R@20) ===")
    print(f"{'tag':<18} {'block':<22} {'R@20':>8} {'dk2':>8} "
          f"{'NDCG':>8} {'dk2N':>8} {'Head':>8} {'Mid':>8} {'Tail':>8}")
    for r in ranked:
        r20 = r["recall20_mean"] or float("nan")
        dk2 = r["delta_vs_k2_r20"] or 0.0
        nd = r["ndcg20_mean"] or float("nan")
        dk2n = r["delta_vs_k2_ndcg"] or 0.0
        h = r["head_mean"] or float("nan")
        m = r["mid_mean"] or float("nan")
        t = r["tail_mean"] or float("nan")
        print(f"{r['tag']:<18} {r['block']:<22} {r20:>8.5f} {dk2:>+8.5f} "
              f"{nd:>8.5f} {dk2n:>+8.5f} {h:>8.5f} {m:>8.5f} {t:>8.5f}")

    if not ranked:
        return
    w = ranked[0]
    print(f"\n=== VERDICT ===")
    print(f"  Winner: '{w['tag']}' "
          f"R@20={w['recall20_mean']:.5f} "
          f"(vs k2_e256 = {w['delta_vs_k2_r20']:+.5f}, "
          f"vs P6.4 = {w['delta_vs_p6_4_r20']:+.5f})")
    d = w['delta_vs_k2_r20'] or 0.0
    if d > 0.0005:
        print("  Signal above k2_e256. Adopt config + replicate 2 seeds.")
    elif d > -0.0005:
        print("  Within noise band of k2_e256. Keep k2 config.")
    else:
        print("  Regression vs k2_e256. Capacity plateau at embed=256.")


if __name__ == "__main__":
    main()
