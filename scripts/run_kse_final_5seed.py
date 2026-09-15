"""scripts/run_kse_final_5seed.py -- Final KSE 2026 benchmark: 1 full + 3 ablations x 5 seeds.
=============================================================================

Locked configuration (from §9.29 K-block + §9.30 P8.0 capacity ridge)
--------------------------------------------------------------------
PACER-NRDMC-lite (full):
    embed_size=320, UI_layers=3, weight_size=[64,64,64], batch_size=1024,
    lr=0.000250995, regs=4.8e-4, alpha_interest=0.50, logq_scale=0.651,
    enable_tamer=1, enable_logq=1, enable_nrdmc_lite=1,
    tamer_interest_cache=results/interest_tree_clothing_rsfp_a010.npz.

Ablation grid (each drops exactly one novel component)
------------------------------------------------------
    A0  kse_full            (baseline for comparison; identical to full config)
    A1  kse_wo_rsfp         --tamer_interest_cache=results/interest_tree_clothing.npz
                             (co-occurrence cache only; no RSFPGrowth blending;
                              proves RSFP's contribution to Mid/Tail Recall@20)
    A2  kse_wo_logq         --enable_logq=0 --logq_scale=0.0
                             (proves LogQ's contribution to Head Recall@20)
    A3  kse_wo_nrdmc        --enable_nrdmc_lite=0 --nrdmc_lite_layers=0
                             (proves NRDMC-lite's contribution to overall
                              R@20/NDCG@20)

Protocol
--------
* seeds:    [1616406634, 1640104851, 52093548, 109649638, 372270914]
* epoch:    250
* early_stopping_monitor=val_recall@20, patience=5 (evaluations),
            min_epochs=30, eval_every=5.
* reduce_lr:  factor=0.5, patience=3.
* Total = 4 configs x 5 seeds = 20 runs.

Reported per run
----------------
For each seed x config we parse from stdout at the val_recall@20 peak:
    (1) test Recall@20         BEST_Test_Recall@20
    (2) test NDCG@20           BEST_Test_NDCG@20
    (3) test Precision@20      BEST_Test_Precision@20
    (4) test Recall@20 Head    BEST_Test_Recall@20_Head
    (5) test Recall@20 Mid     BEST_Test_Recall@20_Mid
    (6) test Recall@20 Tail    BEST_Test_Recall@20_Tail

Per-config we then aggregate seed-wise mean + std (unbiased, ddof=1) into
``ranked`` inside the output JSON.  A companion Markdown table is written
to ``<output>.md`` for direct copy-paste into the paper.

Usage
-----
::

    python scripts/run_kse_final_5seed.py \\
        --seeds 1616406634 1640104851 52093548 109649638 372270914 \\
        --epoch 250 \\
        --output ./results/kse_final_5seed_clothing.json \\
        --dry_run 0

Estimated wall time
-------------------
20 runs x embed=320 with patience=5 (typical stop 60-90 epochs) x ~24s/ep
= ~28 min/run x 20 = ~9 - 10 h.  Fits overnight comfortably.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Locked baseline configuration -- shared by all four variants
# ---------------------------------------------------------------------------
BASELINE_CLI = {
    "dataset": "Clothing",
    "core": 5,
    "seed": None,
    "epoch": 250,
    "batch_size": 1024,
    "lr": 0.000250995,
    "clip_grad_norm": 1.0,
    "embed_size": 320,               # p8_e320 (KSE full config)
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
    "early_stopping_patience": 5,    # 5 evaluations = 25 epochs (eval_every=5)
    "early_stopping_min_epochs": 30, # warmup NRDMC + MACP
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
    "wandb_group": "kse_final_5seed",
}


DEFAULT_SEEDS = [1616406634, 1640104851, 52093548, 109649638, 372270914]


# ---------------------------------------------------------------------------
# Option B: 13-variant grid to isolate RSFP x LogQ x NRDMC interactions.
#   Group A (7 configs): in-batch A0 (alpha=0.10) + A1 (alpha=0)
#                        + alpha sweep {0.02, 0.05, 0.08, 0.15, 0.20}
#   Group B (4 configs): interaction / axis-toggle:
#                        2-axis (LogQ + NRDMC, no TAMER),
#                        1-axis LogQ-only, 1-axis NRDMC-only,
#                        0-axis backbone (MMHCL only).
#   Group C (2 configs): no LogQ + RSFP alpha=0.10  vs  no LogQ + base cooc.
# All 13 variants share the same locked BASELINE_CLI hyperparameters as
# run_kse_final_5seed.py; only the four overrides below vary:
#   tamer_interest_cache, enable_tamer, enable_logq/logq_scale,
#   enable_nrdmc_lite/nrdmc_lite_layers.
# ---------------------------------------------------------------------------
def build_grid_optb(rsfp_prefix: str, base_cache: str) -> list[dict]:
    def _rsfp(tag: str) -> str:
        # matches build_rsfp_interest_tree.py --rebuild_tree 1 output naming.
        return f"{rsfp_prefix}_tree_{tag}.npz"

    def _all_on(cache: str) -> dict:
        return {
            "tamer_interest_cache": cache,
            "enable_tamer": 1,
            "enable_logq": 1,
            "logq_scale": 0.651,
            "enable_nrdmc_lite": 1,
            "nrdmc_lite_layers": 2,
        }

    return [
        # ---- In-batch baselines for paired comparison ----
        {"tag": "B00_A0_alpha010", "block": "OptB_baseline",
         "label": "A0 in-batch (alpha=0.10, all 3 axes)",
         "overrides": _all_on(_rsfp("a010"))},
        {"tag": "B01_A1_alpha000", "block": "OptB_baseline",
         "label": "A1 in-batch (base cooc, alpha=0)",
         "overrides": _all_on(base_cache)},
        # ---- Group A: alpha sweep on Clothing ----
        {"tag": "B02_alpha002", "block": "OptB_alphaSweep",
         "label": "alpha=0.02 (all 3 axes)",
         "overrides": _all_on(_rsfp("a002"))},
        {"tag": "B03_alpha005", "block": "OptB_alphaSweep",
         "label": "alpha=0.05 (all 3 axes)",
         "overrides": _all_on(_rsfp("a005"))},
        {"tag": "B04_alpha008", "block": "OptB_alphaSweep",
         "label": "alpha=0.08 (all 3 axes)",
         "overrides": _all_on(_rsfp("a008"))},
        {"tag": "B05_alpha015", "block": "OptB_alphaSweep",
         "label": "alpha=0.15 (all 3 axes)",
         "overrides": _all_on(_rsfp("a015"))},
        {"tag": "B06_alpha020", "block": "OptB_alphaSweep",
         "label": "alpha=0.20 (all 3 axes)",
         "overrides": _all_on(_rsfp("a020"))},
        # ---- Group B: interaction / axis-toggle ----
        {"tag": "B07_2ax_LogQ_NRDMC", "block": "OptB_interaction",
         "label": "2-axis: LogQ + NRDMC only (no TAMER)",
         "overrides": {
             "tamer_interest_cache": base_cache,  # unused when enable_tamer=0
             "enable_tamer": 0,
             "enable_logq": 1, "logq_scale": 0.651,
             "enable_nrdmc_lite": 1, "nrdmc_lite_layers": 2}},
        {"tag": "B08_1ax_LogQonly", "block": "OptB_interaction",
         "label": "1-axis: LogQ only",
         "overrides": {
             "tamer_interest_cache": base_cache,
             "enable_tamer": 0,
             "enable_logq": 1, "logq_scale": 0.651,
             "enable_nrdmc_lite": 0, "nrdmc_lite_layers": 0}},
        {"tag": "B09_1ax_NRDMConly", "block": "OptB_interaction",
         "label": "1-axis: NRDMC only",
         "overrides": {
             "tamer_interest_cache": base_cache,
             "enable_tamer": 0,
             "enable_logq": 0, "logq_scale": 0.0,
             "enable_nrdmc_lite": 1, "nrdmc_lite_layers": 2}},
        # ---- Group C: no LogQ + RSFP (A2-like) ----
        {"tag": "B10_noLogQ_RSFP010", "block": "OptB_noLogQ",
         "label": "no LogQ + RSFP alpha=0.10 (TAMER+NRDMC on)",
         "overrides": {
             "tamer_interest_cache": _rsfp("a010"),
             "enable_tamer": 1,
             "enable_logq": 0, "logq_scale": 0.0,
             "enable_nrdmc_lite": 1, "nrdmc_lite_layers": 2}},
        {"tag": "B11_noLogQ_cooc", "block": "OptB_noLogQ",
         "label": "no LogQ + base cooc (control for B10)",
         "overrides": {
             "tamer_interest_cache": base_cache,
             "enable_tamer": 1,
             "enable_logq": 0, "logq_scale": 0.0,
             "enable_nrdmc_lite": 1, "nrdmc_lite_layers": 2}},
        # ---- Group B extra: 0-axis backbone (MMHCL only) ----
        {"tag": "B12_backbone_only", "block": "OptB_interaction",
         "label": "0-axis: MMHCL backbone (no TAMER, no LogQ, no NRDMC)",
         "overrides": {
             "tamer_interest_cache": base_cache,
             "enable_tamer": 0,
             "enable_logq": 0, "logq_scale": 0.0,
             "enable_nrdmc_lite": 0, "nrdmc_lite_layers": 0}},
    ]


# ---------------------------------------------------------------------------
# Honest ablation grid (Option C — §9.36):
#   PACER-full-new (C0) = no LogQ + base cooc + TAMER on + NRDMC-lite on
#                         (= B11 config from optb13 grid)
#   A1_noNRDMC     (C1) = same as C0 but with NRDMC-lite off (lambda_cl=0)
# This 2-cell ablation gives the single-axis architectural necessity test
# after the 5-seed rescue verdict (C) BOTH FAIL on Data and Loss axes.
# ---------------------------------------------------------------------------
def build_grid_honest(base_cache: str) -> list[dict]:
    return [
        {
            "tag": "C0_pacer_full_new",
            "block": "KSE_honest_full",
            "label": "PACER-full-new (no LogQ + base cooc + NRDMC-lite on)",
            "overrides": {
                "tamer_interest_cache": base_cache,
                "enable_tamer": 1,
                "enable_logq": 0,
                "logq_scale": 0.0,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "C1_A1_noNRDMC",
            "block": "KSE_honest_ablation",
            "label": "A1 -- w/o NRDMC-lite (no LogQ + base cooc + NRDMC off)",
            "overrides": {
                "tamer_interest_cache": base_cache,
                "enable_tamer": 1,
                "enable_logq": 0,
                "logq_scale": 0.0,
                "enable_nrdmc_lite": 0,
                "nrdmc_lite_layers": 0,
            },
        },
    ]


# Attach a distinct wandb_group so Option B runs don't collide with the
# canonical kse_final_5seed grid in the dashboard.
def _stamp_wandb_group(grid: list[dict], group: str) -> list[dict]:
    for v in grid:
        v["overrides"].setdefault("wandb_group", group)
    return grid


# ---------------------------------------------------------------------------
# 4-variant grid: 1 full + 3 ablations
# ---------------------------------------------------------------------------
def build_grid(rsfp_cache: str, base_cache: str) -> list[dict]:
    return [
        {
            "tag": "A0_kse_full",
            "block": "KSE_full",
            "label": "PACER-NRDMC-lite (full)",
            "overrides": {
                "tamer_interest_cache": rsfp_cache,
                "enable_tamer": 1,
                "enable_logq": 1,
                "logq_scale": 0.651,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "A1_kse_wo_rsfp",
            "block": "KSE_ablation",
            "label": "w/o RSFPGrowth (base co-occurrence cache only)",
            "overrides": {
                "tamer_interest_cache": base_cache,   # no RSFP blending
                "enable_tamer": 1,
                "enable_logq": 1,
                "logq_scale": 0.651,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "A2_kse_wo_logq",
            "block": "KSE_ablation",
            "label": "w/o LogQ (no popularity de-bias)",
            "overrides": {
                "tamer_interest_cache": rsfp_cache,
                "enable_tamer": 1,
                "enable_logq": 0,
                "logq_scale": 0.0,
                "enable_nrdmc_lite": 1,
                "nrdmc_lite_layers": 2,
            },
        },
        {
            "tag": "A3_kse_wo_nrdmc",
            "block": "KSE_ablation",
            "label": "w/o NRDMC-lite (no residual denoising)",
            "overrides": {
                "tamer_interest_cache": rsfp_cache,
                "enable_tamer": 1,
                "enable_logq": 1,
                "logq_scale": 0.651,
                "enable_nrdmc_lite": 0,
                "nrdmc_lite_layers": 0,
            },
        },
    ]


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------
def build_cli(python_exe: str, main_py: Path, variant: dict,
              seed: int, epoch: int) -> list[str]:
    cfg = dict(BASELINE_CLI)
    cfg["seed"] = seed
    cfg["epoch"] = epoch
    cfg.update(variant["overrides"])
    cfg["wandb_run_name"] = f"{variant['tag']}_seed{seed}"
    cfg["wandb_tags"] = ",".join([
        "kse", "kse_final_5seed", variant["block"], variant["tag"],
        f"embed{cfg['embed_size']}",
        f"logq{cfg['logq_scale']}",
        f"nrdmc{cfg['enable_nrdmc_lite']}",
        Path(cfg["tamer_interest_cache"]).stem,
        "r20_monitor", "patience5_min30",
    ])
    cmd = [python_exe, str(main_py)]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = int(v)
        cmd += [f"--{k}", str(v)]
    return cmd


# ---------------------------------------------------------------------------
# Stdout parser -- extract the 6 paper metrics from BEST_Test_* lines
# ---------------------------------------------------------------------------
def _parse_run_output(out: str) -> dict:
    """Parse the 6 KSE paper metrics + best_epoch from stdout."""
    result: dict = {}
    scalar_patterns = [
        ("best_test_recall20",    r"BEST_Test_Recall@20:\s*([\d.eE+\-]+)"),
        ("best_test_ndcg20",      r"BEST_Test_NDCG@20:\s*([\d.eE+\-]+)"),
        ("best_test_precision20", r"BEST_Test_Precision@20:\s*([\d.eE+\-]+)"),
        ("best_val_recall20",     r"BEST_Val_Recall@20:\s*([\d.eE+\-]+)"),
        ("best_val_ndcg20",       r"BEST_Val_NDCG@20:\s*([\d.eE+\-]+)"),
        ("best_epoch",            r"BEST_Val_Recall_Peak_Epoch:\s*(\d+)"),
    ]
    for key, pat in scalar_patterns:
        m = re.search(pat, out)
        if m:
            v = m.group(1)
            result[key] = int(v) if key == "best_epoch" else float(v)

    tercile_patterns = [
        ("best_test_head_recall20", r"BEST_Test_Recall@20_Head=([\d.eE+\-]+)"),
        ("best_test_mid_recall20",  r"BEST_Test_Recall@20_Mid=([\d.eE+\-]+)"),
        ("best_test_tail_recall20", r"BEST_Test_Recall@20_Tail=([\d.eE+\-]+)"),
        # val-side tercile (fallback if test not printed)
        ("best_val_head_recall20",  r"BEST_Recall@20_Head=([\d.eE+\-]+)"),
        ("best_val_mid_recall20",   r"BEST_Recall@20_Mid=([\d.eE+\-]+)"),
        ("best_val_tail_recall20",  r"BEST_Recall@20_Tail=([\d.eE+\-]+)"),
    ]
    for key, pat in tercile_patterns:
        m = re.search(pat, out)
        if m:
            result[key] = float(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
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
    print(f"\n[kse] running (tail): "
          f"{' '.join(shlex.quote(c) for c in cmd[-30:])}")
    print(f"[kse] log: {log_path}")
    with log_path.open("wb") as fh:
        cwd = str(Path(cmd[1]).resolve().parent) if len(cmd) > 1 else None
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
        )
        exit_code = proc.wait()
    wall = time.time() - t0
    out = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[kse] exit={exit_code}  wall={wall/60.0:.1f} min")
    return exit_code, out, wall


# ---------------------------------------------------------------------------
# Aggregation helpers -- seed-wise mean +/- std for the paper table
# ---------------------------------------------------------------------------
_METRIC_KEYS = [
    ("best_test_recall20",     "R@20"),
    ("best_test_ndcg20",       "NDCG@20"),
    ("best_test_precision20",  "P@20"),
    ("best_test_head_recall20", "R@20_Head"),
    ("best_test_mid_recall20",  "R@20_Mid"),
    ("best_test_tail_recall20", "R@20_Tail"),
]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)  # ddof=1
    return m, math.sqrt(var)


def _aggregate(rows: list[dict], grid: list[dict]) -> list[dict]:
    """One record per variant: seed-wise mean & std for each metric."""
    per_tag: dict = {}
    for r in rows:
        per_tag.setdefault(r["tag"], []).append(r)

    tag_to_variant = {v["tag"]: v for v in grid}
    aggregated = []
    for tag, rs in per_tag.items():
        v = tag_to_variant.get(tag, {})
        rec = {
            "tag": tag,
            "block": rs[0].get("block"),
            "label": v.get("label", tag),
            "n_seeds": len(rs),
        }
        for key, _ in _METRIC_KEYS:
            vals = [r[key] for r in rs if key in r]
            m, s = _mean_std(vals)
            rec[f"{key}_mean"] = m
            rec[f"{key}_std"]  = s
            rec[f"{key}_n"]    = len(vals)
        aggregated.append(rec)

    # Rank by mean test R@20 desc.
    aggregated.sort(key=lambda r: (r.get("best_test_recall20_mean") or 0),
                    reverse=True)
    return aggregated


def _write_markdown(agg: list[dict], out_path: Path) -> None:
    header = ("| Variant | " +
              " | ".join(label for _, label in _METRIC_KEYS) +
              " |")
    sep = "|" + "|".join(["---"] * (1 + len(_METRIC_KEYS))) + "|"
    lines = [header, sep]
    for r in agg:
        cells = [f"{r['label']}"]
        for key, _ in _METRIC_KEYS:
            m = r.get(f"{key}_mean")
            s = r.get(f"{key}_std")
            if m is None or (isinstance(m, float) and math.isnan(m)):
                cells.append("—")
            else:
                cells.append(f"{m*100:.3f} ± {s*100:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epoch", type=int, default=250)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path, default=Path("main_tercile.py"))
    ap.add_argument("--output", type=Path,
                    default=Path("./results/kse_final_5seed_clothing.json"))
    ap.add_argument("--log_dir", type=Path,
                    default=Path("./results/_kse_final_5seed_logs"))
    ap.add_argument("--rsfp_cache", type=Path,
                    default=Path("results/interest_tree_clothing_rsfp_a010.npz"))
    ap.add_argument("--rsfp_prefix", type=Path,
                    default=Path("results/interest_tree_clothing_rsfp"),
                    help="Prefix for Option B alpha-sweep caches; the driver "
                         "appends _tree_a{NNN}.npz per alpha.")
    ap.add_argument("--base_cache", type=Path,
                    default=Path("results/interest_tree_clothing.npz"))
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--only_tags", type=str, nargs="*", default=None)
    ap.add_argument("--grid", type=str, default="kse4",
                    choices=["kse4", "optb13", "honest"],
                    help="kse4 = original 1-full + 3-ablation grid (5 seeds x "
                         "250 epochs). optb13 = Option B 13-variant grid "
                         "(alpha sweep + interaction + no-LogQ tests). "
                         "honest = 2-cell Option C ablation (C0 PACER-full-new "
                         "= no LogQ + base cooc + NRDMC on; C1 = C0 without "
                         "NRDMC-lite). Used in notebook \u00a79.36.")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    main_py = _resolve_main(args.main)
    print(f"[kse] main_py: {main_py}")
    print(f"[kse] seeds:   {args.seeds}")

    if args.grid == "optb13":
        grid = _stamp_wandb_group(
            build_grid_optb(str(args.rsfp_prefix), str(args.base_cache)),
            group="kse_optb_2seed_clothing",
        )
    elif args.grid == "honest":
        grid = _stamp_wandb_group(
            build_grid_honest(str(args.base_cache)),
            group="kse_honest_ablation_clothing",
        )
    else:
        grid = build_grid(str(args.rsfp_cache), str(args.base_cache))
    if args.only_tags:
        grid = [v for v in grid if v["tag"] in set(args.only_tags)]

    # Sanity: verify prerequisite caches exist (skip in dry_run).
    if not args.dry_run:
        missing = []
        for v in grid:
            p = Path(v["overrides"]["tamer_interest_cache"])
            if not p.is_file():
                missing.append(str(p))
        if missing:
            print("[kse] WARNING -- missing interest caches:")
            for m in missing:
                print(f"          {m}")
            print("[kse] Build them via scripts/build_rsfp_interest_tree.py first.")

    total_runs = len(grid) * len(args.seeds)
    print(f"[kse] total runs: {total_runs} "
          f"({len(grid)} variants x {len(args.seeds)} seeds x "
          f"epoch=<= {args.epoch}, patience=5 evals, min_epochs=30)")

    rows: list[dict] = []
    run_idx = 0

    for variant in grid:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'='*72}\n"
                  f"[kse] {run_idx}/{total_runs}  "
                  f"tag={variant['tag']}  seed={seed}\n{'='*72}")
            cmd = build_cli(args.python, main_py, variant, seed, args.epoch)
            log_path = args.log_dir / f"{variant['tag']}_seed{seed}.log"
            exit_code, out, wall = _run_one(cmd, log_path, bool(args.dry_run))
            parsed = _parse_run_output(out) if not args.dry_run else {}
            row = {
                "tag": variant["tag"],
                "block": variant["block"],
                "label": variant["label"],
                "seed": seed,
                "epoch_cap": args.epoch,
                "wall_min": wall / 60.0,
                "exit": exit_code,
                **variant["overrides"],
                **parsed,
            }
            rows.append(row)
            # Incremental JSON save after every run.
            aggregated = _aggregate(rows, grid) if not args.dry_run else []
            with args.output.open("w", encoding="utf-8") as fh:
                json.dump({
                    "rows": rows,
                    "aggregated": aggregated,
                    "runs_completed": run_idx,
                    "total_runs": total_runs,
                    "seeds": args.seeds,
                    "epoch_cap": args.epoch,
                    "protocol": {
                        "early_stopping_monitor": "val_recall@20",
                        "early_stopping_patience_evals": 5,
                        "early_stopping_min_epochs": 30,
                        "eval_every": 5,
                        "reduce_lr_factor": 0.5,
                        "reduce_lr_patience": 3,
                    },
                }, fh, indent=2)

    if args.dry_run:
        print("[dry_run] summary skipped.")
        return

    aggregated = _aggregate(rows, grid)
    _write_markdown(aggregated, args.output.with_suffix(".md"))

    # Persist final JSON with aggregation.
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({
            "rows": rows,
            "aggregated": aggregated,
            "runs_completed": len(rows),
            "total_runs": total_runs,
            "seeds": args.seeds,
            "epoch_cap": args.epoch,
            "protocol": {
                "early_stopping_monitor": "val_recall@20",
                "early_stopping_patience_evals": 5,
                "early_stopping_min_epochs": 30,
                "eval_every": 5,
                "reduce_lr_factor": 0.5,
                "reduce_lr_patience": 3,
            },
        }, fh, indent=2)

    print("\n=== KSE final 5-seed benchmark (seed-wise mean * 100, +/- std) ===")
    print(f"{'Variant':<48} " + " ".join(f"{lbl:>13}" for _, lbl in _METRIC_KEYS))
    for r in aggregated:
        cells = []
        for key, _ in _METRIC_KEYS:
            m = r.get(f"{key}_mean")
            s = r.get(f"{key}_std")
            if m is None or (isinstance(m, float) and math.isnan(m)):
                cells.append(f"{'—':>13}")
            else:
                cells.append(f"{m*100:>6.3f}+/-{s*100:.3f}")
        print(f"{r['label'][:47]:<48} " + " ".join(cells))

    if not aggregated:
        return
    # Choose the baseline for delta report: kse4 -> A0_kse_full; optb13 -> B00_A0_alpha010.
    if args.grid == "optb13":
        baseline_tag = "B00_A0_alpha010"
    elif args.grid == "honest":
        baseline_tag = "C0_pacer_full_new"
    else:
        baseline_tag = "A0_kse_full"
    full = next((r for r in aggregated if r["tag"] == baseline_tag), None)
    if not full:
        return
    print(f"\n=== Ablation deltas (percentage points on * 100 scale, vs {baseline_tag}) ===")
    full_r20 = (full.get("best_test_recall20_mean") or 0) * 100
    full_nd  = (full.get("best_test_ndcg20_mean")   or 0) * 100
    full_h   = (full.get("best_test_head_recall20_mean") or 0) * 100
    full_m   = (full.get("best_test_mid_recall20_mean")  or 0) * 100
    full_t   = (full.get("best_test_tail_recall20_mean") or 0) * 100
    for r in aggregated:
        if r["tag"] == baseline_tag:
            continue
        d_r20 = (r.get("best_test_recall20_mean") or 0) * 100 - full_r20
        d_nd  = (r.get("best_test_ndcg20_mean")   or 0) * 100 - full_nd
        d_h   = (r.get("best_test_head_recall20_mean") or 0) * 100 - full_h
        d_m   = (r.get("best_test_mid_recall20_mean")  or 0) * 100 - full_m
        d_t   = (r.get("best_test_tail_recall20_mean") or 0) * 100 - full_t
        print(f"  {r['tag']:<20}  dR@20={d_r20:+.3f}  dNDCG={d_nd:+.3f}  "
              f"dHead={d_h:+.3f}  dMid={d_m:+.3f}  dTail={d_t:+.3f}")

    print(f"\n=== Paper-ready Markdown table written to {args.output.with_suffix('.md')} ===")


if __name__ == "__main__":
    main()
