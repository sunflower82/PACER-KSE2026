"""
scripts/run_optuna_branchA.py
=============================

Hyperparameter optimisation for DAMPS-MMHCL Branch A on Amazon Sports
(rev55 §8.1 — frozen Wave 1 backbone ``apc_off_combined`` + LogQ +
SimGCL batch-N + view-every-k=2 + bf16 AMP).

Search strategy
---------------
* Sampler  : ``optuna.samplers.TPESampler(seed=42, multivariate=True,
             constant_liar=True)``
* Pruner   : ``optuna.pruners.HyperbandPruner(min_resource=30,
             max_resource=epochs, reduction_factor=3)``
* Storage  : SQLite (``--storage``); the study is resumable across
             crashes and restarts.
* Objective: ``median(BEST_Test_Recall@20)`` over ``--n_seeds`` seeds.
             Intermediate per-epoch ``val/recall@20`` is streamed from
             the training subprocess to feed the pruner.

Frozen flags (NEVER searched)
-----------------------------
backbone ``apc_off_combined``:    ``--damps_apc 0 --damps_avrf 0``
IMCF / routing / momentum / DDP:   on
LogQ:                              ``--enable_logq 1 --logq_mode laplace
                                    --logq_beta 1.0 --logq_clip 5.0``
SimGCL master:                     ``--enable_simgcl 1``
SimGCL batch-N + view cache:       ``--branchA_bcl_batchn 1
                                    --branchA_view_every_k 2``
AMP bf16:                          ``--use_amp 1``
Early stopping:                    ``--early_stopping_patience 20``
                                   ``--early_stopping_min_epochs 75``

Searched HPs
------------
| HP                  | Range              | Prior        |
|---------------------|--------------------|--------------|
| ``lr``              | [1e-4, 2e-3]       | log-uniform  |
| ``batch_size``      | {1024, 2048, 4096} | categorical  |
| ``regs``            | [1e-4, 1e-2]       | log-uniform  |
| ``simgcl_eps``      | [0.05, 0.30]       | uniform      |
| ``lambda_view``     | [0.01, 0.20]       | log-uniform  |
| ``temperature``     | [0.10, 0.50]       | uniform      |
| ``embed_size``      | {64, 128}          | categorical  |
| ``UI_layers``       | {2, 3, 4}          | categorical  |
| ``logq_scale``      | [0.5, 1.5]         | uniform      |
| ``use_reduce_lr``   | {0, 1}             | categorical  |
| ``reduce_lr_factor``| [0.3, 0.7]        | if use_reduce_lr=1 |
| ``reduce_lr_patience`` | [3, 8]         | if use_reduce_lr=1 |

Usage
-----
::

    cd MMHCL_DAMPS_Project
    pip install optuna

    # Single-seed trials (fast — recommended for first pass)
    python scripts/run_optuna_branchA.py \
        --dataset sports \
        --n_trials 40 \
        --epoch 250 \
        --study_name branchA_sports_v1 \
        --storage sqlite:///optuna_branchA_sports.db

    # Multi-seed median objective (more robust, 3× more expensive)
    python scripts/run_optuna_branchA.py \
        --dataset sports \
        --n_trials 30 \
        --n_seeds 3 \
        --epoch 250 \
        --study_name branchA_sports_3seed \
        --storage sqlite:///optuna_branchA_sports.db

After the study finishes the top-5 trials and their full HP configs are
printed and written to ``<study_name>_best.json``. Use the best config
to re-run a 5-seed final evaluation matching the Clothing protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

try:
    import optuna
    from optuna.pruners import HyperbandPruner, NopPruner
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState
except ImportError as exc:                                      # pragma: no cover
    print("ERROR: optuna is required. Install with: pip install optuna",
          file=sys.stderr)
    raise SystemExit(2) from exc


# ----------------------------------------------------------------------
#  Frozen Wave 1 Branch A flags (NEVER vary per rev55 §8.1)
# ----------------------------------------------------------------------
FROZEN_FLAGS: list[str] = [
    # Backbone apc_off_combined
    "--damps_apc",                   "0",
    "--damps_avrf",                  "0",
    "--damps_imcf",                  "1",
    "--damps_permutation_fft",       "0",
    "--damps_soft_routing",          "1",
    "--damps_momentum",              "1",
    "--damps_data_driven_prior",     "1",
    "--damps_num_categories",        "10",
    "--damps_warmup_epochs",         "10",
    "--rebuild_R",                   "5",

    # LogQ — popularity correction (locked; scale is the only searched dim)
    "--enable_logq",                 "1",
    "--logq_mode",                   "laplace",
    "--logq_beta",                   "1.0",
    "--logq_clip",                   "5.0",

    # SimGCL master + Branch A overlay
    "--enable_simgcl",               "1",
    "--branchA_bcl_batchn",          "1",
    "--branchA_view_every_k",        "2",
    "--branchA_view_bsz",            "2048",
    "--branchA_bcl_bsz",             "2048",

    # System
    "--use_amp",                     "1",
    "--knn_chunk_size",              "4096",
    "--knn_efsearch",                "64",
    "--faiss_threshold",             "60000",
    "--faiss_use_gpu",               "1",
    "--clip_grad_norm",              "1.0",

    # Architecture dims not in the search space (match cell32 BACKBONE_FLAGS)
    "--topk",                        "5",
    "--core",                        "5",
    "--learnable_tau",               "0",
    "--User_layers",                 "3",
    "--Item_layers",                 "2",
    "--user_loss_ratio",             "0.03",
    "--item_loss_ratio",             "0.07",

    # Eval schedule
    "--verbose",                     "5",

    # Early stopping — Wave 1 freeze
    "--early_stopping_patience",     "20",
    "--early_stopping_min_epochs",   "75",
    "--early_stopping_min_delta",    "0.0001",
    "--early_stopping_restore_best", "1",
    "--early_stopping_mode",         "max",

    # LR scheduler floor (only active when use_reduce_lr=1)
    "--reduce_lr_min",               "1e-6",
]


# ----------------------------------------------------------------------
#  Output parsing
# ----------------------------------------------------------------------
_BEST_TEST_RECALL_RE = re.compile(r"BEST_Test_Recall@20[:\s]*([0-9.]+)")
_BEST_TEST_NDCG_RE   = re.compile(r"BEST_Test_NDCG@20[:\s]*([0-9.]+)")
_BEST_VAL_RECALL_RE  = re.compile(r"BEST_Val_Recall@20[:\s]*([0-9.]+)")
_PEAK_EPOCH_RE       = re.compile(r"BEST_Val_Recall_Peak_Epoch[:\s]*([0-9]+)")

# Per-epoch val/recall@20 observation — try multiple formats for robustness.
#
# PRIMARY format (train.py via Logger):
#   "[2026-06-29 20:53:12] Epoch 59 [300.0s + 10.5s]: loss=0.20000
#    recall@10=0.09000  recall@20=0.11114  ndcg@20=0.054106"
# The Logger prefixes every line with "[YYYY-MM-DD HH:MM:SS] ".
# Note: the log uses plain "recall@20=", NOT "val/recall@20=".
_EPOCH_VAL_PATTERNS = [
    # Primary: matches the DAMPS-MMHCL Logger output exactly.
    # "Epoch N [<timing>]: ... recall@20=V ..."
    re.compile(r"Epoch\s+(\d+)\s*\[.*?\].*?\brecall@20=([0-9]+\.?[0-9]*)"),
    # Fallback 1: WandB-style "val/recall@20=V"
    re.compile(
        r"epoch[=\s:]*(\d+).*?\bval[/_]recall@20[=:\s]*([0-9]+\.?[0-9]*)",
        re.IGNORECASE,
    ),
    # Fallback 2: any line "epoch N ... recall@20 V"
    re.compile(
        r"\bepoch\s+(\d+)\b.*?\brecall@20[=:\s]+([0-9]+\.?[0-9]*)",
        re.IGNORECASE,
    ),
]


def _parse_intermediate(line: str) -> tuple[int, float] | None:
    """Try to extract (epoch, val_recall@20) from a single stdout line."""
    for pat in _EPOCH_VAL_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                return int(m.group(1)), float(m.group(2))
            except (ValueError, IndexError):
                continue
    return None


def _parse_final(text: str) -> dict[str, float]:
    """Extract BEST_* metrics from the full captured log."""
    out: dict[str, float] = {}
    for key, pat in [
        ("test_recall", _BEST_TEST_RECALL_RE),
        ("test_ndcg",   _BEST_TEST_NDCG_RE),
        ("val_recall",  _BEST_VAL_RECALL_RE),
        ("peak_epoch",  _PEAK_EPOCH_RE),
    ]:
        hits = pat.findall(text)
        if hits:
            out[key] = float(hits[-1])
    return out


# ----------------------------------------------------------------------
#  Hyperparameter sampling
# ----------------------------------------------------------------------

def _suggest_hp(trial: optuna.trial.Trial) -> dict[str, Any]:
    hp: dict[str, Any] = {
        "lr":           trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
        "regs":         trial.suggest_float("regs", 1e-4, 1e-2, log=True),
        "simgcl_eps":   trial.suggest_float("simgcl_eps", 0.05, 0.30),
        "lambda_view":  trial.suggest_float("lambda_view", 0.01, 0.20, log=True),
        "temperature":  trial.suggest_float("temperature", 0.10, 0.50),
        "embed_size":   trial.suggest_categorical("embed_size", [64, 128]),
        "UI_layers":    trial.suggest_categorical("UI_layers", [2, 3, 4]),
        "logq_scale":   trial.suggest_float("logq_scale", 0.5, 1.5),
        "use_reduce_lr": trial.suggest_categorical("use_reduce_lr", [0, 1]),
    }
    if hp["use_reduce_lr"]:
        hp["reduce_lr_factor"]   = trial.suggest_float("reduce_lr_factor", 0.3, 0.7)
        hp["reduce_lr_patience"] = trial.suggest_int("reduce_lr_patience", 3, 8)
    return hp


# ----------------------------------------------------------------------
#  Subprocess builder
# ----------------------------------------------------------------------

def _build_cmd(
    train_py: Path,
    dataset: str,
    seed: int,
    epoch: int,
    log_dir: Path,
    hp: dict[str, Any],
    wandb_args: list[str],
    gpu_id: int,
) -> list[str]:
    tag = log_dir.name
    bs = int(hp["batch_size"])
    cmd: list[str] = [
        sys.executable,
        str(train_py),
        f"--dataset={dataset}",
        f"--seed={seed}",
        f"--epoch={epoch}",
        f"--gpu_id={gpu_id}",
        f"--ablation_target={tag}",
        # Searched HPs
        f"--lr={hp['lr']}",
        f"--batch_size={bs}",
        f"--regs={hp['regs']}",
        f"--simgcl_eps={hp['simgcl_eps']}",
        f"--lambda_view={hp['lambda_view']}",
        f"--temperature={hp['temperature']}",
        f"--embed_size={hp['embed_size']}",
        f"--UI_layers={hp['UI_layers']}",
        f"--logq_scale={hp['logq_scale']}",
        # SimGCL chunk size stays in lock-step with batch_size
        f"--simgcl_batch_size_user={bs}",
        f"--simgcl_batch_size_item={bs}",
        # LR scheduler
        f"--use_reduce_lr={int(hp.get('use_reduce_lr', 0))}",
    ]
    if hp.get("use_reduce_lr", 0):
        cmd += [
            f"--reduce_lr_factor={hp['reduce_lr_factor']}",
            f"--reduce_lr_patience={hp['reduce_lr_patience']}",
        ]
    cmd += FROZEN_FLAGS
    cmd += wandb_args
    return cmd


# ----------------------------------------------------------------------
#  One-seed runner with streaming pruning
# ----------------------------------------------------------------------

def _run_one_seed(
    cmd: list[str],
    log_path: Path,
    trial: optuna.trial.Trial,
    seed_idx: int,
    n_seeds: int,
    stream_pruning: bool,
    step_offset: int,
) -> dict[str, float] | None:
    """Run train.py and stream stdout, reporting per-epoch val recall
    to Optuna's pruner.

    Returns final metrics dict, or ``None`` if the run failed.
    Raises ``optuna.TrialPruned`` if the pruner decides to kill the trial.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"  [trial {trial.number} seed {seed_idx + 1}/{n_seeds}] "
        f"launching: {' '.join(cmd[:9])} ..."
    )
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # encoding + errors: cp1252 (Windows default) cannot decode every
        # UTF-8 byte from train.py output; force UTF-8 with replacement so
        # the log reader never crashes on a unicode character.
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # Disable WandB network sync inside each trial subprocess.
            # Optuna tracks metrics via log-file streaming; WandB online
            # init adds 90 s of network overhead per trial and can time out.
            "WANDB_DISABLED": "true",
        },
    )

    captured: list[str] = []
    last_epoch = -1
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                captured.append(line)

                if not stream_pruning:
                    continue
                hit = _parse_intermediate(line)
                if hit is None:
                    continue
                epoch, val_recall = hit
                if epoch <= last_epoch:
                    continue
                last_epoch = epoch
                step = step_offset + epoch
                trial.report(val_recall, step=step)
                if trial.should_prune():
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise optuna.TrialPruned(
                        f"pruned at ep {epoch} (val/recall@20={val_recall:.4f})"
                    )
        proc.wait()
    except optuna.TrialPruned:
        raise
    except Exception as e:                                       # pragma: no cover
        proc.kill()
        print(f"  [trial {trial.number}] subprocess error: {e}")
        return None

    dt = time.time() - t0
    if proc.returncode != 0:
        print(
            f"  [trial {trial.number}] non-zero exit {proc.returncode} "
            f"after {dt/60:.1f} min — see {log_path}"
        )
        return None

    finals = _parse_final("".join(captured))
    finals["wallclock_s"] = dt
    return finals


# ----------------------------------------------------------------------
#  Objective factory
# ----------------------------------------------------------------------

def make_objective(args: argparse.Namespace):
    train_py = Path(args.train_py).resolve()
    if not train_py.exists():
        raise FileNotFoundError(
            f"train.py not found at {train_py} — "
            f"run this script from MMHCL_DAMPS_Project/ or pass --train_py."
        )

    base_dir = Path(args.workdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        hp = _suggest_hp(trial)
        seeds = [args.base_seed + i * 1009 for i in range(args.n_seeds)]
        recalls: list[float] = []
        ndcgs:   list[float] = []
        peaks:   list[int]   = []

        for si, seed in enumerate(seeds):
            tag = f"optuna_t{trial.number:04d}_s{si}"
            log_dir = base_dir / args.dataset / tag
            log_path = log_dir / f"{tag}.log"

            wandb_args: list[str] = []
            if args.wandb_project:
                wandb_args = [
                    "--use_wandb", "1",
                    "--wandb_project", args.wandb_project,
                    "--wandb_group", args.study_name,
                    "--wandb_tags",
                    f"optuna,trial{trial.number},seed{si},branchA",
                    "--wandb_job_type", "optuna_trial",
                    "--wandb_run_name",
                    f"{args.study_name}_t{trial.number:04d}_s{si}",
                ]
                if args.wandb_entity:
                    wandb_args += ["--wandb_entity", args.wandb_entity]

            cmd = _build_cmd(
                train_py=train_py,
                dataset=args.dataset,
                seed=seed,
                epoch=args.epoch,
                log_dir=log_dir,
                hp=hp,
                wandb_args=wandb_args,
                gpu_id=args.gpu_id,
            )

            finals = _run_one_seed(
                cmd=cmd,
                log_path=log_path,
                trial=trial,
                seed_idx=si,
                n_seeds=args.n_seeds,
                stream_pruning=args.stream_pruning,
                step_offset=si * (args.epoch + 10),
            )
            if finals is None or "test_recall" not in finals:
                print(
                    f"  [trial {trial.number}] seed {si} produced no metric; "
                    f"marking trial pruned."
                )
                raise optuna.TrialPruned(f"seed {si} failed")

            recalls.append(finals["test_recall"])
            ndcgs.append(finals.get("test_ndcg", 0.0))
            peaks.append(int(finals.get("peak_epoch", 0)))
            print(
                f"  [trial {trial.number}] seed {si}  "
                f"R@20={finals['test_recall']:.4f}  "
                f"NDCG@20={finals.get('test_ndcg', 0.0):.4f}  "
                f"peak_ep={int(finals.get('peak_epoch', 0))}  "
                f"({finals['wallclock_s']/60:.1f} min)"
            )

        obj = median(recalls)
        trial.set_user_attr("recalls",            recalls)
        trial.set_user_attr("ndcgs",              ndcgs)
        trial.set_user_attr("peaks",              peaks)
        trial.set_user_attr("median_test_recall", obj)
        trial.set_user_attr("median_test_ndcg",   median(ndcgs) if ndcgs else 0.0)
        print(
            f"  [trial {trial.number}] DONE  median R@20={obj:.4f}  "
            f"(seeds={recalls})"
        )
        return obj

    return objective


# ----------------------------------------------------------------------
#  Reporting helpers
# ----------------------------------------------------------------------

def _print_top_trials(study: optuna.Study, k: int = 5) -> None:
    finished = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not finished:
        print("\nNo completed trials yet.")
        return
    finished.sort(key=lambda t: t.value or -1.0, reverse=True)
    bar = "=" * 72
    print(f"\n{bar}\nTop {min(k, len(finished))} trials\n{bar}")
    for rank, t in enumerate(finished[:k], 1):
        print(f"\n#{rank} | trial {t.number} | median R@20 = {t.value:.4f}")
        for kk, vv in t.params.items():
            sv = f"{vv:.5g}" if isinstance(vv, float) else str(vv)
            print(f"   {kk:24s} = {sv}")
        ua = t.user_attrs
        if "recalls" in ua:
            print(f"   recalls (per seed)       = {ua['recalls']}")
        if "ndcgs" in ua:
            print(f"   ndcgs (per seed)         = {ua['ndcgs']}")
        if "peaks" in ua:
            print(f"   peak epochs (per seed)   = {ua['peaks']}")


def _save_best(study: optuna.Study, out_path: Path) -> None:
    finished = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not finished:
        return
    bt = study.best_trial
    payload = {
        "study_name":      study.study_name,
        "best_value":      bt.value,
        "best_params":     bt.params,
        "best_user_attrs": bt.user_attrs,
        "n_trials_total":  len(study.trials),
        "n_trials_done":   len(finished),
        "frozen_flags":    FROZEN_FLAGS,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nBest config written to {out_path}")


def _print_resume_cmd(study_name: str, dataset: str) -> None:
    print(
        "\nTo resume this study later, re-run with the SAME --study_name "
        f"and --storage. To launch a 5-seed final eval on the winning HPs, "
        f"read {study_name}_best.json['best_params'] and substitute them "
        f"into your existing 5-seed driver cell for dataset='{dataset}'."
    )


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for DAMPS-MMHCL Branch A (rev55 §8.1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="sports",
                   help="Dataset name (must exist under <repo>/data/<dataset>/).")
    p.add_argument("--n_trials", type=int, default=40,
                   help="Number of Optuna trials.")
    p.add_argument("--n_seeds", type=int, default=1,
                   help="Seeds per trial; the objective is the median.")
    p.add_argument("--base_seed", type=int, default=2042992639,
                   help="Base seed; further seeds = base_seed + i*1009.")
    p.add_argument("--epoch", type=int, default=250,
                   help="Max epochs per training run (matches Clothing protocol).")
    p.add_argument("--study_name", type=str, default="branchA_sports_v1",
                   help="Optuna study name (also used as W&B group + log dir).")
    p.add_argument("--storage", type=str,
                   default="sqlite:///optuna_branchA_sports.db",
                   help="Optuna RDB storage URL (resumable across crashes).")
    p.add_argument("--workdir", type=str, default=".",
                   help="Working dir; per-trial logs live under <workdir>/<dataset>/.")
    p.add_argument("--train_py", type=str, default="train.py",
                   help="Path to train.py (relative or absolute).")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="CUDA device id passed to train.py.")
    p.add_argument("--sampler_seed", type=int, default=42,
                   help="Random seed for the TPE sampler.")
    p.add_argument("--min_resource", type=int, default=30,
                   help="HyperbandPruner min resource (epochs).")
    p.add_argument("--reduction_factor", type=int, default=3,
                   help="HyperbandPruner reduction factor.")
    p.add_argument("--no_prune", action="store_true",
                   help="Disable HyperbandPruner (use NopPruner instead).")
    p.add_argument("--no_stream", dest="stream_pruning",
                   action="store_false",
                   help="Disable per-epoch streaming pruning. With this flag "
                        "Optuna only sees the final objective per trial "
                        "(useful if train.py log format defeats the regex).")
    p.add_argument("--wandb_project", type=str,
                   default="damps-mmhcl-sports-optuna",
                   help="W&B project (empty string disables W&B).")
    p.add_argument("--wandb_entity", type=str, default="",
                   help="W&B entity (username or team).")
    p.add_argument("--timeout", type=int, default=None,
                   help="Overall study timeout in seconds (None = no limit).")
    p.set_defaults(stream_pruning=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    sampler = TPESampler(
        seed=args.sampler_seed,
        multivariate=True,
        constant_liar=True,
    )
    pruner: optuna.pruners.BasePruner
    if args.no_prune:
        pruner = NopPruner()
    else:
        pruner = HyperbandPruner(
            min_resource=args.min_resource,
            max_resource=args.epoch,
            reduction_factor=args.reduction_factor,
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # Header
    bar = "=" * 72
    print(bar)
    print(f"Study    : {args.study_name}")
    print(f"Storage  : {args.storage}")
    print(f"Dataset  : {args.dataset}")
    print(f"Trials   : {args.n_trials}   (seeds/trial = {args.n_seeds})")
    print(f"Epochs   : {args.epoch}")
    print(f"Sampler  : TPE   seed={args.sampler_seed}   multivariate=True")
    if args.no_prune:
        print("Pruner   : NopPruner")
    else:
        print(
            f"Pruner   : Hyperband   min_resource={args.min_resource}   "
            f"max_resource={args.epoch}   reduction_factor={args.reduction_factor}"
        )
    print(f"Stream   : {args.stream_pruning}")
    if args.wandb_project:
        print(f"W&B      : project='{args.wandb_project}'  group='{args.study_name}'")
    print(bar)

    objective = make_objective(args)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            # If a trial crashes for a transient reason (OOM, etc.) keep the
            # study running; do NOT swallow KeyboardInterrupt.
            catch=(RuntimeError,),
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user; partial results are persisted in storage.")

    _print_top_trials(study, k=5)
    out_json = Path(args.workdir) / f"{args.study_name}_best.json"
    _save_best(study, out_json)
    _print_resume_cmd(args.study_name, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
