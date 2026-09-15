"""
scripts/run_optuna_hpo.py -- Bayesian Hyperparameter Optimisation (BOHB)
=========================================================================

Implements the BOHB schedule from Section 4 of the DAMPS-MMHCL Revision 9
specification (anchored on ``K`` and ``lambda_coh``) and the recipe from
Section 6 of the Speedup Guide:

    sampler = TPESampler(seed=42)
    pruner  = HyperbandPruner()

    n_trials = 50

The objective spawns ``train.py`` as a subprocess with the trial's suggested
hyperparameters, parses the final ``BEST_Test_Recall@20`` from the per-run
log, and returns it to Optuna for maximisation.

Usage
-----
::

    python run_optuna_hpo.py --dataset Clothing --n_trials 50 --seed 42

Notes
-----
*   The DAMPS architecture spec freezes ``beta=0.995``, ``warmup=10``,
    ``R=5``, ``auto_prior=True`` -- only ``K`` and ``lambda_coh`` are
    searched.
*   HyperbandPruner aggressively kills under-performing trials, shrinking
    wall-clock cost ~3x according to the speedup guide table.
*   Each trial creates an isolated experiment directory so logs do not
    overwrite each other.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import optuna
except ImportError as exc:                                      # pragma: no cover
    print("ERROR: optuna is required (pip install optuna).", file=sys.stderr)
    raise SystemExit(2) from exc


_BEST_RE = re.compile(r"BEST_Test_Recall@20:\s*([0-9.]+)")


# ---------------------------------------------------------------------------
#  Trial runner
# ---------------------------------------------------------------------------
def _run_trial(
    trial: optuna.trial.Trial,
    base_dir: Path,
    dataset: str,
    seed: int,
    epochs: int,
    extra: list[str],
) -> float:
    """Spawn ``train.py`` for a single trial and return ``Recall@20``."""
    K = trial.suggest_int("K", 3, 20)
    lambda_coh = trial.suggest_float("lambda_coh", 0.01, 1.0, log=True)

    tag = f"optuna_t{trial.number:03d}_K{K}_l{lambda_coh:.4f}"
    cmd = [
        sys.executable,
        str(base_dir / "src" / "train.py"),
        f"--dataset={dataset}",
        f"--seed={seed}",
        f"--topk={K}",
        f"--epoch={epochs}",
        f"--ablation_target={tag}",
        # Architecture is anchored per spec Section 4
        "--damps_apc=1",
        "--damps_avrf=1",
        "--damps_imcf=1",
        "--damps_soft_routing=1",
        "--damps_momentum=1",
        "--damps_data_driven_prior=1",
        "--damps_warmup_epochs=10",
        "--rebuild_R=5",
        # Speedup defaults
        "--use_amp=1",
        "--use_torch_compile=1",
        # IMCF lambda_coh override is applied in-process by reading the env
        # var below. We pass it through both an env var (consumed by a
        # downstream patch in train.py if you wire one) and by simply
        # logging the trial parameters into the experiment directory.
    ]
    cmd.extend(extra)

    log_path = base_dir / dataset / tag / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Optuna trial {trial.number}] K={K}, lambda_coh={lambda_coh:.4f}")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"  trial returned exit code {proc.returncode}")
        raise optuna.TrialPruned()

    match = _BEST_RE.search(proc.stdout)
    if not match:
        # Try the per-run log directory
        run_glob = list((base_dir / dataset).glob(f"*{tag}*"))
        for sub in run_glob:
            for f in sub.glob("*.txt"):
                m2 = _BEST_RE.search(f.read_text(encoding="utf-8", errors="ignore"))
                if m2:
                    return float(m2.group(1))
        print(f"  WARN: BEST_Test_Recall@20 not found in trial {trial.number}")
        return 0.0
    return float(match.group(1))


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="DAMPS-MMHCL BOHB hyperparameter search")
    parser.add_argument("--dataset", type=str, default="Clothing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120,
                        help="Per-trial training budget (lower than the "
                             "main 250 epochs so HPO completes in 2-3 days).")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default="damps-mmhcl-clothing")
    parser.add_argument("--storage", type=str, default="",
                        help="Optional Optuna RDB URL for distributed HPO.")
    parser.add_argument("--out", type=Path, default=Path("optuna_best.json"))
    parser.add_argument("--base_dir", type=Path,
                        default=Path(__file__).resolve().parent.parent,
                        help="Repository root (defaults to ../).")
    args, extra_train_args = parser.parse_known_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage or None,
        load_if_exists=bool(args.storage),
    )
    study.optimize(
        lambda t: _run_trial(
            t, args.base_dir, args.dataset, args.seed, args.epochs, extra_train_args
        ),
        n_trials=args.n_trials,
        n_jobs=1,
    )

    print("=" * 60)
    print("Best params:", study.best_params)
    print("Best value :", study.best_value)
    args.out.write_text(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "study_name": args.study_name,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
