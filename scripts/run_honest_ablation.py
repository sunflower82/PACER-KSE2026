#!/usr/bin/env python3
"""scripts/run_honest_ablation.py -- honest 2-cell architecture ablation.
=============================================================================
Runs the two honest-ablation configs on Clothing:
  * C0_pacer_full_new  = no LogQ + base cooc + TAMER on + NRDMC-lite on
                         (= B11 config from optb13 grid; new PACER-full)
  * C1_A1_noNRDMC      = C0 with NRDMC-lite disabled (single ablation cell)

Purpose: after the 5-seed rescue verdict (C) BOTH FAIL on Data (RSFP) and
Loss (LogQ) axes, this run establishes the single architectural axis
(NRDMC-lite) as the surviving PACER contribution.

Default seeds: the 5 KSE-locked seeds (matches PACER v11 protocol).
Default epoch cap: 100 (mirrors the Option B rescue protocol; best_val_epoch
was <=~60 for all previous B11 runs so a 100-epoch cap is well above the
model-selection window). Use --epoch 250 if you want to match the shipping
KSE-final cap (typically ~1.5x wall).

Usage (Windows):
    python scripts/run_honest_ablation.py --epoch 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_SEEDS = [1616406634, 1640104851, 52093548, 109649638, 372270914]
TARGETS = ["C0_pacer_full_new", "C1_A1_noNRDMC"]


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_root = repo_root / "src"
    ap = argparse.ArgumentParser(
        description="2-cell honest ablation batch (C0 full vs C1 no-NRDMC)."
    )
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main", type=Path,
                    default=src_root / "main_tercile.py")
    ap.add_argument("--driver", type=Path,
                    default=here / "run_kse_final_5seed.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--output", type=Path,
                    default=repo_root / "results"
                                     / "honest_ablation_clothing.json")
    ap.add_argument("--log_dir", type=Path,
                    default=repo_root / "results"
                                     / "_honest_ablation_logs")
    ap.add_argument("--base_cache", type=Path,
                    default=repo_root / "results"
                                     / "interest_tree_clothing.npz")
    ap.add_argument("--only_tags", type=str, nargs="*", default=None,
                    help="Optionally restrict to C0 or C1 only.")
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tags = args.only_tags or TARGETS
    cmd = [
        args.python, str(args.driver),
        "--grid", "honest",
        "--only_tags", *tags,
        "--seeds", *[str(s) for s in args.seeds],
        "--epoch", str(args.epoch),
        "--python", args.python,
        "--main", str(args.main),
        "--output", str(args.output),
        "--log_dir", str(args.log_dir),
        "--base_cache", str(args.base_cache),
        "--dry_run", str(args.dry_run),
    ]
    n = len(tags) * len(args.seeds)
    print(f"[honest] tags   ({len(tags)}): {tags}")
    print(f"[honest] seeds  ({len(args.seeds)}): {args.seeds}")
    print(f"[honest] epoch cap: {args.epoch}")
    print(f"[honest] total runs: {n}  (est. wall "
          f"~{n * (18 if args.epoch <= 100 else 27) / 60:.1f} h)")
    print("=" * 72)
    return subprocess.run(cmd, cwd=src_root, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
