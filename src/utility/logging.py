"""
utility/logging.py — Dual-Destination Logger
==============================================

Writes timestamped messages to:
  1. The console (stdout).
  2. A per-experiment log file in ``<path>/<target>.txt``.
  3. Optionally, a copy in a shared aggregation directory ``<path2>``.

Used uniformly across DAMPS-MMHCL training, evaluation, and ablation runs.
"""

from __future__ import annotations

import os
import pathlib
from datetime import datetime
from typing import Optional


class Logger:
    """Timestamped file + console logger."""

    def __init__(
        self,
        path: str,
        is_debug: str = "True",
        target: str = "log",
        path2: Optional[str] = None,
        ablation_target: Optional[str] = None,
    ) -> None:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        if path2:
            pathlib.Path(path2).mkdir(parents=True, exist_ok=True)

        self.target: str = target
        self.path: str = path
        self.log_: str = is_debug
        self.path2: Optional[str] = path2
        self.ablation_target: Optional[str] = ablation_target

    # ------------------------------------------------------------------
    #  Primary write
    # ------------------------------------------------------------------
    def logging(self, msg: str) -> None:
        """Write a timestamped message to console + log file(s)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)

        if str(self.log_) == "True":
            primary = os.path.join(self.path, f"{self.target}.txt")
            with open(primary, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

            if self.path2:
                secondary = os.path.join(self.path2, f"{self.target}.txt")
                with open(secondary, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    # ------------------------------------------------------------------
    #  Aggregation summary (one-line per ablation)
    # ------------------------------------------------------------------
    def logging_sum(self, msg: str) -> None:
        """Append a one-line summary to ``<path2>/sum_<ablation>.txt``."""
        if not self.path2:
            return
        tag = self.ablation_target or "default"
        sum_path = os.path.join(self.path2, f"sum_{tag}.txt")
        with open(sum_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
