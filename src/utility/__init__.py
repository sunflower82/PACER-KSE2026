"""
utility — DAMPS-MMHCL helper modules.

Provides argument parsing, dataset loading, metric computation,
batched evaluation, and dual-destination logging.
"""

from .parser import parse_args
from .logging import Logger

__all__ = ["parse_args", "Logger"]
