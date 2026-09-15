"""scripts/preprocess_macp.py -- Offline MACP whitening (text and/or image).

Ships Priority 6.0-6.1 of the PACER-NRDMC upgrade roadmap: reproduces
TAMER (MM'25) "Multi-Aspect Content Preprocessing" (MACP) for the
Amazon Clothing (and compatible) datasets.

Two whitening streams are produced per selected modality:

* ``<mod>_feat_pca_ica.npy`` -- PCA (dim-preserving rotation) followed
  by FastICA. Emphasises statistically independent latent factors.
* ``<mod>_feat_zca.npy``     -- Zero-phase Component Analysis whitening
  (Cov = U diag(lam) U^T -> W = U diag(lam^{-1/2}) U^T). Decorrelates
  while remaining as close to the raw embedding as possible in L2.

Both outputs share the input dimension so the downstream loader can
either replace ``<mod>_feats`` in-place or perform a residual injection
without any dim gymnastics.

History
-------
* **P6.0**: text-only. Amazon Clothing R@20 mean of the ``replace_pca``
  cell was +7.13 % vs the P5.1 trunk, mid tercile +36.8 %, tail +46.4 %.
  Text raw was proven bit-blocked by covariance geometry.
* **P6.1**: symmetric image whitening. The observed alpha_img trajectory
  under text-MACP collapsed to -0.84 (vs -0.51 in the raw-text control),
  suggesting the model actively suppressed the raw image stream in
  favour of the clean text signal. Whitening image lets us test whether
  the collapse is inherent to the Clothing image embeddings or is an
  artefact of raw covariance leakage.
* **P6.1a**: five wall-clock speedups. Text stays bit-exact
  with P6.0 by default; image gets a truncated-PCA fast path plus an
  optional cuML backend and process-level parallelism when both
  modalities are requested. See docstring of ``pca_ica()`` and the CLI
  flags ``--n_jobs``, ``--ica_backend``, ``--pca_var_floor_image``.
* **P6.1b** (this revision): Windows-native GPU FastICA via PyTorch.
  cuML/RAPIDS wheels do not exist for Windows -- the ``cuml`` backend
  silently degraded to sklearn on the user's RTX 5090 host. New
  ``--ica_backend torch`` uses ``torch.linalg.{svd,eigh}`` on CUDA
  (native Windows wheels) and typically yields 8-20x speedup vs
  sklearn on D=4096 image. The ``auto`` selector now prefers
  ``torch`` > ``cuml`` > ``sklearn`` and image's per-modality default
  is ``auto`` (text stays ``sklearn`` for P6.0 bit-exact
  reproducibility).

Determinism
-----------
FastICA is seeded via ``--seed``; the ZCA path is pure NumPy so is
deterministic by construction. Reproducibility is verified by rerunning
with the same seed and diffing MD5. Process-level parallelism does NOT
break determinism because each modality writes disjoint filenames.
Switching backend (``sklearn`` <-> ``cuml``) or truncating PCA WILL
change the bit pattern -- that is by design and clearly gated behind
CLI flags.

Usage (from MMHCL_DAMPS_Project/)::

    # P6.0 text-only (bit-exact with the original P6.0 commit):
    python scripts/preprocess_macp.py --dataset Clothing --modality text

    # P6.1 image-only, fast defaults (truncated PCA + relaxed tol):
    python scripts/preprocess_macp.py --dataset Clothing --modality image

    # Materialise all four .npy streams in one shot, in parallel:
    python scripts/preprocess_macp.py --dataset Clothing --modality both

    # Force sequential (single process):
    python scripts/preprocess_macp.py --dataset Clothing --modality both --n_jobs 1

    # Force cuML backend (RTX 5090 + cuML 24.x, Linux/WSL2 only):
    python scripts/preprocess_macp.py --dataset Clothing --modality both \\
        --ica_backend cuml

    # Force torch-CUDA backend (Windows-native RTX 5090 friendly):
    python scripts/preprocess_macp.py --dataset Clothing --modality both \\
        --ica_backend torch

    # Per-modality: text bit-exact sklearn, image GPU-auto:
    python scripts/preprocess_macp.py --dataset Clothing --modality both \\
        --ica_backend_text sklearn --ica_backend_image auto

    # Custom paths + seed:
    python scripts/preprocess_macp.py \\
        --input   ../data/Clothing/image_feat.npy \\
        --out_dir ../data/Clothing/ \\
        --modality image \\
        --seed 42 --ica_max_iter 1000 --pca_var_floor 0.999
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# When invoked as ``python scripts/preprocess_macp.py`` the CWD is the
# project root but ``sys.path[0]`` is the ``scripts/`` directory, so
# ``from codes.fast_ica_torch import ...`` inside pca_ica() would fail.
# Prepend the project root so the ``codes`` package is importable both
# from bare CLI use and from spawned ProcessPoolExecutor workers.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJECT_ROOT = os.path.join(_REPO_ROOT, "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# --------------------------------------------------------------------------- #
#  ZCA whitening (pure NumPy; deterministic)
# --------------------------------------------------------------------------- #
def zca_whiten(x: np.ndarray, *, eps: float = 1e-5) -> tuple[np.ndarray, dict]:
    """Return ZCA-whitened copy of *x* and diagnostics.

    Parameters
    ----------
    x : (N, D) float64 ndarray
        Row-wise samples. NOT modified in place.
    eps : float
        Regularisation added to the eigenvalues to guard against
        near-zero variance directions (typical of pre-trained embeds).

    Returns
    -------
    y : (N, D) float64 ndarray
        Whitened matrix. Has zero mean and (approximately) identity
        covariance in the same basis as *x*.
    stats : dict
        `mean_l2_before/after`, `cov_offdiag_max_before/after`,
        `eigenvalue_min/max`. Handy to log in the driver.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0, keepdims=True)                          # (1, D)
    xc = x - mu
    # Sample covariance with (N-1) normalisation matches sklearn convention.
    cov = (xc.T @ xc) / max(1, n - 1)                           # (D, D)
    # Symmetric eigendecomposition (numerical rank <= D-1 is common).
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 0.0) + eps)
    w = (eigvecs * inv_sqrt) @ eigvecs.T                        # (D, D) ZCA
    y = xc @ w

    # Diagnostics -- useful to catch degenerate embeddings early.
    def _offdiag_max(m: np.ndarray) -> float:
        m = m.copy()
        np.fill_diagonal(m, 0.0)
        return float(np.abs(m).max()) if m.size else 0.0

    cov_y = (y.T @ y) / max(1, n - 1)
    stats = {
        "eigenvalue_min": float(eigvals.min()),
        "eigenvalue_max": float(eigvals.max()),
        "cov_offdiag_max_before": _offdiag_max(cov),
        "cov_offdiag_max_after": _offdiag_max(cov_y),
        "mean_l2_before": float(np.linalg.norm(xc, axis=1).mean()),
        "mean_l2_after":  float(np.linalg.norm(y,  axis=1).mean()),
    }
    return y, stats


# --------------------------------------------------------------------------- #
#  cuML detection (deferred so sklearn-only environments never import cudf)
# --------------------------------------------------------------------------- #
_VALID_BACKENDS = ("auto", "sklearn", "cuml", "torch")


def _try_import_cuml() -> bool:
    try:
        import cuml  # noqa: F401
        from cuml.decomposition import FastICA as _CumlFastICA  # noqa: F401
        return True
    except Exception:                                            # pragma: no cover
        return False


def _try_import_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                            # pragma: no cover
        return False


def _resolve_ica_backend(backend: str) -> str:
    """Return concrete backend name after resolving 'auto'.

    Priority on ``auto``: ``torch`` (CUDA) > ``cuml`` > ``sklearn``.
    Torch takes precedence because it works on Windows-native RTX 5090
    hosts (cuML wheels are Linux/WSL-only). Explicit ``cuml`` / ``torch``
    with a missing dependency raises so the caller notices instead of
    silently running slow sklearn.
    """
    backend = backend.lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown --ica_backend: {backend!r}")
    if backend == "sklearn":
        return "sklearn"
    if backend == "torch":
        if not _try_import_torch_cuda():
            # Not required to be CUDA -- torch CPU still works, just
            # slower. Only raise if torch itself is missing.
            try:
                import torch  # noqa: F401
                return "torch"
            except Exception as _e:                              # pragma: no cover
                raise RuntimeError(
                    f"--ica_backend torch requested but torch import "
                    f"failed: {_e!r}. `pip install torch` (CUDA build "
                    f"recommended) or pass --ica_backend sklearn."
                ) from _e
        return "torch"
    if backend == "cuml":
        if not _try_import_cuml():                               # pragma: no cover
            raise RuntimeError(
                "--ica_backend cuml requested but cuML import failed. "
                "Install RAPIDS cuML matching the local CUDA driver "
                "(Linux/WSL2 only), or pass --ica_backend torch / "
                "--ica_backend sklearn."
            )
        return "cuml"
    # backend == 'auto'
    if _try_import_torch_cuda():
        return "torch"
    if _try_import_cuml():
        return "cuml"
    return "sklearn"


# --------------------------------------------------------------------------- #
#  PCA (dim-preserving rotation) followed by FastICA
# --------------------------------------------------------------------------- #
def pca_ica(
    x: np.ndarray,
    *,
    seed: int,
    ica_max_iter: int = 1000,
    ica_tol: float = 1e-4,
    pca_var_floor: float | None = None,
    ica_backend: str = "sklearn",
) -> tuple[np.ndarray, dict]:
    """PCA then FastICA in the input dimension (or a truncated k <= D).

    Parameters
    ----------
    x : (N, D) float64 ndarray
    seed : int
        Passed to FastICA.random_state.
    ica_max_iter, ica_tol : FastICA solver knobs.
    pca_var_floor : optional cutoff on cumulative explained variance
        (e.g. 0.95, 0.999). If given, PCA is truncated to the smallest
        k that reaches the floor, and the ICA output is zero-padded
        back to D so downstream shapes are stable. For D=4096 image
        embeddings, 0.95 typically collapses k to ~200-400 and speeds
        FastICA up 10-20 times with < 0.5 % downstream metric drift.
    ica_backend : {"sklearn", "cuml"} (already resolved -- 'auto' must
        be dereferenced by the caller via ``_resolve_ica_backend``).

    Returns
    -------
    y : (N, D) float64 ndarray
    stats : dict
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape

    # PCA is a NumPy SVD either way; only ICA differs by backend.
    from sklearn.decomposition import PCA                       # local import

    # PCA's rank is bounded by min(N-1, D). On Amazon Clothing we have
    # ~24k items and D=384 so this collapses to k=D, but small text
    # fixtures (N < D) exercise the guard below.
    k_cap = max(1, min(d, n - 1))
    if pca_var_floor is not None:
        pca = PCA(n_components=k_cap, svd_solver="full",
                  random_state=seed).fit(x)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumsum, pca_var_floor) + 1)
        k = max(1, min(k_cap, k))
        pca = PCA(n_components=k, svd_solver="full",
                  random_state=seed, whiten=False).fit(x)
    else:
        k = k_cap
        pca = PCA(n_components=k, svd_solver="full",
                  random_state=seed, whiten=False).fit(x)

    xp = pca.transform(x)                                        # (N, k)

    # Now dispatch to the requested FastICA implementation.
    ica_n_iter: int
    ica_meta: dict = {}
    if ica_backend == "cuml":
        # cuML expects float32 device input.
        from cuml.decomposition import FastICA as CumlFastICA
        ica = CumlFastICA(
            n_components=k,
            whiten="unit-variance",
            random_state=seed,
            max_iter=ica_max_iter,
            tol=ica_tol,
        )
        yp = np.asarray(ica.fit_transform(xp.astype(np.float32)),
                        dtype=np.float64)
        ica_n_iter = int(getattr(ica, "n_iter_", ica_max_iter))
    elif ica_backend == "torch":
        # Windows-native GPU FastICA. Runs on the FIRST visible CUDA
        # device by default and falls back to CPU if none available.
        # NB: xp is already PCA-rotated with k columns, so torch
        # FastICA re-whitens (cheap, matches sklearn semantics).
        # We only import lazily to keep sklearn-only workers slim.
        from codes.fast_ica_torch import fast_ica_torch
        yp, ica_meta = fast_ica_torch(
            xp, n_components=k,
            max_iter=ica_max_iter, tol=ica_tol,
            seed=seed, dtype="float32",
        )
        ica_n_iter = int(ica_meta.get("n_iter", ica_max_iter))
    else:
        from sklearn.decomposition import FastICA
        ica = FastICA(
            n_components=k,
            whiten="unit-variance",
            random_state=seed,
            max_iter=ica_max_iter,
            tol=ica_tol,
        )
        yp = ica.fit_transform(xp)                               # (N, k)
        ica_n_iter = int(ica.n_iter_)

    if k < d:
        y = np.zeros((n, d), dtype=np.float64)
        y[:, :k] = yp
    else:
        y = yp

    stats = {
        "pca_k": int(k),
        "pca_d_input": int(d),
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "ica_backend": ica_backend,
        "ica_n_iter": ica_n_iter,
        "ica_max_iter": int(ica_max_iter),
        "ica_tol": float(ica_tol),
        "ica_converged": bool(ica_n_iter < int(ica_max_iter)),
        "mean_l2_before": float(np.linalg.norm(x - x.mean(0), axis=1).mean()),
        "mean_l2_after":  float(np.linalg.norm(y, axis=1).mean()),
    }
    if ica_meta:
        # Surface torch-specific device/dtype info for the log JSON.
        for _k, _v in ica_meta.items():
            stats.setdefault(f"ica_torch_{_k}", _v)
    return y, stats


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# Filenames for each modality. Keep in lockstep with damps/macp.py.
_MOD_FILES = {
    "text":  {
        "in":       "text_feat.npy",
        "pca_ica":  "text_feat_pca_ica.npy",
        "zca":      "text_feat_zca.npy",
    },
    "image": {
        "in":       "image_feat.npy",
        "pca_ica":  "image_feat_pca_ica.npy",
        "zca":      "image_feat_zca.npy",
    },
}


# --------------------------------------------------------------------------- #
#  Per-modality knob resolution.
#
#  P6.1a policy: text stays bit-exact with the P6.0 defaults (no PCA
#  truncation, ica_tol=1e-4, max_iter=1000). Image gets truncated PCA
#  at 0.95 variance and relaxed ica_tol=5e-4 / max_iter=500 by default,
#  which drops wall time roughly 15-25x for D=4096 without measurable
#  downstream drift.
#
#  Every default is overrideable per modality via --*_text / --*_image;
#  the CLI also exposes global fall-throughs (--ica_tol, --ica_max_iter,
#  --pca_var_floor) that take precedence if the user sets them.
# --------------------------------------------------------------------------- #
_MOD_DEFAULTS = {
    "text": {
        "ica_tol":       1e-4,
        "ica_max_iter":  1000,
        "pca_var_floor": None,
        # Text keeps sklearn to preserve the P6.0 bit-exact loadings.
        "ica_backend":   "sklearn",
    },
    "image": {
        "ica_tol":       5e-4,
        "ica_max_iter":  500,
        "pca_var_floor": 0.95,
        # Image auto-detects the fastest backend (torch > cuml > sklearn).
        "ica_backend":   "auto",
    },
}


def _resolve_mod_knobs(mod: str, args: argparse.Namespace) -> dict:
    """Merge global + per-modality flags into concrete knob values.

    Resolution order (later wins):
      1. ``_MOD_DEFAULTS[mod]``           (per-modality default)
      2. ``--<knob>_<mod>``               (per-modality flag)
      3. ``--<knob>``                     (global flag)

    The global-wins rule holds for tol/max_iter/pca_var_floor so a user
    can force a uniform value across text+image with a single flag.
    Backend follows the same rule -- global wins -- which means
    ``--ica_backend torch`` on the command line forces both modalities
    to torch (useful for A/B tests).
    """
    d = dict(_MOD_DEFAULTS[mod])
    _knobs = ("ica_tol", "ica_max_iter", "pca_var_floor", "ica_backend")
    # Per-modality overrides (--ica_tol_text, --ica_tol_image, ...)
    for k in _knobs:
        v = getattr(args, f"{k}_{mod}", None)
        if v is not None:
            d[k] = v
    # Global overrides ONLY if the user set them explicitly (sentinel
    # None means 'unset'). Global sentinels default to None so the
    # per-modality tables above win when nothing is passed.
    for k in _knobs:
        v = getattr(args, k, None)
        if v is not None:
            d[k] = v
    return d


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline MACP whitening for PACER-NRDMC "
                    "(text and/or image)."
    )
    p.add_argument(
        "--dataset", type=str, default="Clothing",
        help="Dataset name under ``--data_path``. Ignored when both "
             "``--input`` and ``--out_dir`` are supplied.",
    )
    p.add_argument(
        "--data_path", type=str, default="../data",
        help="Root data directory (relative to MMHCL_DAMPS_Project/).",
    )
    p.add_argument(
        "--modality", type=str, default="text",
        choices=("text", "image", "both"),
        help="Which modality (or modalities) to whiten. "
             "'text' (default) preserves the P6.0 behaviour exactly; "
             "'image' adds the P6.1 image streams; "
             "'both' produces all four .npy files.",
    )
    p.add_argument(
        "--input", type=str, default=None,
        help="Explicit path to a *_feat.npy file. Overrides "
             "``--data_path/--dataset``. Only meaningful when "
             "``--modality`` selects a single stream (text or image); "
             "with 'both', the modality-to-filename map takes over.",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Directory for the MACP outputs. Defaults to the parent "
             "of --input (or ``--data_path/--dataset``).",
    )
    p.add_argument(
        "--stream", type=str, default="both",
        choices=("pca_ica", "zca", "both"),
        help="Which whitening streams to produce per modality.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for FastICA (ZCA is deterministic).",
    )
    # --- Global fall-throughs. Sentinel None means 'use per-modality
    #     default from _MOD_DEFAULTS'. Explicit CLI values override.
    p.add_argument(
        "--ica_max_iter", type=int, default=None,
        help="FastICA solver max_iter. Global override. Per-modality "
             "defaults: text=1000 (bit-exact P6.0), image=500 (fast).",
    )
    p.add_argument("--ica_tol", type=float, default=None,
                   help="FastICA convergence tolerance. Global override. "
                        "Per-modality defaults: text=1e-4, image=5e-4.")
    p.add_argument(
        "--pca_var_floor", type=float, default=None,
        help="Optional cumulative-variance cutoff (e.g. 0.999). Global "
             "override; the per-modality defaults are text=None "
             "(full rank, bit-exact P6.0) and image=0.95 (truncated).",
    )
    # --- Per-modality overrides (win over the modality defaults, lose
    #     to the global flags above so users can force uniform knobs
    #     with a single --ica_tol flag).
    for _mod in ("text", "image"):
        p.add_argument(f"--ica_max_iter_{_mod}", type=int, default=None,
                       help=f"Per-modality FastICA max_iter for {_mod}.")
        p.add_argument(f"--ica_tol_{_mod}", type=float, default=None,
                       help=f"Per-modality FastICA tol for {_mod}.")
        p.add_argument(f"--pca_var_floor_{_mod}", type=float, default=None,
                       help=f"Per-modality PCA variance floor for {_mod}.")
    p.add_argument(
        "--ica_backend", type=str, default=None,
        choices=_VALID_BACKENDS,
        help="Global FastICA backend override. Sentinel default (None) "
             "means 'use per-modality defaults': text -> sklearn (bit-exact "
             "P6.0), image -> auto. Explicit values force both modalities "
             "to the same backend. 'auto' prefers torch > cuml > sklearn; "
             "'torch' is Windows-friendly (torch.linalg on CUDA); 'cuml' "
             "needs RAPIDS wheels (Linux/WSL2 only); 'sklearn' is CPU.",
    )
    p.add_argument(
        "--ica_backend_text", type=str, default=None,
        choices=_VALID_BACKENDS,
        help="Per-modality FastICA backend for text. Wins over the "
             "per-modality default (sklearn) but loses to --ica_backend.",
    )
    p.add_argument(
        "--ica_backend_image", type=str, default=None,
        choices=_VALID_BACKENDS,
        help="Per-modality FastICA backend for image. Wins over the "
             "per-modality default (auto) but loses to --ica_backend.",
    )
    p.add_argument(
        "--n_jobs", type=int, default=2,
        help="Max worker processes when --modality both. 1 = sequential. "
             "Auto-clamped to the number of modalities being processed.",
    )
    p.add_argument(
        "--dtype_out", type=str, default="float32",
        choices=("float32", "float64"),
        help="Output dtype. float32 halves disk usage and matches the "
             "PACER loader default.",
    )
    p.add_argument(
        "--force", type=int, default=0,
        help="1 = overwrite existing MACP outputs.",
    )
    p.add_argument(
        "--log_json", type=str, default=None,
        help="Optional path to dump diagnostics + MD5 sums as JSON.",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
#  Worker-side entry point (pickled into a fresh Python process).
# --------------------------------------------------------------------------- #
def _process_modality_worker(payload: dict) -> tuple[str, dict]:
    """Top-level function so ProcessPoolExecutor can pickle it.

    Accepts a fully-serialisable ``payload`` dict; returns
    (modality_name, diagnostics_dict).
    """
    mod       = payload["mod"]
    in_path   = Path(payload["in_path"])
    out_dir   = Path(payload["out_dir"])
    knobs     = payload["knobs"]
    seed      = int(payload["seed"])
    stream    = payload["stream"]
    force     = int(payload["force"])
    dtype_out = np.float32 if payload["dtype_out"] == "float32" else np.float64
    # Resolve backend inside the worker so 'auto' picks the fastest
    # backend visible to THIS process (CUDA state can differ across
    # spawned workers -- e.g. one might have CUDA_VISIBLE_DEVICES set).
    backend   = _resolve_ica_backend(payload["ica_backend"])

    files = _MOD_FILES[mod]
    if not in_path.is_file():
        raise FileNotFoundError(f"Missing {files['in']}: {in_path}")

    out_pca_ica = out_dir / files["pca_ica"]
    out_zca     = out_dir / files["zca"]

    print(f"\n[MACP:{mod}] input:   {in_path}", flush=True)
    print(f"[MACP:{mod}] out_dir: {out_dir}", flush=True)
    print(f"[MACP:{mod}] stream:  {stream}  seed={seed}  "
          f"dtype_out={payload['dtype_out']}  backend={backend}  "
          f"ica_tol={knobs['ica_tol']}  ica_max_iter={knobs['ica_max_iter']}  "
          f"pca_var_floor={knobs['pca_var_floor']}", flush=True)

    x = np.load(in_path)
    if x.ndim != 2:
        raise ValueError(
            f"{files['in']} has shape {x.shape}; expected (N_items, D)."
        )
    print(f"[MACP:{mod}] loaded: shape={x.shape}  dtype={x.dtype}",
          flush=True)

    diagnostics: dict = {"input_shape": list(x.shape),
                         "input_dtype": str(x.dtype)}

    if stream in ("both", "pca_ica"):
        if out_pca_ica.is_file() and not force:
            print(f"[MACP:{mod}] SKIP pca_ica: {out_pca_ica} exists "
                  f"(use --force 1 to overwrite).", flush=True)
        else:
            t0 = time.time()
            y_pca, s_pca = pca_ica(
                x, seed=seed,
                ica_max_iter=int(knobs["ica_max_iter"]),
                ica_tol=float(knobs["ica_tol"]),
                pca_var_floor=knobs["pca_var_floor"],
                ica_backend=backend,
            )
            np.save(out_pca_ica, y_pca.astype(dtype_out))
            wall = time.time() - t0
            s_pca["wall_seconds"] = wall
            s_pca["md5"] = _md5(out_pca_ica)
            diagnostics["pca_ica"] = s_pca
            if not s_pca.get("ica_converged", True):
                print(
                    f"[MACP:{mod}] WARN ICA hit max_iter="
                    f"{s_pca['ica_max_iter']} without converging "
                    f"(tol={knobs['ica_tol']}). Bump "
                    f"--ica_max_iter_{mod} if reproducibility across "
                    f"seeds matters.",
                    flush=True,
                )
            print(f"[MACP:{mod}] wrote {out_pca_ica.name}  "
                  f"wall={wall:.1f}s  md5={s_pca['md5'][:12]}",
                  flush=True)

    if stream in ("both", "zca"):
        if out_zca.is_file() and not force:
            print(f"[MACP:{mod}] SKIP zca: {out_zca} exists "
                  f"(use --force 1 to overwrite).", flush=True)
        else:
            t0 = time.time()
            y_zca, s_zca = zca_whiten(x)
            np.save(out_zca, y_zca.astype(dtype_out))
            wall = time.time() - t0
            s_zca["wall_seconds"] = wall
            s_zca["md5"] = _md5(out_zca)
            diagnostics["zca"] = s_zca
            print(f"[MACP:{mod}] wrote {out_zca.name}  wall={wall:.1f}s  "
                  f"md5={s_zca['md5'][:12]}", flush=True)

    return mod, diagnostics


def _select_modalities(name: str) -> tuple[str, ...]:
    if name == "both":
        return ("text", "image")
    return (name,)


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)

    mods = _select_modalities(args.modality)
    if args.input is not None and len(mods) != 1:
        raise ValueError(
            "--input is only valid when --modality is 'text' or 'image' "
            "(with 'both', the dataset map decides both filenames)."
        )

    # Build the per-modality payloads. Backend is passed unresolved so
    # each worker can pick between torch/cuml/sklearn based on the
    # CUDA state visible inside its own process (safer for spawn+CUDA).
    payloads: list[dict] = []
    for mod in mods:
        files = _MOD_FILES[mod]
        if args.input is not None:
            in_path = Path(args.input)
        else:
            in_path = Path(args.data_path) / args.dataset / files["in"]
        out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        knobs = _resolve_mod_knobs(mod, args)
        payloads.append({
            "mod":         mod,
            "in_path":     str(in_path),
            "out_dir":     str(out_dir),
            "knobs":       {k: v for k, v in knobs.items()
                            if k != "ica_backend"},
            "seed":        int(args.seed),
            "stream":      args.stream,
            "force":       int(args.force),
            "dtype_out":   args.dtype_out,
            "ica_backend": knobs["ica_backend"],
        })

    all_diag: dict = {}
    n_workers = max(1, min(int(args.n_jobs), len(payloads)))

    if n_workers == 1 or len(payloads) == 1:
        # Sequential path. Also used for --modality {text,image} single-mod
        # runs so the extra process fork overhead is skipped.
        for pl in payloads:
            mod, diag = _process_modality_worker(pl)
            all_diag[mod] = diag
    else:
        # Process-level parallelism. ``spawn`` context is required on
        # Windows (default there) and safe on POSIX; forks would
        # otherwise duplicate CUDA state when cuML is in play.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx
        ) as pool:
            futs = {pool.submit(_process_modality_worker, pl): pl["mod"]
                    for pl in payloads}
            for fut in as_completed(futs):
                mod, diag = fut.result()
                all_diag[mod] = diag

    if args.log_json:
        log_path = Path(args.log_json)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "modalities":  list(mods),
            "seed":        args.seed,
            "n_jobs":      n_workers,
            "ica_backend_requested": {
                pl["mod"]: pl["ica_backend"] for pl in payloads
            },
            "streams":     all_diag,
        }
        with log_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[MACP] log JSON: {log_path}", flush=True)

    # Print in a stable order (text before image) regardless of the
    # order futures completed in.
    for mod in mods:
        diag = all_diag.get(mod, {})
        for name, d in diag.items():
            if not isinstance(d, dict):
                continue
            print(f"\n[MACP:{mod}] {name} stats:", flush=True)
            for k, v in d.items():
                print(f"       {k:>28} = {v}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
