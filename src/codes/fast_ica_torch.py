"""codes/fast_ica_torch.py -- Windows-native GPU FastICA via PyTorch.

Motivation (P6.1b)
------------------
`preprocess_macp.py` already exposes ``--ica_backend cuml`` for RAPIDS,
but RAPIDS/cuML wheels are Linux-only -- on the user's Windows 11 IoT
Enterprise + RTX 5090 host the ``cuml`` import always fails and the
backend silently falls back to sklearn (CPU). This module provides a
Windows-friendly GPU FastICA implementation that only depends on
``torch`` (which ships CUDA wheels for Windows natively).

Design goals
------------
* Mirror ``sklearn.decomposition.FastICA(whiten='unit-variance',
  fun='logcosh')`` closely enough that the downstream loadings are
  qualitatively identical for the D=4096 image features (the small
  numerical differences are within the ~1e-4 tolerance FastICA already
  accepts across different `random_state` values in sklearn).
* No cuML / RAPIDS / TF dependency. Only ``torch``.
* Deterministic under a fixed seed on a fixed GPU. Cross-device
  (CPU vs GPU) determinism is NOT promised -- same as sklearn cuML.
* Falls back to CPU if CUDA is unavailable but caller still asked
  for ``torch``.

References
----------
Hyvarinen & Oja (2000), Independent Component Analysis: Algorithms and
Applications. Neural Networks 13(4-5):411-430. We implement the
"parallel" fixed-point algorithm with symmetric decorrelation
(equation 45).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _sym_decorrelate(w: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
    """Symmetric decorrelation: W <- (W W^T)^{-1/2} W.

    Uses ``torch.linalg.eigh`` on the symmetric matrix ``W W^T``.
    """
    import torch

    ww = w @ w.T
    # ``eigh`` returns ascending eigenvalues + eigenvectors. Clamp for
    # numerical safety (matrix can be positive-semidefinite due to fp32).
    s, u = torch.linalg.eigh(ww)
    s = torch.clamp(s, min=1e-12)
    inv_sqrt = u @ torch.diag(s.rsqrt()) @ u.T
    return inv_sqrt @ w


def _pca_whiten_torch(
    x: "torch.Tensor", n_components: int  # noqa: F821
) -> tuple["torch.Tensor", "torch.Tensor"]:  # noqa: F821
    """Return (X_white, K) with ``X_white = (X - mean) @ K.T``.

    Uses ``torch.linalg.svd`` on the centred matrix which is the same
    numerical route sklearn's FastICA takes internally (``svd_solver =
    'randomized' if n < d else 'full'``). We use `full` for stability.
    """
    import torch

    n, _ = x.shape
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    # Economy SVD -- returns U (n, min(n,d)), S (min(n,d),), Vh (min, d).
    u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    # Whitening rows of Vh have unit variance after scaling by
    # sqrt(n - 1) / s. Take the top n_components rows.
    k = int(n_components)
    scale = np.sqrt(max(1, n - 1))
    K = (vh[:k] / s[:k, None]) * scale  # (k, d)
    x_white = xc @ K.T                  # (n, k) -- unit variance columns
    return x_white, K


def fast_ica_torch(
    x_np: np.ndarray,
    *,
    n_components: int,
    max_iter: int = 500,
    tol: float = 5e-4,
    seed: int = 42,
    device: str | None = None,
    dtype: str = "float32",
    fun: str = "logcosh",
    alpha: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """PCA-whitening + parallel FastICA on ``x_np``.

    Parameters
    ----------
    x_np : (N, D) numpy array
    n_components : int
        Number of independent components to extract. Must satisfy
        ``1 <= n_components <= min(N, D)``.
    max_iter, tol : FastICA convergence knobs (matches sklearn).
    seed : Torch RNG seed.
    device : ``'cuda:0'``, ``'cpu'``, or ``None`` (auto-detect).
    dtype : ``'float32'`` (default, ~2x faster on GPU) or ``'float64'``.
    fun : Non-linearity. ``'logcosh'`` (default) or ``'exp'`` -- matches
        sklearn's ``fun`` argument.
    alpha : logcosh slope parameter (sklearn default 1.0).

    Returns
    -------
    y_np : (N, n_components) numpy array (float64 for parity with
        sklearn ``fit_transform`` output).
    stats : dict with n_iter, converged, device, dtype.
    """
    import torch

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64

    torch.manual_seed(int(seed))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    x = torch.as_tensor(x_np, dtype=torch_dtype, device=device)
    n, d = x.shape
    k = int(n_components)
    if not (1 <= k <= min(n, d)):
        raise ValueError(
            f"n_components={k} out of range for input shape {(n, d)}"
        )

    # ------------------------------------------------------------------ #
    #  Step 1: PCA whitening (centres and rescales columns to unit var).
    # ------------------------------------------------------------------ #
    x_white, _K = _pca_whiten_torch(x, n_components=k)   # (n, k)
    # sklearn's parallel FastICA operates on (k, n) internally.
    xw = x_white.T.contiguous()                          # (k, n)

    # ------------------------------------------------------------------ #
    #  Step 2: Random init + symmetric decorrelation.
    # ------------------------------------------------------------------ #
    w_init = torch.randn(k, k, dtype=torch_dtype, device=device)
    w = _sym_decorrelate(w_init)                         # (k, k)

    # ------------------------------------------------------------------ #
    #  Step 3: Parallel fixed-point iteration.
    # ------------------------------------------------------------------ #
    n_iter = 0
    converged = False
    n_samples_inv = 1.0 / float(n)

    for it in range(int(max_iter)):
        # (k, n)
        wx = w @ xw

        if fun == "logcosh":
            gwx = torch.tanh(alpha * wx)
            # g'(u) = alpha * (1 - tanh^2)
            g_prime = alpha * (1.0 - gwx * gwx)          # (k, n)
        elif fun == "exp":
            wx2 = wx * wx
            exp_neg = torch.exp(-0.5 * wx2)
            gwx = wx * exp_neg
            g_prime = (1.0 - wx2) * exp_neg
        else:
            raise ValueError(f"Unknown fun: {fun!r}")

        # Parallel update (Eq. 45 of Hyvarinen 1999):
        #   W_new = (1/n) * G(W X) X^T - diag(mean(G'(W X))) * W
        w1 = (gwx @ xw.T) * n_samples_inv \
            - g_prime.mean(dim=1, keepdim=True) * w      # (k, k)
        w_new = _sym_decorrelate(w1)

        # Convergence: max |diag(W_new W^T) - 1|  (sklearn diagnostic).
        with torch.no_grad():
            lim = (torch.abs(torch.abs(
                (w_new * w).sum(dim=1)
            ) - 1.0)).max().item()

        w = w_new
        n_iter = it + 1
        if lim < tol:
            converged = True
            break

    # ------------------------------------------------------------------ #
    #  Step 4: Compose ICA output: sources S = W X_white.T -> back to
    #  (n_samples, n_components) which is what sklearn.fit_transform
    #  returns.
    # ------------------------------------------------------------------ #
    s = (w @ xw).T                                       # (n, k)

    # Return in float64 for downstream compatibility with sklearn path.
    y = s.detach().to(torch.float64).cpu().numpy()

    stats: dict[str, Any] = {
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "device": str(device),
        "dtype": str(dtype),
        "fun": fun,
    }
    return y, stats


def is_torch_available() -> bool:
    """Return True iff ``torch`` imports cleanly."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def is_torch_cuda_available() -> bool:
    """Return True iff ``torch`` imports AND has a working CUDA device."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
