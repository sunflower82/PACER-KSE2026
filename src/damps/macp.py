"""damps/macp.py -- Multi-Aspect Content Preprocessing fusion.

Ships Priority 6.0 (P6.0) and Priority 6.1 (P6.1). PACER-NRDMC treats
the text and image modalities as frozen features loaded from
``text_feat.npy`` / ``image_feat.npy``. This module reads the
pre-computed MACP streams (see ``scripts/preprocess_macp.py``) and
returns a fused tensor per modality that can drop-in replace
``text_feats`` / ``image_feats`` inside ``utility.load_data.Data``.

Design decisions
----------------
* **Symmetric text / image fusion.** P6.0 shipped ``fuse_text`` only,
  because P5.0 confirmed alpha_img<0 on Clothing under the P5.1 trunk
  (raw text). P6.0 changed that landscape: under ``replace_pca`` on
  text, alpha_img collapsed further to -0.84 -- consistent with the
  model routing gradient away from raw image once clean text became
  useful. P6.1 tests whether whitening image itself reverses that
  collapse. ``fuse_image`` is a bit-for-bit mirror of ``fuse_text``.

* **Same input dim.** Both PCA->ICA and ZCA streams are produced at
  the input dimension by the offline script, so fusion is a straight
  additive residual with no projection.

* **Std matching.** Whitening scales the streams differently (ZCA
  shrinks; ICA amplifies). Before mixing we rescale each auxiliary
  stream to match the raw stream's row-wise L2 mean, so the fusion
  weights `alpha_p, alpha_z` act as *relative* injection ratios
  rather than absolute magnitudes.

* **Flag-off equals baseline.** ``fuse_text(..., mode='raw')`` and
  ``fuse_image(..., mode='raw')`` return the input tensor unmodified
  so any bit-exact regression test against the P5 baseline still
  passes.

Modes
-----
raw
    No change. Used to disable MACP without removing the wiring.
replace_pca
    Replace the raw feats with the PCA->ICA stream.
replace_zca
    Replace the raw feats with the ZCA stream.
residual
    ``raw + alpha_p * scale_p * pca_ica + alpha_z * scale_z * zca``
    where ``scale_*`` matches the auxiliary stream's mean row L2 to the
    raw stream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# Filenames must stay in lockstep with scripts/preprocess_macp.py.
TEXT_PCA_ICA_FILE  = "text_feat_pca_ica.npy"
TEXT_ZCA_FILE      = "text_feat_zca.npy"
IMAGE_PCA_ICA_FILE = "image_feat_pca_ica.npy"
IMAGE_ZCA_FILE     = "image_feat_zca.npy"

# Backwards-compatible aliases (P6.0 shipped these two names).
PCA_ICA_FILE = TEXT_PCA_ICA_FILE
ZCA_FILE     = TEXT_ZCA_FILE

VALID_MODES = ("raw", "replace_pca", "replace_zca", "residual")


@dataclass(frozen=True)
class MacpConfig:
    """Immutable knobs consumed by :func:`fuse_text` and :func:`fuse_image`.

    ``mode == 'raw'`` short-circuits every branch below and returns the
    input tensor untouched, so the whole module is a no-op when the
    parser flag is off.
    """

    mode: str = "raw"
    alpha_p: float = 0.10        # PCA->ICA residual weight
    alpha_z: float = 0.10        # ZCA residual weight

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"MacpConfig.mode='{self.mode}' invalid; "
                f"expected one of {VALID_MODES}."
            )


def _load_stream(path: str) -> Optional[torch.Tensor]:
    """Load a MACP .npy stream or return ``None`` if absent."""
    if not os.path.isfile(path):
        return None
    arr = np.load(path)
    return torch.from_numpy(arr).float()


def _rescale_to(match: torch.Tensor, aux: torch.Tensor,
                *, eps: float = 1e-8) -> tuple[torch.Tensor, float]:
    """Rescale *aux* so its row-wise L2 mean matches *match*.

    Multiplicative scalar; preserves geometry, only fixes magnitude.
    """
    match_norm = match.norm(dim=1).mean().clamp_min(eps)
    aux_norm   = aux.norm(dim=1).mean().clamp_min(eps)
    scale = float(match_norm / aux_norm)
    return aux * scale, scale


def load_macp_streams(
    dataset_dir: str,
    modality: str = "text",
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (pca_ica_tensor, zca_tensor) for the requested modality.

    Both entries are ``None`` if the corresponding .npy files are
    missing. This lets callers gracefully fall back to raw features
    without failing the whole run.
    """
    if modality == "text":
        pca_name, zca_name = TEXT_PCA_ICA_FILE, TEXT_ZCA_FILE
    elif modality == "image":
        pca_name, zca_name = IMAGE_PCA_ICA_FILE, IMAGE_ZCA_FILE
    else:
        raise ValueError(
            f"load_macp_streams: unknown modality '{modality}' "
            f"(expected 'text' or 'image')."
        )
    pca_ica = _load_stream(os.path.join(dataset_dir, pca_name))
    zca     = _load_stream(os.path.join(dataset_dir, zca_name))
    return pca_ica, zca


def _fuse(
    raw: torch.Tensor,
    *,
    modality: str,
    dataset_dir: str,
    cfg: MacpConfig,
    verbose: bool,
) -> tuple[torch.Tensor, dict]:
    """Internal fusion kernel shared by fuse_text / fuse_image.

    Modality-aware only in the file names it loads and the diagnostic
    tag it prints; the maths is symmetric.
    """
    diag: dict = {"mode": cfg.mode, "modality": modality}
    if cfg.mode == "raw" or raw is None:
        diag["status"] = "bypass"
        if verbose:
            print(f"[MACP:{modality}] mode=raw -- feats unchanged.",
                  flush=True)
        return raw, diag

    pca_ica, zca = load_macp_streams(dataset_dir, modality=modality)
    diag["has_pca_ica"] = pca_ica is not None
    diag["has_zca"]     = zca is not None

    if modality == "text":
        pca_name, zca_name = TEXT_PCA_ICA_FILE, TEXT_ZCA_FILE
    else:
        pca_name, zca_name = IMAGE_PCA_ICA_FILE, IMAGE_ZCA_FILE

    if cfg.mode == "replace_pca":
        if pca_ica is None:
            raise FileNotFoundError(
                f"macp_mode=replace_pca requires {pca_name} in "
                f"{dataset_dir}. Run scripts/preprocess_macp.py "
                f"--modality {modality} first."
            )
        if pca_ica.shape != raw.shape:
            raise ValueError(
                f"[MACP:{modality}] PCA->ICA stream shape "
                f"{tuple(pca_ica.shape)} does not match raw "
                f"{tuple(raw.shape)}. Regenerate with the correct "
                f"{modality}_feat.npy."
            )
        fused = pca_ica
        diag["status"] = "replaced_with_pca_ica"

    elif cfg.mode == "replace_zca":
        if zca is None:
            raise FileNotFoundError(
                f"macp_mode=replace_zca requires {zca_name} in "
                f"{dataset_dir}. Run scripts/preprocess_macp.py "
                f"--modality {modality} first."
            )
        if zca.shape != raw.shape:
            raise ValueError(
                f"[MACP:{modality}] ZCA stream shape "
                f"{tuple(zca.shape)} does not match raw "
                f"{tuple(raw.shape)}."
            )
        fused = zca
        diag["status"] = "replaced_with_zca"

    elif cfg.mode == "residual":
        if pca_ica is None and zca is None:
            raise FileNotFoundError(
                f"macp_mode=residual requires at least one MACP "
                f"stream in {dataset_dir}. Run "
                f"scripts/preprocess_macp.py --modality {modality} first."
            )
        fused = raw.clone()
        if pca_ica is not None and cfg.alpha_p != 0.0:
            if pca_ica.shape != raw.shape:
                raise ValueError(
                    f"[MACP:{modality}] PCA->ICA stream shape "
                    f"{tuple(pca_ica.shape)} does not match raw "
                    f"{tuple(raw.shape)}."
                )
            pca_ica_scaled, scale_p = _rescale_to(raw, pca_ica)
            fused = fused + float(cfg.alpha_p) * pca_ica_scaled
            diag["scale_p"] = scale_p
            diag["alpha_p"] = cfg.alpha_p
        if zca is not None and cfg.alpha_z != 0.0:
            if zca.shape != raw.shape:
                raise ValueError(
                    f"[MACP:{modality}] ZCA stream shape "
                    f"{tuple(zca.shape)} does not match raw "
                    f"{tuple(raw.shape)}."
                )
            zca_scaled, scale_z = _rescale_to(raw, zca)
            fused = fused + float(cfg.alpha_z) * zca_scaled
            diag["scale_z"] = scale_z
            diag["alpha_z"] = cfg.alpha_z
        diag["status"] = "residual"

    else:  # pragma: no cover -- guarded by MacpConfig.__post_init__
        raise ValueError(f"Unhandled MACP mode: {cfg.mode}")

    diag["mean_l2_raw"]   = float(raw.norm(dim=1).mean())
    diag["mean_l2_fused"] = float(fused.norm(dim=1).mean())
    if verbose:
        print(
            f"[MACP:{modality}] mode={cfg.mode}  alpha_p={cfg.alpha_p}  "
            f"alpha_z={cfg.alpha_z}  "
            f"|raw|={diag['mean_l2_raw']:.3f}  "
            f"|fused|={diag['mean_l2_fused']:.3f}",
            flush=True,
        )
    return fused, diag


def fuse_text(
    text_raw: torch.Tensor,
    *,
    dataset_dir: str,
    cfg: MacpConfig,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Return (fused_text, diagnostics).

    Parameters
    ----------
    text_raw : (N, D) float tensor
        The tensor originally produced by ``_load_modality('text')``.
    dataset_dir : str
        Directory that holds ``text_feat.npy`` and (optionally) the
        MACP text streams. Same directory the loader passed to
        ``np.load``.
    cfg : MacpConfig
        Which mode + weights to use. ``mode='raw'`` bypasses I/O.
    verbose : bool
        If True, prints a one-line summary. Wired to
        ``args.macp_verbose`` in the loader.
    """
    return _fuse(
        text_raw, modality="text",
        dataset_dir=dataset_dir, cfg=cfg, verbose=verbose,
    )


def fuse_image(
    image_raw: torch.Tensor,
    *,
    dataset_dir: str,
    cfg: MacpConfig,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Return (fused_image, diagnostics). Mirror of :func:`fuse_text`.

    Parameters
    ----------
    image_raw : (N, D) float tensor
        The tensor originally produced by ``_load_modality('image')``.
    dataset_dir : str
        Directory that holds ``image_feat.npy`` and (optionally) the
        MACP image streams.
    cfg : MacpConfig
        Which mode + weights to use. ``mode='raw'`` bypasses I/O.
    verbose : bool
    """
    return _fuse(
        image_raw, modality="image",
        dataset_dir=dataset_dir, cfg=cfg, verbose=verbose,
    )
