"""tests/test_asc_reparam.py -- rev57 P4 ASC gate reparameterization.

Unit tests for the four ``asc_gate_mode`` variants introduced in
``model.py`` (see rev57 P4 patch). These are model-only tests; they do
not touch train.py or the driver.

Each test builds a minimal DAMPS_MMHCL model with random tiny features,
overrides ``self.alpha_img`` after construction, and asserts:

  * The effective alpha lands in the expected numeric range.
  * At the paper-locked ``theta_init``, the effective alpha equals 0.1
    to five decimal places -- regardless of mode.
  * ``_soft_route`` reproduces ``h_raw + effective_alpha * ln(h_cal)``.
  * The raw / effective values both surface in ``diagnostics()``.

Run with::

    pytest tests/test_asc_reparam.py -v

or ad-hoc from the workspace root::

    python -m pytest MMHCL_DAMPS_Project/tests/test_asc_reparam.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from model import DAMPS_MMHCL


def _tiny_model(mode: str) -> DAMPS_MMHCL:
    """Build a tiny DAMPS_MMHCL for isolated ASC-gate testing."""
    n_users, n_items, d = 32, 48, 8
    torch.manual_seed(0)
    image_feats = torch.randn(n_items, 16)
    text_feats = torch.randn(n_items, 16)
    return DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        audio_feats=None,
        ui_layers=2,
        user_layers=2,
        item_layers=2,
        temperature_init=0.3,
        learnable_tau=False,
        warmup_epochs=1,
        damps_num_categories=3,
        data_driven_prior=True,
        enable_logq=False,
        enable_simgcl=False,
        enable_nrdmc_lite=False,
        enable_ptv=False,
        asc_gate_mode=mode,
    )


# ---------------------------------------------------------------------------
# Test 1 -- init calibration: alpha(theta_init) == 0.1 in every mode.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode",
    ["raw", "sigmoid", "tanh_signed", "tanh01"],
)
def test_alpha_init_calibration(mode: str) -> None:
    m = _tiny_model(mode)
    with torch.no_grad():
        eff = m._alpha_effective(m.alpha_img).item()
    assert math.isclose(eff, 0.1, abs_tol=1e-5), (
        f"asc_gate_mode={mode}: alpha_effective at init = {eff}, "
        "expected 0.1"
    )


# ---------------------------------------------------------------------------
# Test 2 -- range constraints for each mode after clamping theta.
# ---------------------------------------------------------------------------
def test_range_sigmoid() -> None:
    m = _tiny_model("sigmoid")
    with torch.no_grad():
        for theta in [-10.0, -3.0, 0.0, 3.0, 10.0]:
            m.alpha_img.data.fill_(theta)
            eff = m._alpha_effective(m.alpha_img).item()
            assert 0.0 < eff < 1.0, (theta, eff)
        # extreme negatives -> ~0, extreme positives -> ~1
        m.alpha_img.data.fill_(-100.0)
        assert m._alpha_effective(m.alpha_img).item() < 1e-6
        m.alpha_img.data.fill_(100.0)
        assert 1.0 - m._alpha_effective(m.alpha_img).item() < 1e-6


def test_range_tanh_signed() -> None:
    m = _tiny_model("tanh_signed")
    with torch.no_grad():
        for theta in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            m.alpha_img.data.fill_(theta)
            eff = m._alpha_effective(m.alpha_img).item()
            assert -1.0 <= eff <= 1.0, (theta, eff)


def test_range_tanh01() -> None:
    m = _tiny_model("tanh01")
    with torch.no_grad():
        for theta in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            m.alpha_img.data.fill_(theta)
            eff = m._alpha_effective(m.alpha_img).item()
            assert 0.0 <= eff <= 1.0, (theta, eff)


def test_raw_is_identity() -> None:
    m = _tiny_model("raw")
    with torch.no_grad():
        for theta in [-2.0, -0.1, 0.0, 0.1, 2.0]:
            m.alpha_img.data.fill_(theta)
            eff = m._alpha_effective(m.alpha_img).item()
            assert math.isclose(eff, theta, abs_tol=1e-7)


# ---------------------------------------------------------------------------
# Test 3 -- reject unknown modes.
# ---------------------------------------------------------------------------
def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="asc_gate_mode"):
        _tiny_model("bogus")


# ---------------------------------------------------------------------------
# Test 4 -- diagnostics() surfaces both raw theta and effective alpha.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode,theta,expected_eff",
    [
        ("raw", 0.5, 0.5),
        ("sigmoid", 0.0, 0.5),
        ("tanh_signed", 0.0, 0.0),
        ("tanh01", 0.0, 0.5),
    ],
)
def test_diagnostics_reports_both(
    mode: str, theta: float, expected_eff: float
) -> None:
    m = _tiny_model(mode)
    with torch.no_grad():
        m.alpha_img.data.fill_(theta)
        m.alpha_txt.data.fill_(theta)
    diag = m.diagnostics()
    assert diag["asc_gate_mode"] == mode
    assert math.isclose(diag["alpha_img_theta"], theta, abs_tol=1e-6)
    assert math.isclose(diag["alpha_img"], expected_eff, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 5 -- _soft_route reproduces h_raw + effective_alpha * ln(h_cal).
# ---------------------------------------------------------------------------
def test_soft_route_uses_effective_alpha() -> None:
    m = _tiny_model("sigmoid")
    # Force effective alpha to a known value: sigmoid(0)=0.5.
    with torch.no_grad():
        m.alpha_img.data.fill_(0.0)
    ln = nn.LayerNorm(8)
    h_raw = torch.randn(4, 8)
    h_cal = torch.randn(4, 8)
    out = m._soft_route(h_raw, h_cal, ln, m.alpha_img)
    expected = h_raw + 0.5 * ln(h_cal)
    assert torch.allclose(out, expected, atol=1e-6), (out - expected).abs().max()


# ---------------------------------------------------------------------------
# Test 6 -- backward-compat: raw mode reproduces pre-P4 numeric behaviour.
# ---------------------------------------------------------------------------
def test_raw_mode_backward_compat() -> None:
    m_raw = _tiny_model("raw")
    # Pre-P4: alpha was nn.Parameter(torch.tensor(0.1)) so alpha_effective
    # equalled 0.1 at init. Post-P4 raw mode must reproduce this exactly.
    assert math.isclose(m_raw.alpha_img.item(), 0.1, abs_tol=1e-6)
    assert math.isclose(
        m_raw._alpha_effective(m_raw.alpha_img).item(), 0.1, abs_tol=1e-6
    )
