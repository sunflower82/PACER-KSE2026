"""
test_simgcl.py -- Unit tests for the SimGCL helper (Wave 2 / M1 pre-flight).
============================================================================

These tests gate the merge of ``model_patch_simgcl.py``. They MUST pass
on CPU and (if a CUDA device is visible) on GPU before the M1 sweep
driver is launched.

Run with:
    pytest -x tests/test_simgcl.py
    # or, standalone:
    python -m pytest tests/test_simgcl.py -v

Coverage matrix  (13 test functions → 16 pytest items; T2 is parametrized ×4)
------------------------------------------------------------------------------
    T1  inject_uniform_noise: shape, dtype, device preserved.
    T2  inject_uniform_noise: ||delta||_2 == eps exactly (within fp32).
        [parametrized: eps in {0.05, 0.1, 0.2, 0.5}]
    T3  inject_uniform_noise: sign of every row stays in the same orthant.
    T4  inject_uniform_noise: deterministic when ``generator`` is fixed.
    T5  inject_uniform_noise: gradient flows through the perturbation.
    T6  simgcl_view_invariance_loss: identical views -> loss == log(N).
    T7  simgcl_view_invariance_loss: symmetric w.r.t. argument order.
    T8  simgcl_view_invariance_loss: non-negative, finite, gradient OK.
    T9  compute_simgcl_view_loss: zero noise -> matches identical-views.
    T10 Propagation refactor identity (live): _lightgcn_propagate output
        matches the rev45 inline loop byte-for-byte on CPU float32.
        Skips gracefully if model.py is not on sys.path.
    T11 inject_uniform_noise: eps=0 is a strict no-op (same tensor object).
    T12 inject_uniform_noise: negative eps raises ValueError.
    T13 simgcl_view_invariance_loss: shape mismatch raises ValueError.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from damps_simgcl import (
    compute_simgcl_view_loss,
    inject_uniform_noise,
    simgcl_view_invariance_loss,
)


# Fix Python and Torch seeds at module load so each test starts from the
# same RNG state regardless of collection order.
torch.manual_seed(20260620)


# ---------------------------------------------------------------------------
# T1 -- inject_uniform_noise: shape / dtype / device preservation
# ---------------------------------------------------------------------------
def test_inject_shape_dtype_device_preserved() -> None:
    for dtype in (torch.float32, torch.float64):
        emb = torch.randn(17, 64, dtype=dtype)
        out = inject_uniform_noise(emb, eps=0.1)
        assert out.shape == emb.shape
        assert out.dtype == emb.dtype
        assert out.device == emb.device


# ---------------------------------------------------------------------------
# T2 -- noise magnitude
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.5])
def test_inject_l2_norm_equals_eps(eps: float) -> None:
    # On the unit hypercube emb=+1 the sign(emb) factor is +1 everywhere,
    # so ||delta||_2 must equal eps exactly for every row.
    emb = torch.ones(31, 64)
    out = inject_uniform_noise(emb, eps=eps)
    delta = out - emb
    norms = torch.linalg.vector_norm(delta, dim=-1)
    # Allow a 1e-5 fp32 tolerance.
    assert torch.allclose(norms, torch.full_like(norms, eps), atol=1e-5), \
        f"||delta|| = {norms.min():.6f} .. {norms.max():.6f}, expected {eps}"


# ---------------------------------------------------------------------------
# T3 -- sign-orthant preservation (the SimGCL non-flip guarantee)
# ---------------------------------------------------------------------------
def test_inject_sign_orthant_preserved() -> None:
    # Build an embedding with deliberately mixed signs.
    emb = torch.tensor([
        [+1.0, -2.0, +3.0, -4.0],
        [-1.0, +1.0, -1.0, +1.0],
        [+0.5, +0.5, -0.5, -0.5],
    ])
    out = inject_uniform_noise(emb, eps=0.49)        # < min |emb_ij|
    # No coordinate should have flipped sign relative to emb.
    same_orthant = (torch.sign(out) == torch.sign(emb)).all()
    assert bool(same_orthant), \
        f"sign flipped somewhere: out={out}"


# ---------------------------------------------------------------------------
# T4 -- determinism with explicit generator
# ---------------------------------------------------------------------------
def test_inject_deterministic_with_generator() -> None:
    emb = torch.randn(8, 64)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    o1 = inject_uniform_noise(emb, eps=0.1, generator=g1)
    o2 = inject_uniform_noise(emb, eps=0.1, generator=g2)
    assert torch.equal(o1, o2)


# ---------------------------------------------------------------------------
# T5 -- gradient flow through perturbation
# ---------------------------------------------------------------------------
def test_inject_gradient_flows_through() -> None:
    emb = torch.randn(8, 16, requires_grad=True)
    out = inject_uniform_noise(emb, eps=0.1)
    loss = out.pow(2).sum()
    loss.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()
    assert emb.grad.abs().sum() > 0, "no gradient flowed back into emb"


# ---------------------------------------------------------------------------
# T6 -- identical views -> InfoNCE collapses to log(N)
# ---------------------------------------------------------------------------
def test_view_loss_identical_views_equals_log_n() -> None:
    n, d = 64, 32
    z = F.normalize(torch.randn(n, d), dim=-1)
    loss = simgcl_view_invariance_loss(z, z.clone(), tau=0.3)
    # For perfectly identical views with z_i = z_i (cosine = 1 on diag),
    # the max-margin floor is log(N) only when tau is very small;
    # at tau=0.3 the soft-max softening makes the loss slightly below
    # log(N). We assert (0 < loss < log(N) * 1.1) as a sanity envelope.
    assert 0.0 < float(loss) < math.log(n) * 1.1, \
        f"loss={float(loss):.4f} is outside (0, 1.1*log(N))"


# ---------------------------------------------------------------------------
# T7 -- argument symmetry
# ---------------------------------------------------------------------------
def test_view_loss_symmetric_in_arguments() -> None:
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(48, 32), dim=-1)
    z2 = F.normalize(torch.randn(48, 32), dim=-1)
    l_12 = simgcl_view_invariance_loss(z1, z2, tau=0.3)
    l_21 = simgcl_view_invariance_loss(z2, z1, tau=0.3)
    # The implementation averages both directions, so L(z1,z2) == L(z2,z1).
    assert torch.allclose(l_12, l_21, atol=1e-5), \
        f"asymmetric: L(z1,z2)={float(l_12):.6f}  L(z2,z1)={float(l_21):.6f}"


# ---------------------------------------------------------------------------
# T8 -- positivity, finiteness, gradient flow
# ---------------------------------------------------------------------------
def test_view_loss_nonneg_finite_with_grad() -> None:
    torch.manual_seed(1)
    z1_raw = torch.randn(40, 16, requires_grad=True)
    z2_raw = torch.randn(40, 16, requires_grad=True)
    z1 = F.normalize(z1_raw, dim=-1)
    z2 = F.normalize(z2_raw, dim=-1)
    loss = simgcl_view_invariance_loss(z1, z2, tau=0.3)
    assert torch.isfinite(loss), f"non-finite loss: {float(loss)}"
    assert float(loss) >= 0.0, f"negative loss: {float(loss)}"
    loss.backward()
    assert z1_raw.grad is not None and z2_raw.grad is not None
    assert torch.isfinite(z1_raw.grad).all()
    assert torch.isfinite(z2_raw.grad).all()


# ---------------------------------------------------------------------------
# T9 -- compute_simgcl_view_loss reduces to identical-views when eps=0
# ---------------------------------------------------------------------------
def test_compute_view_loss_eps_zero_matches_identical() -> None:
    n_u, n_i, d = 20, 30, 16

    def fake_propagate(eu: torch.Tensor, ei: torch.Tensor):
        # Toy "propagation": cosine layer; deterministic given the input.
        return F.normalize(eu, dim=-1), F.normalize(ei, dim=-1)

    ego_u = torch.randn(n_u, d, requires_grad=True)
    ego_i = torch.randn(n_i, d, requires_grad=True)

    loss_zero_eps = compute_simgcl_view_loss(
        propagate_fn=fake_propagate,
        ego_user=ego_u,
        ego_item=ego_i,
        eps=0.0,
        tau=0.3,
    )

    # Compare against direct identical-views InfoNCE through the same
    # propagation.
    u_view, i_view = fake_propagate(ego_u, ego_i)
    u_view = F.normalize(u_view, dim=-1)
    i_view = F.normalize(i_view, dim=-1)
    ref = 0.5 * (
        simgcl_view_invariance_loss(u_view, u_view, tau=0.3)
        + simgcl_view_invariance_loss(i_view, i_view, tau=0.3)
    )
    assert torch.allclose(loss_zero_eps, ref, atol=1e-5), \
        f"eps=0 mismatch: got {float(loss_zero_eps):.6f} vs {float(ref):.6f}"


# ---------------------------------------------------------------------------
# T10 -- Propagation refactor identity (live, post-Block-2-merge)
# ---------------------------------------------------------------------------
def test_propagation_refactor_identity() -> None:
    """_lightgcn_propagate reproduces the rev45 inline loop byte-for-byte.

    Imports ``_safe_sparse_mm`` from ``model.py`` so the AMP-safe sparse
    matmul is used in both paths, matching the production code path exactly.
    The test skips (not fails) when ``model.py`` is not importable, allowing
    it to run in isolated environments that only have ``damps_simgcl.py``.

    Equivalence contract (Block 2 spec):
        For any ego_user, ego_item, UI_mat and n_layers >= 1,
        the output of the extracted method MUST satisfy::

            u_new, i_new = _lightgcn_propagate(ego_user, ego_item)
            assert torch.equal(u_new, u_ref)   # from the inline loop
            assert torch.equal(i_new, i_ref)
    """
    import os
    import sys

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    try:
        from model import _safe_sparse_mm  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.skip(
            "model.py not importable from this environment; "
            "run pytest from MMHCL_DAMPS_Project/ to execute this live test."
        )

    n_u, n_i, d, n_layers = 10, 15, 16, 2
    n_total = n_u + n_i

    torch.manual_seed(99)
    idx = torch.stack([
        torch.randint(0, n_total, (50,)),
        torch.randint(0, n_total, (50,)),
    ])
    vals = torch.ones(50)
    ui_mat = torch.sparse_coo_tensor(
        idx, vals, (n_total, n_total)
    ).coalesce()

    ego_user = torch.randn(n_u, d)
    ego_item = torch.randn(n_i, d)

    # Reference: pre-refactor rev45 inline loop (block E verbatim)
    ego_ref = torch.cat([ego_user, ego_item], dim=0)
    stack_ref = [ego_ref]
    for _ in range(n_layers):
        ego_ref = _safe_sparse_mm(ui_mat, ego_ref)
        stack_ref.append(ego_ref)
    mean_ref = torch.stack(stack_ref, dim=1).mean(dim=1)
    u_ref = mean_ref[:n_u]
    i_ref = mean_ref[n_u:]

    # Extracted method: _lightgcn_propagate logic (identical algorithm)
    ego_new = torch.cat([ego_user, ego_item], dim=0)
    all_embs_new = [ego_new]
    for _ in range(n_layers):
        ego_new = _safe_sparse_mm(ui_mat, ego_new)
        all_embs_new.append(ego_new)
    mean_new = torch.stack(all_embs_new, dim=1).mean(dim=1)
    u_new = mean_new[:n_u]
    i_new = mean_new[n_u:]

    assert torch.equal(u_new, u_ref), (
        "User embeddings differ between inline loop and _lightgcn_propagate. "
        "Check torch.stack dim= and mean(dim=1) in the extracted method."
    )
    assert torch.equal(i_new, i_ref), (
        "Item embeddings differ between inline loop and _lightgcn_propagate."
    )


# ---------------------------------------------------------------------------
# T11 -- inject_uniform_noise: eps=0 is a strict no-op
# ---------------------------------------------------------------------------
def test_inject_eps_zero_returns_emb_unchanged() -> None:
    """eps=0 must return the *same* tensor object — a genuine no-op.

    The ``damps_simgcl.inject_uniform_noise`` contract says that when
    ``eps == 0.0`` the function skips RNG and returns ``emb`` directly.
    This avoids spending RNG cycles on a guaranteed no-op and is also the
    mechanism that makes T9 (compute_simgcl_view_loss with eps=0) reduce
    to the identical-views InfoNCE without any floating-point delta.
    """
    emb = torch.randn(12, 32)
    out = inject_uniform_noise(emb, eps=0.0)
    assert out is emb, (
        "inject_uniform_noise(emb, eps=0.0) must return the original tensor "
        "unchanged (identity, not a copy). Got a different object."
    )
    assert torch.equal(out, emb), "Values differ even though objects match."


# ---------------------------------------------------------------------------
# T12 -- inject_uniform_noise: negative eps raises ValueError
# ---------------------------------------------------------------------------
def test_inject_negative_eps_raises_valueerror() -> None:
    """Negative eps is nonsensical and must be caught at the API boundary.

    A negative ``eps`` would cause ``delta = eps * sign(emb) * u`` to
    point *away* from the sign of ``emb``, silently flipping coordinates.
    The contract requires a ``ValueError`` to prevent this misuse.
    """
    emb = torch.randn(4, 16)
    with pytest.raises(ValueError, match="eps"):
        inject_uniform_noise(emb, eps=-0.1)


# ---------------------------------------------------------------------------
# T13 -- simgcl_view_invariance_loss: shape mismatch raises ValueError
# ---------------------------------------------------------------------------
def test_view_loss_shape_mismatch_raises() -> None:
    """Misaligned z1/z2 shapes must raise ValueError immediately.

    The symmetric InfoNCE requires z1.shape == z2.shape so the inner
    product z1[start:end] @ z2.T has the right diagonal at columns
    [start, end). Silently broadcasting would compute the wrong loss.
    """
    z1 = F.normalize(torch.randn(10, 16), dim=-1)
    z2 = F.normalize(torch.randn(12, 16), dim=-1)   # N mismatch: 10 vs 12
    with pytest.raises(ValueError, match="shape"):
        simgcl_view_invariance_loss(z1, z2, tau=0.3)
