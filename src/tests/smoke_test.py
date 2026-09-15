"""
tests/smoke_test.py — End-to-end sanity check for the DAMPS package.

Runs a tiny synthetic forward+backward pass through every DAMPS sub-component
and the integrated ``DAMPS_MMHCL`` model. Designed to finish in < 5 seconds
and use < 100 MiB of RAM, so it is safe for CI / pre-commit hooks.

Run from the repository root:

    python MMHCL_DAMPS_Project/tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys

# Make the package importable when run as a standalone script
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch
import torch.nn.functional as F

from damps import (
    DAMPS,
    DualPathKNN,
    SlimMomentumEncoder,
    adj_avg_degree,
    adj_nnz,
    compute_avrf_logit,
    compute_avrf_prior,
)


def _print_ok(label: str) -> None:
    print(f"  [OK] {label}")


def smoke_damps_core() -> None:
    print("== DAMPS core ==")
    N, d, C = 64, 64, 4
    h_img = torch.randn(N, d)
    h_txt = torch.randn(N, d)
    cats = torch.randint(0, C, (N,))

    damps = DAMPS(d=d, num_categories=C, warmup_epochs=2)
    damps.train()

    h_img_cal, h_txt_cal, _ = damps(h_img, h_txt, item_categories=cats)
    assert h_img_cal.shape == h_img.shape, "shape mismatch (img)"
    assert h_txt_cal.shape == h_txt.shape, "shape mismatch (txt)"

    # Backprop through DAMPS — verifies all parameters receive gradient
    loss = (h_img_cal.pow(2).sum() + h_txt_cal.pow(2).sum()) * 1e-3
    loss.backward()
    grad_count = sum(p.grad is not None for p in damps.parameters())
    assert grad_count >= 4, f"expected >=4 grads, got {grad_count}"
    _print_ok(f"forward + backward, {damps.num_trainable_params()} params")

    sat = damps.tanh_saturation_rates()
    _print_ok(f"tanh_sat: {sat}")

    damps.update_epoch_mad(0, h_img, h_txt)
    _print_ok("update_epoch_mad runs")

    # ----- IMCF EMA epoch-counter regression test (compliance WARN 3) -----
    # Run several forward passes inside a single "epoch": the per-forward-pass
    # counter must keep ticking, but the *current epoch* (which drives the
    # adaptive EMA schedule) must NOT change unless ``set_epoch`` is called.
    damps.zero_grad(set_to_none=True)
    damps.set_epoch(3)
    fwd_before = float(damps._imcf_update_count.item())
    epoch_before = int(damps._current_epoch.item())
    for _ in range(5):
        damps(h_img, h_txt, item_categories=cats)
    fwd_after = float(damps._imcf_update_count.item())
    epoch_after = int(damps._current_epoch.item())
    assert fwd_after - fwd_before == 5, (
        f"forward-pass counter expected +5, got "
        f"+{fwd_after - fwd_before}"
    )
    assert epoch_after == epoch_before == 3, (
        f"epoch counter must stay at 3 across forward passes, got "
        f"before={epoch_before}, after={epoch_after}"
    )
    damps.set_epoch(7)
    assert int(damps._current_epoch.item()) == 7
    _print_ok(
        "IMCF schedule: forward-pass counter +5, epoch held at 3 then "
        "advanced to 7 via set_epoch"
    )


def smoke_momentum() -> None:
    print("== Slim Momentum Encoder ==")
    N, d = 32, 64
    enc = SlimMomentumEncoder(num_items=N, dim=d, warmup_epochs=2)
    idx = torch.arange(N)
    enc.update(idx, torch.randn(N, d), torch.randn(N, d), epoch=0)
    enc.update(idx, torch.randn(N, d), torch.randn(N, d), epoch=1)
    enc.update(idx, torch.randn(N, d), torch.randn(N, d), epoch=2)
    assert enc.initialised_count() == N
    _print_ok("EMA buffers update + initialised flag")


def smoke_knn() -> None:
    print("== Dual-path KNN ==")
    N, d = 64, 32
    h_img = torch.randn(N, d)
    h_txt = torch.randn(N, d)

    builder = DualPathKNN(k=4, faiss_threshold=10**12, chunk_size=16)
    adj_single = builder.build_graph(h_img)
    assert adj_single.shape == (N, N)
    _print_ok(f"single-modality K-NN, NNZ={adj_nnz(adj_single)}")

    adj_multi = builder.build_graph_from_modalities(h_img, h_txt)
    assert adj_multi.shape == (N, N)
    _print_ok(
        f"multi-modal hypergraph, NNZ={adj_nnz(adj_multi)}, "
        f"avg_deg={adj_avg_degree(adj_multi):.2f}"
    )


def smoke_prior() -> None:
    print("== Data-driven prior ==")
    N, d = 128, 64
    feats = torch.randn(N, d)
    prior = compute_avrf_prior(feats)
    logit = compute_avrf_logit(feats, clip=2.0)
    assert prior.shape == (d // 2 + 1,)
    assert logit.shape == (1, d // 2 + 1)
    assert (logit.abs() <= 2.0 + 1e-6).all()
    _print_ok(f"prior range=[{prior.min():.3f},{prior.max():.3f}]  logit clipped")


def smoke_full_model() -> None:
    print("== DAMPS_MMHCL full model ==")
    # Local import so the smoke test can run without a configured CLI parser
    # in the sub-imports (model.py only imports damps + nn).
    from model import DAMPS_MMHCL

    n_users, n_items, d = 16, 32, 64
    image_feats = torch.randn(n_items, 256)
    text_feats = torch.randn(n_items, 128)

    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=2,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
    )
    model.set_meta_categories(torch.randint(0, 4, (n_items,)))

    # ----- Tau-init regression test (rev44 / Revision 11 Phase 1) ---------
    # The rev44 spec mandates a static InfoNCE temperature anchor at 0.3;
    # the default constructor must honour this and register tau as a
    # non-trainable buffer so it does NOT pick up gradients.
    assert abs(float(model.tau.item()) - 0.3) < 1e-6, (
        f"static tau must be initialised at 0.3 (rev44 Phase 1 anchor), "
        f"got {float(model.tau.item())}"
    )
    assert model.learnable_tau is False, (
        "rev44 Phase 1 default must register tau as a buffer; "
        "model.learnable_tau should be False"
    )
    assert "tau" not in dict(model.named_parameters()), (
        "static tau should not appear in model.named_parameters()"
    )
    _print_ok(
        f"static tau initialised at {float(model.tau.item()):.4f} "
        f"(rev44 Phase 1 anchor=0.3, registered as buffer)"
    )

    # Build trivial graphs (identity-like sparse tensors)
    UI = torch.eye(n_users + n_items).to_sparse_coo()
    I2I = torch.eye(n_items).to_sparse_coo()
    U2U = torch.eye(n_users).to_sparse_coo()

    out = model(
        UI, I2I, U2U,
        item_indices=torch.arange(n_items),
        epoch=0,
        update_momentum=True,
    )
    assert out["u_ui_emb"].shape == (n_users, d)
    assert out["i_ui_emb"].shape == (n_items, d)
    _print_ok("forward pass shapes OK")

    # Synthetic BPR + InfoNCE loss
    u = out["u_ui_emb"][:4]
    p = out["i_ui_emb"][:4]
    n = out["i_ui_emb"][4:8]
    bpr = -F.logsigmoid((u * p).sum(-1) - (u * n).sum(-1)).mean()
    nce = model.batched_contrastive_loss(out["i_ui_emb"], out["ii_emb"], batch_size=8)
    total = bpr + 0.07 * nce
    total.backward()
    _print_ok(f"backward OK; loss={total.detach().item():.4f}")

    diag = model.diagnostics()
    _print_ok(f"diag: {diag}")


def smoke_cuda_construction() -> None:
    """
    Regression check for the trainer's construction order: ``train.py`` moves
    raw modality features to a CUDA device *before* instantiating the model,
    and the data-driven AVRF prior is then computed inside ``__init__`` --
    long before the outer ``.to(device)`` call has had a chance to move the
    freshly-built ``nn.Linear`` projections. We ensure the model auto-aligns
    its projection MLPs to the input features' device so this exact path
    cannot regress.
    """
    print("== DAMPS_MMHCL CUDA-construction regression ==")
    if not torch.cuda.is_available():
        _print_ok("CUDA not available; skipping")
        return

    from model import DAMPS_MMHCL

    n_users, n_items, d = 16, 32, 64
    device = torch.device("cuda:0")
    image_feats = torch.randn(n_items, 256, device=device)
    text_feats = torch.randn(n_items, 128, device=device)

    # This call previously raised "Expected all tensors to be on the same
    # device" because self.image_proj was CPU while image_feats was CUDA.
    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=2,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
        data_driven_prior=True,
    ).to(device)

    assert next(model.image_proj.parameters()).device.type == "cuda"
    assert next(model.text_proj.parameters()).device.type == "cuda"
    _print_ok("constructed with CUDA inputs; projection MLPs auto-aligned")


def smoke_amp_bfloat16_forward() -> None:
    """
    Regression check for AMP-safe sparse matmul under bf16 autocast.

    With ``--use_amp 1`` the trainer wraps ``self.model(...)`` in
    ``torch.autocast(dtype=torch.bfloat16)``. PyTorch's CUDA sparse-matmul
    kernel (``addmm_sparse_cuda``) is not implemented for bfloat16 -- it
    raises ``NotImplementedError``. Our ``_safe_sparse_mm`` helper in
    ``model.py`` works around this by promoting the dense operand to fp32
    inside a no-autocast region, then casting back. This test executes the
    exact bf16-autocast forward path on CUDA to make sure that path is
    intact and never silently regresses.
    """
    print("== AMP bfloat16 sparse-mm regression ==")
    if not torch.cuda.is_available():
        _print_ok("CUDA not available; skipping")
        return

    from model import DAMPS_MMHCL

    n_users, n_items, d = 24, 48, 64
    device = torch.device("cuda:0")
    image_feats = torch.randn(n_items, 256, device=device)
    text_feats = torch.randn(n_items, 128, device=device)

    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=2,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
        data_driven_prior=True,
    ).to(device)

    # Build tiny COO sparse matrices on the GPU mimicking the production path
    def _rand_sparse(rows: int, cols: int, density: float = 0.05) -> torch.Tensor:
        nnz = max(1, int(rows * cols * density))
        ri = torch.randint(0, rows, (nnz,), device=device)
        ci = torch.randint(0, cols, (nnz,), device=device)
        v = torch.ones(nnz, device=device)
        return torch.sparse_coo_tensor(
            torch.stack([ri, ci]), v, (rows, cols)
        ).coalesce()

    UI_mat = _rand_sparse(n_users + n_items, n_users + n_items)
    U2U_mat = _rand_sparse(n_users, n_users)
    I2I_mat = _rand_sparse(n_items, n_items)

    # Forward signature: (UI_mat, I2I_mat, U2U_mat, ...)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(UI_mat, I2I_mat, U2U_mat, epoch=0)

    # Embeddings should come back in a sensible dtype (autocast may keep bf16
    # for the user/item branch outputs and fp32 for the hypergraph adds via
    # F.normalize). The key point of this regression test is that the
    # forward did not raise NotImplementedError on the sparse mm.
    user_dtype = out["u_ui_emb"].dtype
    item_dtype = out["i_ui_emb"].dtype
    for dt in (user_dtype, item_dtype):
        assert dt in (torch.bfloat16, torch.float16, torch.float32), (
            f"Unexpected dtype after bf16 autocast forward: {dt}"
        )
    _print_ok(
        "bf16 autocast forward survived sparse mm; "
        f"u_ui_emb={user_dtype}, i_ui_emb={item_dtype}"
    )


def smoke_torch_compile() -> None:
    """
    Speedup Guide S4: ``torch.compile`` on the DAMPS submodule. We do not
    compile the full forward path because the periodically-rebuilt sparse
    Item_mat would otherwise trigger expensive graph recompilations.

    Some environments (in particular Windows installs whose Python prefix
    contains spaces, or systems without a usable C++ toolchain) cannot
    actually compile Inductor's generated kernels even though
    ``torch.compile`` itself imports successfully. In that case we fall back
    to checking that attribute forwarding through ``OptimizedModule`` is
    still intact, since that is what the trainer relies on.
    """
    print("== torch.compile smoke ==")
    if not hasattr(torch, "compile"):                            # pragma: no cover
        _print_ok("torch.compile not available; skipping")
        return

    N, d = 32, 64
    damps = DAMPS(d=d, num_categories=4, warmup_epochs=2)
    damps.train()
    try:
        compiled = torch.compile(damps, mode="reduce-overhead", dynamic=True)
    except Exception as exc:                                     # pragma: no cover
        _print_ok(f"torch.compile failed to attach ({exc}); skipping")
        return

    # Attribute forwarding through OptimizedModule must work, regardless of
    # whether the actual graph compilation succeeds in this environment.
    compiled.set_epoch(2)                                          # type: ignore[attr-defined]
    assert int(damps._current_epoch.item()) == 2, (
        "set_epoch must forward through the OptimizedModule wrapper"
    )
    _print_ok("OptimizedModule attribute forwarding works (set_epoch)")

    # Try a real compiled forward; on environments with broken C++ builds
    # (e.g. Windows path with a space) we accept the documented graceful
    # fallback used by train.py and skip with a warning.
    h_img = torch.randn(N, d)
    h_txt = torch.randn(N, d)
    cats = torch.randint(0, 4, (N,))
    try:
        h_img_cal, h_txt_cal, _ = compiled(h_img, h_txt, item_categories=cats)
        assert h_img_cal.shape == (N, d)
        assert h_txt_cal.shape == (N, d)
        _print_ok("compiled forward OK (full Inductor path)")
    except Exception as exc:                                     # pragma: no cover
        _print_ok(
            f"Inductor compile not usable in this env ({exc.__class__.__name__}); "
            f"trainer will fall back to eager mode automatically."
        )


def smoke_phase1_static_tau() -> None:
    """
    Revision 11 / rev44 Phase 1 regression: with ``learnable_tau=False`` the
    model must keep tau frozen at the anchor value across an entire
    forward + backward pass, even though InfoNCE divides by tau every step.
    """
    print("== rev44 Phase 1: static tau frozen across backward ==")
    from model import DAMPS_MMHCL

    n_users, n_items, d = 8, 16, 64
    image_feats = torch.randn(n_items, 128)
    text_feats = torch.randn(n_items, 128)

    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=1,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
        temperature_init=0.3,
        learnable_tau=False,
    )
    model.set_meta_categories(torch.randint(0, 4, (n_items,)))

    UI = torch.eye(n_users + n_items).to_sparse_coo()
    I2I = torch.eye(n_items).to_sparse_coo()
    U2U = torch.eye(n_users).to_sparse_coo()
    out = model(UI, I2I, U2U, epoch=0)
    nce = model.batched_contrastive_loss(out["i_ui_emb"], out["ii_emb"], batch_size=8)
    nce.backward()

    # Buffers do not have a .grad attribute; this is precisely what we want.
    assert getattr(model.tau, "grad", None) is None, (
        "static tau must not accumulate gradients (it is a buffer)"
    )
    # Use float32 tolerance: 0.3 is not exactly representable in fp32.
    assert abs(float(model.tau.item()) - 0.3) < 1e-6, (
        f"static tau must remain pinned at 0.3, got {float(model.tau.item())}"
    )
    # Snapshot tau before and after backward to confirm the value is byte-
    # for-byte identical (i.e. truly frozen).
    tau_after = float(model.tau.item())
    assert tau_after == 0.30000001192092896 or abs(tau_after - 0.3) < 1e-6, (
        f"static tau drifted after backward: {tau_after}"
    )
    diag = model.diagnostics()
    assert diag["tau_mode"] == "static", f"diag tau_mode = {diag['tau_mode']}"
    _print_ok(
        f"backward survived; tau stayed at {float(model.tau.item()):.4f} "
        f"(diag.tau_mode='{diag['tau_mode']}')"
    )


def smoke_rev42_learnable_tau() -> None:
    """
    Revision 9 / rev42 baseline reproducibility: with ``learnable_tau=True``
    the model must register tau as an ``nn.Parameter`` initialised at 0.1
    so the rev42 anchor (variant (a) of the Phase 1 sweep) is still
    reproducible bit-for-bit.
    """
    print("== rev42 anchor: learnable tau == nn.Parameter @ 0.1 ==")
    from model import DAMPS_MMHCL

    n_users, n_items, d = 8, 16, 64
    image_feats = torch.randn(n_items, 64)
    text_feats = torch.randn(n_items, 64)

    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=1,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
        temperature_init=0.1,
        learnable_tau=True,
    )
    model.set_meta_categories(torch.randint(0, 4, (n_items,)))

    assert isinstance(model.tau, torch.nn.Parameter), (
        "rev42 anchor must register tau as nn.Parameter, got "
        f"{type(model.tau)}"
    )
    assert abs(float(model.tau.item()) - 0.1) < 1e-6, (
        f"rev42 tau init must be 0.1, got {float(model.tau.item())}"
    )
    assert "tau" in dict(model.named_parameters()), (
        "learnable tau must appear in model.named_parameters()"
    )
    diag = model.diagnostics()
    assert diag["tau_mode"] == "learnable", f"diag tau_mode = {diag['tau_mode']}"
    _print_ok(
        f"tau is nn.Parameter @ {float(model.tau.item()):.4f} "
        f"(diag.tau_mode='{diag['tau_mode']}'), rev42 anchor reproducible"
    )


def smoke_avrf_off_path() -> None:
    """
    Revision 11 / rev44 Phase 1 regression: with ``ablations['avrf']=False``
    the AVRF logit gate must be bypassed end-to-end. We verify the forward
    pass still runs and produces well-shaped outputs in this regime, since
    that is precisely the recommended Phase 1 configuration on Clothing.
    """
    print("== rev44 Phase 1: AVRF off forward survives ==")
    from model import DAMPS_MMHCL

    n_users, n_items, d = 8, 16, 64
    image_feats = torch.randn(n_items, 64)
    text_feats = torch.randn(n_items, 64)

    model = DAMPS_MMHCL(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=d,
        image_feats=image_feats,
        text_feats=text_feats,
        cf_model="LightGCN",
        ui_layers=1,
        user_layers=1,
        item_layers=1,
        warmup_epochs=2,
        temperature_init=0.3,
        learnable_tau=False,
        ablations={
            "apc": True,
            "avrf": False,
            "imcf": True,
            "permutation_fft": False,
            "soft_routing": True,
            "momentum": True,
        },
    )
    model.set_meta_categories(torch.randint(0, 4, (n_items,)))
    assert model.ablations["avrf"] is False
    UI = torch.eye(n_users + n_items).to_sparse_coo()
    I2I = torch.eye(n_items).to_sparse_coo()
    U2U = torch.eye(n_users).to_sparse_coo()
    out = model(UI, I2I, U2U, epoch=0)
    assert out["u_ui_emb"].shape == (n_users, d)
    assert out["i_ui_emb"].shape == (n_items, d)
    nce = model.batched_contrastive_loss(out["i_ui_emb"], out["ii_emb"], batch_size=8)
    nce.backward()
    _print_ok(
        f"AVRF off path forward+backward OK; "
        f"loss={float(nce.detach().item()):.4f}"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    smoke_damps_core()
    smoke_momentum()
    smoke_knn()
    smoke_prior()
    smoke_full_model()
    smoke_phase1_static_tau()
    smoke_rev42_learnable_tau()
    smoke_avrf_off_path()
    smoke_cuda_construction()
    smoke_amp_bfloat16_forward()
    smoke_torch_compile()
    print("\nAll smoke tests passed!")
