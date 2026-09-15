"""codes/interest_tree.py -- P6.4 TAMER Interest Tree augmentation (Algorithm 1).
==================================================================================

Ports the TAMER "Interest Tree" augmentation stage (Meng et al. MM'25,
Algorithm 1 + Eq. 7-9) to the PACER-NRDMC-lite pipeline.  Given

    * a top-k pruned item-item co-occurrence graph  Sc     (n x n)
    * an already-normalised modality similarity     Sm     (n x n)

we return the enhanced modality similarity ``S_tilde`` = Eq. 8:

    S_tilde^m_{ij} = w_coef_{ij} * Sm_{ij} + Sm_{ij}/2

where the coefficient walks the interest tree rooted at ``i`` in BFS
order up to depth ``n_order`` (default 3), pruning per-level to
``floor(|current_level| / ((order-1) * 2))``, and weighting each hit as
``gamma * exp(-(o - 1)) * w_ij^tau`` -- matching TAMER Eq. 7 and the
reference implementation in Z-last-ONE/TAMER models/tamer.py::
``find_weighted_n_order_relationships`` / ``build_session_tree``.

Complexity
----------
* ``build_weighted_binary_relations`` : O(nnz(Sc)).
* ``build_interest_tree`` per anchor  : O(n_order * mean_deg^n_order); the
  pruning schedule guarantees the total work is O(n * n_order * K) where
  K is the co-occurrence top-k.
* Full augmentation ``augment_similarity`` : O(n * (K + k_mod)) in the
  dense modality case; use sparse S_m via ``augment_similarity_sparse``.

Determinism
-----------
BFS order is deterministic given the sorted neighbour lists.  When two
candidates tie on weight we break ties by neighbour id (ascending).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Co-occurrence dict -> weighted_binary_relations
# ---------------------------------------------------------------------------
def build_weighted_binary_relations(
    cooc_rows: np.ndarray,
    cooc_cols: np.ndarray,
    cooc_vals: np.ndarray,
    top_k: int,
) -> Dict[int, Dict[int, float]]:
    """Return ``{anchor: {neighbour: weight}}`` truncated to ``top_k``.

    The input must already be top-k pruned per anchor (see
    ``codes/roaring_cooc.py::cooc_topk_roaring``); this helper only
    reshapes into the dict-of-dicts layout TAMER expects.
    """
    grouped: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for r, c, v in zip(cooc_rows.tolist(), cooc_cols.tolist(), cooc_vals.tolist()):
        grouped[int(r)].append((int(c), float(v)))
    out: Dict[int, Dict[int, float]] = {}
    for anchor, pairs in grouped.items():
        pairs.sort(key=lambda t: (-t[1], t[0]))
        out[anchor] = {j: w for j, w in pairs[:top_k]}
    return out


# ---------------------------------------------------------------------------
# BFS Interest Tree (Algorithm 1)
# ---------------------------------------------------------------------------
def build_interest_tree(
    graph: Dict[int, Dict[int, float]],
    anchor: int,
    n_order: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """BFS the interest graph up to ``n_order`` hops, pruning per level.

    Returns:
        ``{order: [(node, weight), ...]}`` for order in 1..n_order.
        Level pruning matches TAMER Eq. 7:
        ``keep = floor(|current_level| / ((order - 1) * 2))`` for order > 1.
    """
    order_dict: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    visited = {anchor}
    current_level: List[Tuple[int, float]] = [(anchor, 0.0)]
    for order in range(1, n_order + 1):
        next_level: List[Tuple[int, float]] = []
        for node, _ in current_level:
            for neighbour, weight in graph.get(node, {}).items():
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_level.append((neighbour, float(weight)))
        if not next_level:
            break
        if order > 1:
            keep = int(len(current_level) / ((order - 1) * 2))
            if keep < 1:
                keep = 1  # avoid killing the frontier entirely
            next_level.sort(key=lambda t: (-t[1], t[0]))
            next_level = next_level[:keep]
        order_dict[order] = next_level
        current_level = next_level
    return dict(order_dict)


# ---------------------------------------------------------------------------
# Similarity augmentation (Eq. 8) -- dense S_m
# ---------------------------------------------------------------------------
def augment_similarity(
    sim: torch.Tensor,
    graph: Dict[int, Dict[int, float]],
    knn_k: int,
    n_order: int = 3,
    gamma: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """Return ``S_tilde`` -- interest-tree-augmented modality similarity.

    Args:
        sim     : (n, n) dense modality similarity matrix (on any device).
        graph   : output of ``build_weighted_binary_relations``.
        knn_k   : # nearest neighbours to keep per anchor in the base S^m.
        n_order : BFS depth (default 3, matches TAMER).
        gamma   : Eq. 7 coefficient (default 1.0).
        tau     : Eq. 7 exponent on the co-occurrence weight (default 1.0).
    """
    n = sim.shape[0]
    enhanced = torch.zeros_like(sim)
    # base contribution: top-k neighbours from the modality graph
    _, cols_k = torch.topk(sim, knn_k, dim=1)
    for i in range(n):
        enhanced[i, cols_k[i]] += sim[i, cols_k[i]] / 2.0
    # interest-tree bonus
    for i in range(n):
        tree = build_interest_tree(graph, i, n_order)
        for order, level in tree.items():
            if not level:
                continue
            neigh_idx = torch.tensor([j for j, _ in level], dtype=torch.long, device=sim.device)
            weights = torch.tensor([w for _, w in level], dtype=sim.dtype, device=sim.device)
            coef = gamma * float(np.exp(-(order - 1)))
            enhanced[i, neigh_idx] = enhanced[i, neigh_idx] + coef * (weights ** tau) * sim[i, neigh_idx]
    return enhanced


# ---------------------------------------------------------------------------
# Similarity augmentation -- sparse S_m (memory-efficient path)
# ---------------------------------------------------------------------------
def augment_similarity_sparse(
    sim_csr: sp.csr_matrix,
    graph: Dict[int, Dict[int, float]],
    n_order: int = 3,
    gamma: float = 1.0,
    tau: float = 1.0,
) -> sp.csr_matrix:
    """Sparse variant of ``augment_similarity``.

    Base contribution is ``sim_csr / 2`` (the top-k pruning is assumed to
    have happened upstream).  Interest-tree bonus is added row-by-row.
    """
    n = sim_csr.shape[0]
    lil = (sim_csr / 2.0).tolil()
    for i in range(n):
        tree = build_interest_tree(graph, i, n_order)
        for order, level in tree.items():
            coef = gamma * float(np.exp(-(order - 1)))
            for j, w in level:
                # need sim[i, j]; look up in the CSR row
                s_ij = sim_csr[i, j]
                if s_ij == 0.0:
                    continue
                lil[i, j] = lil[i, j] + coef * (w ** tau) * s_ij
    return lil.tocsr()


# ---------------------------------------------------------------------------
# TAMER Eq. 9 fusion: S_tilde = Sum_hg alpha_hg * S_tilde^hg
# ---------------------------------------------------------------------------
def fuse_augmented(
    per_view: Dict[str, torch.Tensor],
    alphas: Dict[str, float],
) -> torch.Tensor:
    """Weighted sum of augmented similarities across content views {v, p, z, c}.

    Missing views default to 0-contribution; ``alphas`` sum-to-one is NOT
    enforced -- caller controls normalisation (TAMER treats alpha_hg as
    learnable soft weights).
    """
    keys = list(per_view.keys())
    if not keys:
        raise ValueError("per_view is empty")
    out = torch.zeros_like(per_view[keys[0]])
    for name, tensor in per_view.items():
        a = float(alphas.get(name, 0.0))
        if a == 0.0:
            continue
        out = out + a * tensor
    return out


# ---------------------------------------------------------------------------
# Precomputed interest-tree flattener (P6.4a)
# ---------------------------------------------------------------------------
def precompute_interest_tree_flat(
    graph: Dict[int, Dict[int, float]],
    n_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``build_interest_tree`` for every anchor and flatten to arrays.

    Returns (anchors, neighbours, orders, weights) as parallel int64/int64/
    int32/float32 arrays with one entry per (anchor, hop, neighbour) tuple.
    Because the BFS depends only on ``graph`` (top-k co-occurrence) and
    ``n_order`` -- not on gamma/tau/alpha -- the flat result is reused
    across every P6.4 grid cell, moving the ~Python BFS cost off the
    training loop.
    """
    anchors: List[int] = []
    neighbours: List[int] = []
    orders: List[int] = []
    weights: List[float] = []
    for i in graph.keys():
        tree = build_interest_tree(graph, i, n_order)
        for order, level in tree.items():
            for j, w in level:
                anchors.append(i)
                neighbours.append(int(j))
                orders.append(int(order))
                weights.append(float(w))
    return (
        np.asarray(anchors, dtype=np.int64),
        np.asarray(neighbours, dtype=np.int64),
        np.asarray(orders, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
    )


def precompute_interest_tree_flat_parallel(
    graph: Dict[int, Dict[int, float]],
    n_order: int,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel variant of ``precompute_interest_tree_flat``.

    Splits the anchor keys across ``num_workers`` processes (0 or 1 =
    sequential). Each worker gets a shallow copy of ``graph`` (BFS reads
    only, no writes). Windows-native ``spawn`` friendly -- the worker
    function is a module-level callable.
    """
    if num_workers <= 1:
        return precompute_interest_tree_flat(graph, n_order)
    import concurrent.futures as _cf

    anchor_ids = sorted(graph.keys())
    if not anchor_ids:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty(0, np.int32), np.empty(0, np.float32))
    # Chunk contiguously so each worker gets a similar-sized share.
    chunk = (len(anchor_ids) + num_workers - 1) // num_workers
    slices = [anchor_ids[k:k + chunk] for k in range(0, len(anchor_ids), chunk)]

    out_a: List[np.ndarray] = []
    out_n: List[np.ndarray] = []
    out_o: List[np.ndarray] = []
    out_w: List[np.ndarray] = []
    with _cf.ProcessPoolExecutor(max_workers=num_workers) as pool:
        futs = [pool.submit(_bfs_chunk_worker, graph, s, n_order) for s in slices]
        for fut in futs:
            a, n, o, w = fut.result()
            out_a.append(a)
            out_n.append(n)
            out_o.append(o)
            out_w.append(w)
    return (np.concatenate(out_a),
            np.concatenate(out_n),
            np.concatenate(out_o),
            np.concatenate(out_w))


def _bfs_chunk_worker(
    graph: Dict[int, Dict[int, float]],
    anchors: List[int],
    n_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ProcessPoolExecutor worker: BFS the given anchor slice."""
    a_out: List[int] = []
    n_out: List[int] = []
    o_out: List[int] = []
    w_out: List[float] = []
    for i in anchors:
        tree = build_interest_tree(graph, i, n_order)
        for order, level in tree.items():
            for j, w in level:
                a_out.append(i)
                n_out.append(int(j))
                o_out.append(int(order))
                w_out.append(float(w))
    return (np.asarray(a_out, dtype=np.int64),
            np.asarray(n_out, dtype=np.int64),
            np.asarray(o_out, dtype=np.int32),
            np.asarray(w_out, dtype=np.float32))


__all__ = [
    "build_weighted_binary_relations",
    "build_interest_tree",
    "augment_similarity",
    "augment_similarity_sparse",
    "fuse_augmented",
    "precompute_interest_tree_flat",
    "precompute_interest_tree_flat_parallel",
]
