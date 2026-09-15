"""
utility/load_data.py — Data Loading & Graph Construction
==========================================================

Compatible drop-in replacement for the original
``codes/utility/load_data.py``.

Responsibilities
----------------
1.  Read train / val / test JSON splits.
2.  Build the sparse user-item interaction matrix R.
3.  Construct three graph structures used by the MMHCL backbone:
    *   ``UI_mat``    — user-item bipartite graph (sym normalised).
    *   ``User_mat``  — user-user co-interaction graph (random-walk normalised).
    *   ``Item_mat``  — item-item multi-modal hypergraph (sym normalised).
4.  **NEW (DAMPS)**: expose raw image / text / [audio] features so that
    the DAMPS spectral calibrator can use them at every forward pass and
    the data-driven AVRF prior can be derived once at model construction
    time.
5.  Provide BPR sampling.

Expected dataset layout
-----------------------
::

    data/<dataset>/
        5-core/
            train.json, val.json, test.json
        image_feat.npy            # (n_items, image_dim)  e.g. 4096
        text_feat.npy             # (n_items, text_dim)   e.g. 768
        audio_feat.npy            # (n_items, audio_dim)  Tiktok only
        meta_categories.npy       # (n_items,) int  *optional*
                                   # Static metadata category for APC

If ``meta_categories.npy`` is missing we fall back to a deterministic hash
of the item IDs so APC still has a clustering signal (still no k-means).
"""

from __future__ import annotations

import json
import os
import random as rd
from time import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import torch

from utility.parser import parse_args


args = parse_args()


def _torch_load(path: str):
    """Compatibility wrapper for torch.load (forces weights_only=False)."""
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
#  Data class
# ---------------------------------------------------------------------------
class Data:
    """
    Loads a recommendation dataset and builds all required adjacency matrices.

    Attributes (post-init):
        n_users, n_items, n_train, n_test, n_val
        R          : sp.dok_matrix — (n_users, n_items) binary interactions
        train_items, test_set, val_set : per-user item lists
        exist_users : training users with >= 1 interaction
        meta_categories : (n_items,) Long — static metadata clusters for APC
        image_feats, text_feats, audio_feats : raw modality tensors (or None)
    """

    def __init__(self, path: str, batch_size: int) -> None:
        # ------------------------------------------------------------------
        # 1. Resolve file paths
        # ------------------------------------------------------------------
        self.dataset: str = args.dataset
        self.path: str = os.path.join(path, f"{args.core}-core")
        self.batch_size: int = batch_size

        train_file = os.path.join(self.path, "train.json")
        val_file = os.path.join(self.path, "val.json")
        test_file = os.path.join(self.path, "test.json")

        # ------------------------------------------------------------------
        # 2. Read JSON splits
        # ------------------------------------------------------------------
        with open(train_file, "r", encoding="utf-8") as f:
            train = json.load(f)
        with open(test_file, "r", encoding="utf-8") as f:
            test = json.load(f)
        with open(val_file, "r", encoding="utf-8") as f:
            val = json.load(f)

        self.n_users: int = 0
        self.n_items: int = 0
        self.n_train: int = 0
        self.n_test: int = 0
        self.n_val: int = 0
        self.exist_users: list[int] = []
        self.neg_pools: dict[int, list[int]] = {}

        # Scan training data
        for uid_str, items in train.items():
            if not items:
                continue
            uid = int(uid_str)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        # Scan test/val data
        for uid_str, items in test.items():
            try:
                if items:
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
            except Exception:
                continue
        for uid_str, items in val.items():
            try:
                if items:
                    self.n_items = max(self.n_items, max(items))
                    self.n_val += len(items)
            except Exception:
                continue

        self.n_items += 1
        self.n_users += 1
        self.print_statistics()

        # ------------------------------------------------------------------
        # 3. Build the binary user-item interaction matrix R
        # ------------------------------------------------------------------
        self.R: sp.dok_matrix = sp.dok_matrix(
            (self.n_users, self.n_items), dtype=np.float32
        )
        self.train_items: dict[int, list[int]] = {}
        self.test_set: dict[int, list[int]] = {}
        self.val_set: dict[int, list[int]] = {}

        for uid_str, train_items_list in train.items():
            if not train_items_list:
                continue
            uid = int(uid_str)
            for i in train_items_list:
                self.R[uid, i] = 1.0
            self.train_items[uid] = train_items_list

        for uid_str, lst in test.items():
            if lst:
                self.test_set[int(uid_str)] = lst
        for uid_str, lst in val.items():
            if lst:
                self.val_set[int(uid_str)] = lst

        # ------------------------------------------------------------------
        # 4. Load raw modality features (used by DAMPS + cached I2I builds)
        # ------------------------------------------------------------------
        image_raw = self._load_modality("image")
        text_raw = self._load_modality("text")
        # ------------------------------------------------------------------
        # P6.0 / P6.1 -- MACP whitening (text and/or image).
        # When --use_macp 0 (default) both branches short-circuit and the
        # loader behaves bit-for-bit like the pre-P6 baseline. When
        # --use_macp 1, the per-modality --macp_mode / --macp_image_mode
        # flags decide whether to swap in the pre-computed PCA->ICA or
        # ZCA streams (produced by ``scripts/preprocess_macp.py``).
        #
        # P6.0 established replace_pca on text as the winner (+7.1 %
        # R@20 on Clothing, +37 % / +46 % mid/tail). P6.1 tests whether
        # symmetric image whitening reverses the alpha_img<0 collapse
        # observed in the P6.0 log (alpha_img -> -0.84 under clean text).
        # ------------------------------------------------------------------
        self._macp_diag: dict = {}
        self._macp_image_diag: dict = {}
        if bool(int(getattr(args, "use_macp", 0))):
            from damps.macp import MacpConfig, fuse_text, fuse_image
            dataset_dir = os.path.join(args.data_path, args.dataset)
            verbose = bool(int(getattr(args, "macp_verbose", 1)))

            if text_raw is not None:
                text_cfg = MacpConfig(
                    mode=getattr(args, "macp_mode", "residual"),
                    alpha_p=float(getattr(args, "macp_alpha_p", 0.10)),
                    alpha_z=float(getattr(args, "macp_alpha_z", 0.10)),
                )
                text_raw, self._macp_diag = fuse_text(
                    text_raw, dataset_dir=dataset_dir, cfg=text_cfg,
                    verbose=verbose,
                )

            if image_raw is not None:
                image_cfg = MacpConfig(
                    mode=getattr(args, "macp_image_mode", "raw"),
                    alpha_p=float(getattr(args, "macp_image_alpha_p", 0.10)),
                    alpha_z=float(getattr(args, "macp_image_alpha_z", 0.10)),
                )
                image_raw, self._macp_image_diag = fuse_image(
                    image_raw, dataset_dir=dataset_dir, cfg=image_cfg,
                    verbose=verbose,
                )
        self.image_feats: Optional[torch.Tensor] = image_raw
        self.text_feats: Optional[torch.Tensor] = text_raw
        self.audio_feats: Optional[torch.Tensor] = (
            self._load_modality("audio") if self.dataset.lower() == "tiktok" else None
        )

        # ------------------------------------------------------------------
        # 5. Load (or fabricate) static metadata categories for APC
        # ------------------------------------------------------------------
        self.meta_categories: torch.Tensor = self._load_metadata_categories()

    # ==================================================================
    #  Stats
    # ==================================================================
    def print_statistics(self) -> None:
        print(f"n_users={self.n_users}, n_items={self.n_items}")
        print(f"n_interactions={self.n_train + self.n_test}")
        if self.n_users * self.n_items:
            sparsity = (self.n_train + self.n_test) / (self.n_users * self.n_items)
        else:
            sparsity = 0.0
        print(
            f"n_train={self.n_train}, n_test={self.n_test}, sparsity={sparsity:.5f}"
        )

    # ==================================================================
    #  Modality loaders
    # ==================================================================
    def _load_modality(self, name: str) -> Optional[torch.Tensor]:
        """Load <dataset>/<name>_feat.npy if present, else return None."""
        candidate_paths = [
            os.path.join(args.data_path, args.dataset, f"{name}_feat.npy"),
            os.path.join(args.data_path, args.dataset, f"{name}_feat.pt"),
        ]
        for fp in candidate_paths:
            if not os.path.exists(fp):
                continue
            try:
                if fp.endswith(".npy"):
                    arr = np.load(fp)
                    return torch.tensor(arr).float()
                return _torch_load(fp).float()
            except Exception as exc:                              # pragma: no cover
                print(f"[load_data] failed to load {fp}: {exc}")
                continue
        print(f"[load_data] modality '{name}' not found — DAMPS will skip it")
        return None

    def _load_metadata_categories(self) -> torch.Tensor:
        """
        Load static metadata categories used by Metadata-Aware APC.

        Falls back to a deterministic hash if no file is present so the
        ``num_categories`` parameter is still meaningful.
        """
        n_cats = max(1, int(args.damps_num_categories))
        cand = os.path.join(args.data_path, args.dataset, "meta_categories.npy")
        if os.path.exists(cand):
            try:
                arr = np.load(cand).astype(np.int64)
                if arr.shape[0] != self.n_items:
                    print(
                        f"[load_data] meta_categories shape {arr.shape} "
                        f"does not match n_items={self.n_items} — falling back"
                    )
                else:
                    return torch.from_numpy(np.clip(arr, 0, n_cats - 1)).long()
            except Exception as exc:                              # pragma: no cover
                print(f"[load_data] failed to load meta_categories: {exc}")

        # Deterministic fallback: hash item index into n_cats buckets
        ids = torch.arange(self.n_items, dtype=torch.long)
        return (ids * 2654435761 % n_cats).long()

    # ==================================================================
    #  BPR Sampling
    # ==================================================================
    def sample(self) -> tuple[list[int], list[int], list[int]]:
        """Sample a batch of (user, pos_item, neg_item) triplets."""
        if self.batch_size <= len(self.exist_users):
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        pos_items: list[int] = []
        neg_items: list[int] = []

        for u in users:
            pos_items.append(self._sample_pos(u))
            neg_items.append(self._sample_neg(u))
        return users, pos_items, neg_items

    def _sample_pos(self, u: int) -> int:
        items = self.train_items[u]
        return items[np.random.randint(0, len(items))]

    def _sample_neg(self, u: int) -> int:
        seen = set(self.train_items[u])
        while True:
            cand = int(np.random.randint(0, self.n_items))
            if cand not in seen:
                return cand

    def _ensure_gpu_sample_cache(self, device: torch.device) -> None:
        """Build padded positive-item tables on ``device`` (once per device)."""
        cache_dev = getattr(self, "_gpu_sample_device", None)
        if (
            cache_dev is not None
            and cache_dev == device
            and hasattr(self, "_exist_users_t")
        ):
            return

        exist = torch.tensor(self.exist_users, dtype=torch.long, device=device)
        max_len = max(len(self.train_items[u]) for u in self.exist_users)
        # Pad with the first positive so out-of-range gathers stay valid.
        pos_pad = torch.zeros(
            (self.n_users, max_len), dtype=torch.long, device=device
        )
        pos_lens = torch.ones(self.n_users, dtype=torch.long, device=device)
        for u in self.exist_users:
            items = self.train_items[u]
            n = len(items)
            pos_pad[u, :n] = torch.tensor(items, dtype=torch.long, device=device)
            if n < max_len:
                pos_pad[u, n:] = items[0]
            pos_lens[u] = n

        self._exist_users_t = exist
        self._pos_pad_t = pos_pad
        self._pos_lens_t = pos_lens
        self._gpu_sample_device = device

    def sample_gpu(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU BPR triplet sampling (speedup guide Section C / step 7).

        Negatives are drawn uniformly on-device; collisions with a user's
        positives (~1% on Clothing density) are ignored — the same
        approximation used by SASRec / LightGCN.

        Args:
            device: CUDA (or CPU) device that owns the returned tensors.

        Returns:
            ``(users, pos_items, neg_items)`` each of shape ``[batch_size]``.
        """
        self._ensure_gpu_sample_cache(device)
        bsz = self.batch_size
        n_exist = self._exist_users_t.numel()
        u_idx = torch.randint(0, n_exist, (bsz,), device=device)
        users = self._exist_users_t[u_idx]
        lens = self._pos_lens_t[users]
        # Uniform index in [0, len_u) per row.
        offsets = (torch.rand(bsz, device=device) * lens.float()).long()
        offsets = torch.minimum(offsets, lens - 1)
        pos_items = self._pos_pad_t[users, offsets]
        neg_items = torch.randint(0, self.n_items, (bsz,), device=device)
        return users, pos_items, neg_items

    # ==================================================================
    #  GPU eval masks (PACER_NRDMC_lite_eval_bottleneck_EN §4)
    # ==================================================================
    def _build_bool_mask(
        self,
        user_items: dict[int, list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Dense ``[n_users, n_items]`` bool mask from a user→items map."""
        mask = torch.zeros(
            (self.n_users, self.n_items), dtype=torch.bool, device=device
        )
        if not user_items:
            return mask
        rows: list[int] = []
        cols: list[int] = []
        for uid, items in user_items.items():
            if not items:
                continue
            u = int(uid)
            rows.extend([u] * len(items))
            cols.extend(int(i) for i in items)
        if rows:
            mask[
                torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(cols, dtype=torch.long, device=device),
            ] = True
        return mask

    def get_train_mask_gpu(self, device: torch.device) -> torch.Tensor:
        """Cached train-item exclusion mask ``[n_users, n_items]`` on device."""
        cache = getattr(self, "_train_mask_gpu", None)
        cache_dev = getattr(self, "_train_mask_device", None)
        if cache is not None and cache_dev == device:
            return cache
        mask = self._build_bool_mask(self.train_items, device)
        self._train_mask_gpu = mask
        self._train_mask_device = device
        return mask

    def get_gt_mask_gpu(
        self,
        is_val: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Cached ground-truth mask for val (``is_val``) or test split."""
        attr = "_val_mask_gpu" if is_val else "_test_mask_gpu"
        dev_attr = "_val_mask_device" if is_val else "_test_mask_device"
        cache = getattr(self, attr, None)
        cache_dev = getattr(self, dev_attr, None)
        if cache is not None and cache_dev == device:
            return cache
        src = self.val_set if is_val else self.test_set
        mask = self._build_bool_mask(src, device)
        setattr(self, attr, mask)
        setattr(self, dev_attr, device)
        return mask

    def get_gt_counts(
        self,
        users: list[int] | torch.Tensor,
        is_val: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-user ground-truth positive counts for ``users`` on ``device``."""
        gt = self.get_gt_mask_gpu(is_val, device)
        if isinstance(users, list):
            users_t = torch.tensor(users, dtype=torch.long, device=device)
        else:
            users_t = users.to(device=device, dtype=torch.long)
        return gt[users_t].sum(dim=1).to(dtype=torch.float32)

    # ==================================================================
    #  Sparse helpers
    # ==================================================================
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx: sp.spmatrix) -> torch.Tensor:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        idx = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        vals = torch.from_numpy(sparse_mx.data)
        return torch.sparse_coo_tensor(idx, vals, torch.Size(sparse_mx.shape))

    # ==================================================================
    #  Adjacency normalisation (dense)
    # ==================================================================
    @staticmethod
    def norm_dense(adj: torch.Tensor, normalization: str = "origin") -> torch.Tensor:
        if normalization == "sym":
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            d_mat = torch.diagflat(d_inv_sqrt)
            return torch.mm(torch.mm(d_mat, adj), d_mat)
        if normalization == "2sym":
            rowsum = torch.sum(adj, -1)
            d_row = torch.pow(rowsum, -0.5)
            d_row[torch.isinf(d_row)] = 0.0
            d_row_mat = torch.diagflat(d_row)
            colsum = torch.sum(adj, -2)
            d_col = torch.pow(colsum, -0.5)
            d_col[torch.isinf(d_col)] = 0.0
            d_col_mat = torch.diagflat(d_col)
            return torch.mm(torch.mm(d_row_mat, adj), d_col_mat)
        if normalization == "rw":
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.0
            d_mat = torch.diagflat(d_inv)
            return torch.mm(d_mat, adj)
        return adj

    # ==================================================================
    #  Graph builders (cached on disk under <path>)
    # ==================================================================
    def get_UI_mat(self, norm_type: str = "sym") -> torch.Tensor:
        cache = os.path.join(self.path, f"UI_mat_{norm_type}.pth")
        try:
            return _torch_load(cache)
        except Exception:
            pass

        adj_lil = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        ).tolil()
        R = self.R.tolil()
        adj_lil[: self.n_users, self.n_users :] = R
        adj_lil[self.n_users :, : self.n_users] = R.T
        dense = torch.from_numpy(np.asarray(adj_lil.todense())).float()
        dense = self.norm_dense(dense, norm_type)
        sparse_t = dense.to_sparse()
        torch.save(sparse_t, cache)
        return sparse_t

    def get_U2U_mat(self, norm_type: str = "rw") -> torch.Tensor:
        cache = os.path.join(self.path, f"User_mat_{norm_type}.pth")
        try:
            return _torch_load(cache)
        except Exception:
            pass

        R = torch.from_numpy(np.asarray(self.R.todense())).float()
        user_mat = R @ R.T
        n_user = user_mat.size(0)
        mask = torch.eye(n_user)
        user_mat[mask > 0] = 0.0
        user_mat = self.norm_dense(user_mat, norm_type)
        sparse_t = user_mat.to_sparse()
        torch.save(sparse_t, cache)
        return sparse_t

    # ------------------------------------------------------------------
    #  Item-Item multi-modal hypergraph helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_sim(context: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix from a feature matrix."""
        norm = torch.norm(context, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        ctx = context / norm
        return torch.mm(ctx, ctx.transpose(0, 1))

    @staticmethod
    def build_knn_normalized_graph(adj: torch.Tensor, topk: int) -> torch.Tensor:
        """K-NN sparsification: keep top-K per row, binarise."""
        _, knn_ind = torch.topk(adj, topk, dim=-1)
        out = torch.zeros_like(adj)
        out.scatter_(-1, knn_ind, 1.0)
        return out

    def build_static_hypergraph(
        self,
        norm_type: str = "sym",
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build the **static** multi-modal hypergraph (raw features → K-NN →
        H @ H^T → normalise). Used as the "bootstrap" hypergraph during the
        warm-up phase before Pattern B' takes over.

        Cached at ``<path>/hypergraph_mat_mul_<norm>_topk_<K>.pth``.
        """
        if topk is None:
            topk = args.topk
        cache = os.path.join(
            self.path, f"hypergraph_mat_mul_{norm_type}_topk_{topk}.pth"
        )
        try:
            return _torch_load(cache)
        except Exception:
            pass

        modalities: list[torch.Tensor] = []
        if self.image_feats is not None:
            adj = self.build_knn_normalized_graph(
                self.build_sim(self.image_feats), topk=topk
            )
            modalities.append(adj)
        if self.text_feats is not None:
            adj = self.build_knn_normalized_graph(
                self.build_sim(self.text_feats), topk=topk
            )
            modalities.append(adj)
        if self.audio_feats is not None:
            adj = self.build_knn_normalized_graph(
                self.build_sim(self.audio_feats), topk=topk
            )
            modalities.append(adj)

        if not modalities:
            raise RuntimeError(
                "No modality features found — at least one of image/text/audio "
                "_feat.npy must exist for DAMPS-MMHCL"
            )

        H = torch.cat(modalities, dim=1)            # (N, m * N)
        adj = H @ H.transpose(0, 1)
        adj = self.norm_dense(adj, norm_type)
        sparse_t = adj.to_sparse()
        torch.save(sparse_t, cache)
        return sparse_t
