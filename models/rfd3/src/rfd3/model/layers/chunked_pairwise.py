"""
Chunked pairwise embedding implementation for memory-efficient large structure processing.

This module provides memory-optimized versions of pairwise embedders that compute
only the pairs needed for sparse attention, reducing memory usage from O(L²) to O(L×k).
"""

import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from rfd3.model.layers.blocks import PositionPairDistEmbedder, SinusoidalDistEmbed
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias
from rfd3.utils.tracing import trace_range


def _chunked_pairwise_compute(
    indices: torch.Tensor,          # [B, L, k]
    C_L: torch.Tensor,              # [B, L, c_token] (already batched)
    c_atompair: int,
    # Motif inputs (may be zero-length dummies if no motif)
    motif_pos: torch.Tensor,        # [L, 3]
    is_motif: torch.Tensor,         # [L] bool
    has_motif: bool,
    motif_output_proj_weight: torch.Tensor,   # [2*n_freqs, c_atompair]
    motif_valid_mask_weight: torch.Tensor,    # [1, c_atompair]
    motif_n_freqs: int,
    # Ref inputs (may be zero-length dummies if no ref)
    ref_pos: torch.Tensor,          # [L, 3]
    ref_space_uid: torch.Tensor,    # [L]
    is_motif_seq: torch.Tensor,     # [L] bool
    has_ref: bool,
    ref_embed_frame: bool,
    ref_process_d_weight: torch.Tensor,       # [3, c_atompair]
    ref_inv_dist_weight: torch.Tensor,        # [1, c_atompair]
    ref_valid_mask_weight: torch.Tensor,      # [1, c_atompair]
    # Single feature inputs
    sl_cached: Optional[torch.Tensor],        # [L, c_atompair] or None
    sm_cached: Optional[torch.Tensor],        # [L, c_atompair] or None
    # Token pair inputs
    tok_idx: torch.Tensor,          # [L]
    Z_processed: torch.Tensor,      # [I, I, c_atompair]
) -> torch.Tensor:
    """
    Pure-tensor compute core for chunked pairwise embedding.
    No Python-level loops, no context managers — fully compilable by torch.compile.
    """
    B, L, k = indices.shape
    device = indices.device
    dtype = C_L.dtype

    L_max = C_L.shape[1]
    valid_indices = torch.clamp(indices, 0, L_max - 1)

    P_LL_sparse = torch.zeros(B, L, k, c_atompair, device=device, dtype=dtype)

    # --- 1. Motif sinusoidal embedding ---
    if has_motif:
        key_pos_all = motif_pos[valid_indices]                          # [B, L, k, 3]
        query_pos_all = motif_pos.unsqueeze(0).expand(B, L, 3)
        D_pairs_all = query_pos_all.unsqueeze(2) - key_pos_all          # [B, L, k, 3]
        key_is_motif_all = is_motif[valid_indices].unsqueeze(-1).float() # [B, L, k, 1]

        dist_all = torch.linalg.norm(D_pairs_all, dim=-1)               # [B, L, k]
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, motif_n_freqs, dtype=torch.float32, device=device)
            / motif_n_freqs
        )
        angles = dist_all.unsqueeze(-1) * freq                          # [B, L, k, n_freqs]
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # Linear projection via matmul (output_proj is linearNoBias)
        motif_pairs = torch.matmul(sincos, motif_output_proj_weight)     # [B, L, k, c_atompair]
        motif_pairs = motif_pairs * key_is_motif_all
        motif_pairs = (
            motif_pairs
            + torch.matmul(key_is_motif_all, motif_valid_mask_weight) * key_is_motif_all
        )
        query_is_motif = is_motif.view(1, L, 1, 1).float()
        P_LL_sparse = P_LL_sparse + motif_pairs * query_is_motif

    # --- 2. Reference position pair dist embedding ---
    if has_ref:
        key_pos_all = ref_pos[valid_indices]                             # [B, L, k, 3]
        query_pos_all = ref_pos.unsqueeze(0).expand(B, L, 3)
        D_pairs_all = query_pos_all.unsqueeze(2) - key_pos_all           # [B, L, k, 3]

        key_space_uid = ref_space_uid[valid_indices]
        query_space_uid = ref_space_uid.view(1, L, 1).expand(B, L, k)
        key_is_motif_seq = is_motif_seq[valid_indices]
        same_token = key_space_uid == query_space_uid
        valid_mask_all = (same_token & key_is_motif_seq).unsqueeze(-1).float()

        if ref_embed_frame:
            ref_pairs = torch.matmul(D_pairs_all, ref_process_d_weight) * valid_mask_all
            norm_sq = torch.linalg.norm(D_pairs_all, dim=-1, keepdim=True) ** 2
            inv_dist = 1 / (1 + norm_sq)
            ref_pairs = ref_pairs + torch.matmul(inv_dist, ref_inv_dist_weight) * valid_mask_all
            ref_pairs = (
                ref_pairs
                + torch.matmul(valid_mask_all, ref_valid_mask_weight) * valid_mask_all
            )
        else:
            norm_sq = torch.linalg.norm(D_pairs_all, dim=-1, keepdim=True) ** 2
            norm_sq = torch.clamp(norm_sq, min=1e-6)
            inv_dist = 1 / (1 + norm_sq)
            ref_pairs = torch.matmul(inv_dist, ref_inv_dist_weight) * valid_mask_all
            ref_pairs = (
                ref_pairs
                + torch.matmul(valid_mask_all, ref_valid_mask_weight) * valid_mask_all
            )

        query_is_motif_seq = is_motif_seq.view(1, L, 1, 1).float()
        P_LL_sparse = P_LL_sparse + ref_pairs * query_is_motif_seq

    # --- 3. Single feature terms (cached path only) ---
    if sl_cached is not None and sm_cached is not None:
        single_l = sl_cached.unsqueeze(0).unsqueeze(2).expand(B, -1, k, -1)
        single_m = sm_cached[valid_indices]
        P_LL_sparse = P_LL_sparse + single_l + single_m

    # --- 4. Token pair features ---
    tok_idx_expanded = tok_idx.unsqueeze(0).expand(B, L)
    tok_queries = tok_idx_expanded.unsqueeze(2).expand(-1, -1, k)
    tok_keys = torch.gather(tok_queries, 1, valid_indices)
    I_z = Z_processed.shape[0]
    I_z2 = Z_processed.shape[1]
    tq = torch.clamp(tok_queries, 0, I_z - 1)
    tk = torch.clamp(tok_keys, 0, I_z2 - 1)
    # Flatten for single advanced-index gather (avoids loop over B)
    Z_pairs = Z_processed[tq.reshape(-1), tk.reshape(-1)].reshape(B, L, k, c_atompair)
    P_LL_sparse = P_LL_sparse + Z_pairs

    return P_LL_sparse


@lru_cache(maxsize=1)
def _get_compiled_compute():
    """Lazily compile the compute function on first use."""
    return torch.compile(_chunked_pairwise_compute, dynamic=True)


class ChunkedPositionPairDistEmbedder:
    """
    Memory-efficient version of PositionPairDistEmbedder that computes pairs on-demand.
    Uses a trained PositionPairDistEmbedder instance and shares the forward method.
    """

    def __init__(self, embedder_instance: PositionPairDistEmbedder):
        """
        Initialize the ChunkedPositionPairDistEmbedder from a parent PositionPairDistEmbedder instance.
        """
        self.embed_frame = embedder_instance.embed_frame
        if embedder_instance.embed_frame:
            self.process_d = embedder_instance.process_d
        self.process_inverse_dist = embedder_instance.process_inverse_dist
        self.process_valid_mask = embedder_instance.process_valid_mask
        self.forward = embedder_instance.forward

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,  # [B, 3]
        key_pos: torch.Tensor,  # [B, k, 3]
        valid_mask: torch.Tensor,  # [B, k, 1]
    ) -> torch.Tensor:
        """
        Compute pairwise embeddings for specific query-key pairs.

        Args:
            query_pos: Query positions [B, 3]
            key_pos: Key positions [B, k, 3]
            valid_mask: Valid pair mask [B, k, 1]

        Returns:
            P_sparse: Pairwise embeddings [B, k, c_atompair]
        """
        # Compute pairwise distances: [B, k, 3]
        D_pairs = query_pos.unsqueeze(1) - key_pos  # [B, 1, 3] - [B, k, 3] = [B, k, 3]

        if self.embed_frame:
            # Embed pairwise distances
            P_pairs = self.process_d(D_pairs) * valid_mask  # [B, k, c_atompair]

            # Add inverse distance embedding
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2  # [B, k, 1]
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = P_pairs + self.process_inverse_dist(inv_dist) * valid_mask

            # Add valid mask embedding
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )
        else:
            # Simplified version without frame embedding
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2
            norm_sq = torch.clamp(norm_sq, min=1e-6)
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = self.process_inverse_dist(inv_dist) * valid_mask
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )

        return P_pairs


class ChunkedSinusoidalDistEmbed:
    """
    Memory-efficient version of SinusoidalDistEmbed.
    Uses a trained SinusoidalDistEmbed instance and shares the forward method.
    """

    def __init__(self, embedder_instance: SinusoidalDistEmbed):
        """
        Initialize the ChunkedSinusoidalDistEmbed from a parent SinusoidalDistEmbed instance.
        """
        assert (
            embedder_instance.c_atompair % 2 == 0
        ), "Output embedding dim must be even"

        self.n_freqs = embedder_instance.n_freqs
        self.c_atompair = embedder_instance.c_atompair
        self.output_proj = embedder_instance.output_proj
        self.process_valid_mask = embedder_instance.process_valid_mask
        self.forward = embedder_instance.forward

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,  # [B, 3]
        key_pos: torch.Tensor,  # [B, k, 3]
        valid_mask: torch.Tensor,  # [B, k, 1]
    ) -> torch.Tensor:
        """
        Compute sinusoidal distance embeddings for specific query-key pairs.
        """
        device = query_pos.device

        # Compute pairwise distances
        D_pairs = query_pos.unsqueeze(1) - key_pos  # [B, k, 3]
        dist_matrix = torch.linalg.norm(D_pairs, dim=-1)  # [B, k]

        # Sinusoidal embedding
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / half_dim
        )  # [n_freqs]

        angles = dist_matrix.unsqueeze(-1) * freq  # [B, k, n_freqs]
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [B, k, 2*n_freqs]

        # Linear projection
        P_pairs = self.output_proj(sincos_embed)  # [B, k, c_atompair]
        P_pairs = P_pairs * valid_mask

        # Add linear embedding of valid mask
        P_pairs = (
            P_pairs + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
        )

        return P_pairs


class ChunkedPairwiseEmbedder:
    """
    Main chunked pairwise embedder that combines all embedding types.
    This replaces the full P_LL computation with sparse computation.

    Not an nn.Module: all sub-components are shared references from TokenInitializer
    and already registered there. Inheriting nn.Module would cause them to appear
    under duplicate paths (chunked_pairwise_embedder.*) in state_dict, which don't
    exist in checkpoints trained without use_chunked_pll.
    """

    def __init__(
        self,
        c_atompair: int,
        motif_pos_embedder: ChunkedSinusoidalDistEmbed,
        ref_pos_embedder: ChunkedPositionPairDistEmbedder,
        process_single_l: Optional[nn.Module] = None,
        process_single_m: Optional[nn.Module] = None,
        process_z: Optional[nn.Module] = None,
        pair_mlp: Optional[nn.Module] = None,
        **kwargs,
    ):
        self.c_atompair = c_atompair
        self.motif_pos_embedder = motif_pos_embedder
        self.ref_pos_embedder = ref_pos_embedder

        # Use shared trained MLPs if provided, otherwise create new ones
        if process_single_l is not None:
            self.process_single_l = process_single_l
        else:
            self.process_single_l = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_single_m is not None:
            self.process_single_m = process_single_m
        else:
            self.process_single_m = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_z is not None:
            self.process_z = process_z
        else:
            self.process_z = nn.Sequential(RMSNorm(128), linearNoBias(128, c_atompair))

        if pair_mlp is not None:
            self.pair_mlp = pair_mlp
        else:
            self.pair_mlp = nn.Sequential(
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
            )

        # Cached static projections — populated once at tokenization by
        # cache_static_projections().  None means "not yet cached; run the MLP."
        self._sl_cached: Optional[torch.Tensor] = None  # [L, c_atompair]
        self._sm_cached: Optional[torch.Tensor] = None  # [L, c_atompair]
        self._Z_proc_cached: Optional[torch.Tensor] = None  # [I, I, c_atompair]

    def cache_static_projections(
        self, C_L: torch.Tensor, Z_init_II: torch.Tensor
    ) -> None:
        """
        Precompute and cache the three MLP projections that are identical across
        all diffusion steps (they depend only on the static tokenization outputs).

        Call this once after tokenization, before the diffusion loop.
        forward_chunked will then replace those MLP calls with free tensor indexing.

        Args:
            C_L:       Atom features [L, c_token]
            Z_init_II: Token-pair features [I, I, c_z]
        """
        with trace_range("RFD3/Memory/ChunkedPairwise/CacheStaticProjections"):
            self._sl_cached = self.process_single_l(C_L)  # [L, c_atompair]
            self._sm_cached = self.process_single_m(C_L)  # [L, c_atompair]
            self._Z_proc_cached = self.process_z(Z_init_II)  # [I, I, c_atompair]

    def _forward_chunked_compiled(
        self,
        f: dict,
        indices: torch.Tensor,
        C_L: torch.Tensor,
        Z_init_II: torch.Tensor,
        tok_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled implementation — vectorized ops fused via torch.compile."""
        with trace_range("RFD3/Memory/ChunkedPairwise/ForwardChunkedCompiled"):
            # --- Prepare inputs ---
            if C_L.dim() == 2:
                C_L = C_L.unsqueeze(0)

            has_motif = self.motif_pos_embedder is not None and "motif_pos" in f
            has_ref = self.ref_pos_embedder is not None and "ref_pos" in f
            B, L, k = indices.shape
            device = indices.device
            dummy3 = torch.zeros(1, 3, device=device, dtype=C_L.dtype)
            dummy1 = torch.zeros(1, dtype=torch.bool, device=device)

            # Motif args
            motif_pos = f.get("motif_pos", dummy3) if has_motif else dummy3
            is_motif = f.get("is_motif_atom_with_fixed_coord", dummy1) if has_motif else dummy1
            if has_motif:
                motif_out_w = self.motif_pos_embedder.output_proj.weight.t()       # [2*n_freqs, c_atompair]
                motif_vm_w = self.motif_pos_embedder.process_valid_mask.weight.t()  # [1, c_atompair]
                motif_n_freqs = self.motif_pos_embedder.n_freqs
            else:
                motif_out_w = torch.zeros(1, self.c_atompair, device=device, dtype=C_L.dtype)
                motif_vm_w = torch.zeros(1, self.c_atompair, device=device, dtype=C_L.dtype)
                motif_n_freqs = 1

            # Ref args
            ref_pos = f.get("ref_pos", dummy3) if has_ref else dummy3
            ref_space_uid = f.get("ref_space_uid", dummy1.long()) if has_ref else dummy1.long()
            is_motif_seq = f.get("is_motif_atom_with_fixed_seq", dummy1) if has_ref else dummy1
            if has_ref:
                ref_embed_frame = self.ref_pos_embedder.embed_frame
                ref_d_w = self.ref_pos_embedder.process_d.weight.t() if ref_embed_frame else torch.zeros(3, self.c_atompair, device=device, dtype=C_L.dtype)
                ref_inv_w = self.ref_pos_embedder.process_inverse_dist.weight.t()
                ref_vm_w = self.ref_pos_embedder.process_valid_mask.weight.t()
            else:
                ref_embed_frame = True
                ref_d_w = torch.zeros(3, self.c_atompair, device=device, dtype=C_L.dtype)
                ref_inv_w = torch.zeros(1, self.c_atompair, device=device, dtype=C_L.dtype)
                ref_vm_w = torch.zeros(1, self.c_atompair, device=device, dtype=C_L.dtype)

            # Single feature terms — ensure they're always populated
            if self._sl_cached is not None:
                sl_cached = self._sl_cached
                sm_cached = self._sm_cached
            else:
                sl_cached = self.process_single_l(C_L.squeeze(0))   # [L, c_atompair]
                sm_cached = self.process_single_m(C_L.squeeze(0))   # [L, c_atompair]

            # Token pair features
            if self._Z_proc_cached is not None:
                Z_processed = self._Z_proc_cached
            else:
                Z_processed = self.process_z(Z_init_II)

            # --- Call core compute ---
            P_LL_sparse = _get_compiled_compute()(
                indices, C_L, self.c_atompair,
                motif_pos, is_motif, has_motif,
                motif_out_w, motif_vm_w, motif_n_freqs,
                ref_pos, ref_space_uid, is_motif_seq, has_ref,
                ref_embed_frame, ref_d_w, ref_inv_w, ref_vm_w,
                sl_cached, sm_cached,
                tok_idx, Z_processed,
            )

            # --- 5. Final pair MLP (kept outside compile for tracing) ---
            with trace_range("RFD3/Memory/ChunkedPairwise/FinalPairMLP"):
                P_LL_sparse = P_LL_sparse + self.pair_mlp(P_LL_sparse)

            return P_LL_sparse.contiguous()

    def forward_chunked(
        self,
        f: dict,
        indices: torch.Tensor,  # [B, L, k] - sparse attention indices
        C_L: torch.Tensor,  # [L, c_token] or [B, L, c_token] - atom features
        Z_init_II: torch.Tensor,  # [I, I, c_z] - token pair features
        tok_idx: torch.Tensor,  # [L] - atom to token mapping
        use_loop: bool = True,
    ) -> torch.Tensor:
        """
        Compute P_LL only for the pairs specified by attention indices.

        When cache_static_projections() has been called beforehand, the three MLP
        terms (process_single_l, process_single_m, process_z) are replaced with
        free tensor index operations, since those projections are identical across
        all diffusion steps.

        Args:
            f:         Feature dictionary (motif_pos, ref_pos, etc.)
            indices:   Sparse attention indices [B, L, k]
            C_L:       Atom-level features [L, c_token] or [B, L, c_token]
            Z_init_II: Token-level pair features [I, I, c_z]
            tok_idx:   Atom-to-token mapping [L]
            use_loop:  If True (default), use original looped implementation.
                       If False, use torch.compile'd implementation.

        Returns:
            P_LL_sparse: Sparse pairwise features [B, L, k, c_atompair]
        """
        if not use_loop:
            return self._forward_chunked_compiled(f, indices, C_L, Z_init_II, tok_idx)

        with trace_range("RFD3/Memory/ChunkedPairwise/ForwardChunked"):
            B, L, k = indices.shape
            device = indices.device

            # Initialize sparse P_LL
            P_LL_sparse = torch.zeros(
                B, L, k, self.c_atompair, device=device, dtype=C_L.dtype
            )

            # Handle both batched and non-batched C_L
            if C_L.dim() == 2:  # [L, c_token] - add batch dimension
                C_L = C_L.unsqueeze(0)  # [1, L, c_token]
            # Add bounds checking to prevent index errors
            L_max = C_L.shape[1]
            valid_indices = torch.clamp(
                indices, 0, L_max - 1
            )  # Clamp indices to valid range

            # Ensure indices have the right shape for gathering
            if valid_indices.dim() == 2:  # [L, k] - add batch dimension
                valid_indices = valid_indices.unsqueeze(0).expand(
                    C_L.shape[0], -1, -1
                )  # [B, L, k]

            with trace_range("RFD3/Memory/ChunkedPairwise/MotifEmbedding"):
                # 1. Motif position embedding (if exists)
                if self.motif_pos_embedder is not None and "motif_pos" in f:
                    motif_pos = f["motif_pos"]  # [L, 3]
                    is_motif = f["is_motif_atom_with_fixed_coord"]  # [L]
                    is_motif_idx = torch.where(is_motif)[0]
                    # For each query position
                    for l in is_motif_idx:
                        key_indices = valid_indices[:, l, :]  # [B, k] - use clamped indices
                        key_pos = motif_pos[key_indices]  # [B, k, 3]
                        query_pos = motif_pos[l].unsqueeze(0).expand(B, -1)  # [B, 3]

                        # Valid mask: both query and keys must be motif
                        key_is_motif = is_motif[key_indices]  # [B, k]
                        valid_mask = key_is_motif.unsqueeze(-1).float()  # [B, k, 1]

                        if valid_mask.sum() > 0:
                            motif_pairs = self.motif_pos_embedder.compute_pairs_chunked(
                                query_pos, key_pos, valid_mask
                            )
                            P_LL_sparse[:, l, :, :] += motif_pairs

            with trace_range("RFD3/Memory/ChunkedPairwise/ReferenceEmbedding"):
                # 2. Reference position embedding (if exists)
                if self.ref_pos_embedder is not None and "ref_pos" in f:
                    ref_pos = f["ref_pos"]  # [L, 3]
                    ref_space_uid = f["ref_space_uid"]  # [L]
                    is_motif_seq = f["is_motif_atom_with_fixed_seq"]  # [L]
                    is_motif_seq_idx = torch.where(is_motif_seq)[0]
                    for l in is_motif_seq_idx:
                        key_indices = valid_indices[:, l, :]  # [B, k] - use clamped indices
                        key_pos = ref_pos[key_indices]  # [B, k, 3]
                        query_pos = ref_pos[l].unsqueeze(0).expand(B, -1)  # [B, 3]

                        # Valid mask: same token and both have sequence
                        key_space_uid = ref_space_uid[key_indices]  # [B, k]
                        key_is_motif_seq = is_motif_seq[key_indices]  # [B, k]

                        same_token = key_space_uid == ref_space_uid[l]  # [B, k]
                        valid_mask = (
                            (same_token & key_is_motif_seq).unsqueeze(-1).float()
                        )  # [B, k, 1]

                        if valid_mask.sum() > 0:
                            ref_pairs = self.ref_pos_embedder.compute_pairs_chunked(
                                query_pos, key_pos, valid_mask
                            )
                            P_LL_sparse[:, l, :, :] += ref_pairs

            with trace_range("RFD3/Memory/ChunkedPairwise/SingleFeatureTerms"):
                # 3. Single embedding terms
                if self._sl_cached is not None:
                    # Fast path: MLP already run at tokenisation — just index into the result.
                    # sl_cached [L, c_atompair]: query atom l always maps to row l.
                    single_l = self._sl_cached.unsqueeze(0).unsqueeze(2).expand(B, -1, k, -1)
                    # sm_cached [L, c_atompair]: key atoms are given by valid_indices [B, L, k].
                    single_m = self._sm_cached[valid_indices]  # [B, L, k, c_atompair]
                else:
                    # Slow path (no cache): run the MLPs over the raw atom features.
                    if C_L.shape[0] != B:
                        C_L = C_L.expand(B, -1, -1)  # [B, L, c_token]
                    C_L_queries = C_L.unsqueeze(2).expand(-1, -1, k, -1)  # [B, L, k, c_token]
                    C_L_keys = torch.gather(
                        C_L_queries,
                        1,
                        valid_indices.unsqueeze(-1).expand(-1, -1, -1, C_L.shape[-1]),
                    )  # [B, L, k, c_token]
                    single_l = self.process_single_l(C_L_queries)  # [B, L, k, c_atompair]
                    single_m = self.process_single_m(C_L_keys)  # [B, L, k, c_atompair]
                P_LL_sparse += single_l + single_m

            with trace_range("RFD3/Memory/ChunkedPairwise/TokenPairTerms"):
                # 4. Token pair features Z_init_II
                # Map atoms to tokens and gather token pair features.
                if tok_idx.dim() == 1:  # [L] - add batch dimension for consistency
                    tok_idx_expanded = tok_idx.unsqueeze(0)  # [1, L]
                else:
                    tok_idx_expanded = tok_idx

                if tok_idx_expanded.shape[0] != B:
                    tok_idx_expanded = tok_idx_expanded.expand(B, -1)  # [B, L]
                tok_queries = tok_idx_expanded.unsqueeze(2).expand(-1, -1, k)  # [B, L, k]
                tok_keys = torch.gather(tok_queries, 1, valid_indices)  # [B, L, k]

                if self._Z_proc_cached is not None:
                    # Fast path: process_z already run at tokenisation.
                    Z_processed = self._Z_proc_cached  # [I, I, c_atompair]
                else:
                    # Slow path: run the MLP over the token-pair matrix.
                    Z_processed = self.process_z(Z_init_II)  # [I, I, c_atompair]

                I_z, I_z2 = Z_processed.shape[:2]
                Z_pairs_processed = torch.zeros(
                    B, L, k, self.c_atompair, device=device, dtype=Z_processed.dtype
                )
                for b in range(B):
                    tq = torch.clamp(tok_queries[b], 0, I_z - 1)  # [L, k]
                    tk = torch.clamp(tok_keys[b], 0, I_z2 - 1)  # [L, k]
                    Z_pairs_processed[b] = Z_processed[tq, tk]  # [L, k, c_atompair]

                P_LL_sparse += Z_pairs_processed

            with trace_range("RFD3/Memory/ChunkedPairwise/FinalPairMLP"):
                # 5. Final MLP - ADD the result, don't replace (to match standard implementation)
                P_LL_sparse = P_LL_sparse + self.pair_mlp(P_LL_sparse)

            return P_LL_sparse.contiguous()


def create_chunked_embedders(
    c_atompair: int, embed_frame: bool = True
) -> ChunkedPairwiseEmbedder:
    """
    Factory function to create chunked pairwise embedder with standard components.
    """
    motif_pos_embedder = ChunkedSinusoidalDistEmbed(
        embedder_instance=SinusoidalDistEmbed(c_atompair, embed_frame)
    )
    ref_pos_embedder = ChunkedPositionPairDistEmbedder(
        embedder_instance=PositionPairDistEmbedder(c_atompair, embed_frame)
    )

    return ChunkedPairwiseEmbedder(
        c_atompair=c_atompair,
        motif_pos_embedder=motif_pos_embedder,
        ref_pos_embedder=ref_pos_embedder,
    )


def test_forward_chunked_correctness():
    """
    Runs old looped version and new vectorized version on identical random
    inputs and asserts max absolute difference < 1e-5.
    Prints: PASS or FAIL with the actual max diff value.
    """
    import torch

    import time

    torch.manual_seed(42)
    B, L, k, c = 1, 256, 32, 16
    I = 8  # number of tokens
    c_token = 128
    c_z = 128
    device = "cpu"

    # Build embedder via factory (creates fresh random weights)
    embedder = create_chunked_embedders(c_atompair=c, embed_frame=True)

    # Build dummy inputs
    indices = torch.randint(0, L, (B, L, k), device=device)
    C_L = torch.randn(L, c_token, device=device)
    Z_init_II = torch.randn(I, I, c_z, device=device)
    tok_idx = torch.randint(0, I, (L,), device=device)

    # ~40% of atoms are motif
    is_motif = torch.zeros(L, dtype=torch.bool, device=device)
    is_motif[torch.randperm(L)[: L // 3]] = True
    is_motif_seq = torch.zeros(L, dtype=torch.bool, device=device)
    is_motif_seq[torch.randperm(L)[: L // 3]] = True

    f = {
        "motif_pos": torch.randn(L, 3, device=device),
        "ref_pos": torch.randn(L, 3, device=device),
        "is_motif_atom_with_fixed_coord": is_motif,
        "is_motif_atom_with_fixed_seq": is_motif_seq,
        "ref_space_uid": tok_idx.clone(),  # reuse tok_idx as space uid for simplicity
    }

    # --- Correctness check ---
    with torch.no_grad():
        out_loop = embedder.forward_chunked(
            f, indices, C_L, Z_init_II, tok_idx, use_loop=True
        )
        out_compiled = embedder.forward_chunked(
            f, indices, C_L, Z_init_II, tok_idx, use_loop=False
        )

    max_diff = (out_loop - out_compiled).abs().max().item()
    if max_diff < 1e-5:
        print(f"PASS — compiled max diff = {max_diff:.2e}")
    else:
        print(f"FAIL — compiled max diff = {max_diff:.2e}")

    assert max_diff < 1e-5, f"Compiled outputs differ by {max_diff:.2e}"

    # --- Speed benchmark ---
    def _bench(embed, feat, idx, cl, zii, ti, use_loop, n_warmup=3, n_iters=20):
        with torch.no_grad():
            for _ in range(n_warmup):
                embed.forward_chunked(feat, idx, cl, zii, ti, use_loop=use_loop)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                embed.forward_chunked(feat, idx, cl, zii, ti, use_loop=use_loop)
        return (time.perf_counter() - t0) / n_iters

    for bench_L in [L, 2048]:
        # rebuild inputs at bench_L size
        torch.manual_seed(42)
        b_indices = torch.randint(0, bench_L, (B, bench_L, k), device=device)
        b_C_L = torch.randn(bench_L, c_token, device=device)
        b_tok_idx = torch.randint(0, I, (bench_L,), device=device)
        b_is_motif = torch.zeros(bench_L, dtype=torch.bool, device=device)
        b_is_motif[torch.randperm(bench_L)[: bench_L // 3]] = True
        b_is_motif_seq = torch.zeros(bench_L, dtype=torch.bool, device=device)
        b_is_motif_seq[torch.randperm(bench_L)[: bench_L // 3]] = True
        b_f = {
            "motif_pos": torch.randn(bench_L, 3, device=device),
            "ref_pos": torch.randn(bench_L, 3, device=device),
            "is_motif_atom_with_fixed_coord": b_is_motif,
            "is_motif_atom_with_fixed_seq": b_is_motif_seq,
            "ref_space_uid": b_tok_idx.clone(),
        }
        n_w, n_i = (2, 5) if bench_L >= 2048 else (3, 20)
        t_loop = _bench(embedder, b_f, b_indices, b_C_L, Z_init_II, b_tok_idx, True, n_w, n_i)
        # Extra warmup iterations for torch.compile tracing
        t_compiled = _bench(embedder, b_f, b_indices, b_C_L, Z_init_II, b_tok_idx, False, max(n_w, 3), n_i)
        speedup = t_loop / t_compiled if t_compiled > 0 else float("inf")
        print(f"\n[L={bench_L}]")
        print(f"  Looped:   {t_loop*1000:.2f} ms")
        print(f"  Compiled: {t_compiled*1000:.2f} ms  ({speedup:.1f}x)")

    return max_diff


if __name__ == "__main__":
    test_forward_chunked_correctness()
