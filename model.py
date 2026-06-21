"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    calendar_time_features: torch.Tensor
    seq_gap_buckets: Optional[dict] = None  # {domain: tensor [B, L]}
    # Per-position temporal ids (hour / day-of-week / week-of-year) consumed by
    # ``HistoricalTemporalBiasInjector``. Each value is a sub-dict keyed by
    # ``'hour' | 'day' | 'week'`` mapping to an ``int64`` tensor of shape
    # ``(B, L)``. Optional so existing call sites that don't enable the
    # temporal-bias branch remain wire-compatible.
    seq_history_time_ids: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask supports two semantics under the same interface:
            #   (1) bool tensor: True = attend, False = mask. Legacy callers.
            #   (2) float tensor: additive bias added to softmax logits.
            #       Used by target-aware cross-attention bias to inject a
            #       learnable prior over key positions. The two are unified
            #       here by converting both to a float additive mask of
            #       shape (B, num_heads, Lq, Lk). Padding positions are
            #       filled with a finite large negative (-1e4) instead of
            #       -inf so bf16/fp16 softmax stays numerically stable.
            NEG_LARGE = -1e4
            if attn_mask.dtype == torch.bool:
                bool_attn = attn_mask
                add_mask = torch.zeros_like(bool_attn, dtype=Q.dtype)
                add_mask = add_mask.masked_fill(~bool_attn, NEG_LARGE)
            else:
                add_mask = attn_mask.to(Q.dtype)
                add_mask = torch.nan_to_num(
                    add_mask, nan=0.0, posinf=-NEG_LARGE, neginf=NEG_LARGE)

            # Lift to (B, num_heads, Lq, Lk). Insert dims explicitly so the
            # head axis lands at dim=1; relying on unsqueeze(0) loops would
            # push the batch axis into the head slot and break the expand.
            if add_mask.dim() == 2:
                # (Lq, Lk) position-only mask, e.g. causal mask.
                add_mask = add_mask.unsqueeze(0).unsqueeze(0)
            elif add_mask.dim() == 3:
                # (B, Lq, Lk) or (B, 1, Lk): broadcast over the head axis.
                add_mask = add_mask.unsqueeze(1)
            elif add_mask.dim() != 4:
                raise ValueError(
                    f"attn_mask must be 2D / 3D / 4D, got shape {tuple(attn_mask.shape)}"
                )
            add_mask = add_mask.expand(B, self.num_heads, Lq, Lk)

            if sdpa_attn_mask is not None:
                # sdpa_attn_mask is currently a bool (B, H, Lq, Lk), True=attend.
                add_mask = add_mask.masked_fill(~sdpa_attn_mask, NEG_LARGE)
            sdpa_attn_mask = add_mask  # switch to float additive mask

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_score_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.
            attn_score_bias: Optional (B, L) or (B, Nq, L) float tensor
                added to the cross-attention logits before softmax. Used
                by ``TargetAwareCrossAttentionBias`` to inject a
                target-aware prior over key positions; ``None`` degrades
                to vanilla cross-attention.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        # Lift (B, L) to (B, 1, L) so the bias broadcasts across all Nq
        # query positions; (B, Nq, L) is forwarded as-is.
        attn_mask_for_attn = None
        if attn_score_bias is not None:
            if attn_score_bias.dim() == 2:
                attn_mask_for_attn = attn_score_bias.unsqueeze(1)
            else:
                attn_mask_for_attn = attn_score_bias

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask_for_attn,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class SparseMoEFFN(nn.Module):
    """Sparse top-k MoE FFN for token-wise RankMixer feed-forward routing."""

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        num_experts: int = 4,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * hidden_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * hidden_mult, d_model),
            )
            for _ in range(num_experts)
        ])
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_expert_usage: Optional[torch.Tensor] = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Route each token to top-k experts and combine weighted outputs."""
        batch_size, token_count, hidden_dim = tokens.shape
        flat_tokens = tokens.reshape(-1, hidden_dim)
        router_probabilities = F.softmax(self.router(flat_tokens), dim=-1)
        top_probabilities, top_expert_indices = torch.topk(
            router_probabilities,
            k=self.top_k,
            dim=-1,
        )

        output = torch.zeros_like(flat_tokens)
        for expert_idx, expert in enumerate(self.experts):
            selected_routes = top_expert_indices == expert_idx
            if not bool(selected_routes.any()):
                continue

            token_indices, route_indices = selected_routes.nonzero(as_tuple=True)
            expert_weights = top_probabilities[token_indices, route_indices].unsqueeze(-1)
            expert_output = expert(flat_tokens[token_indices])
            weighted_output = (expert_output * expert_weights).to(dtype=output.dtype)
            output.index_add_(0, token_indices, weighted_output)

        top1_expert_indices = top_expert_indices[:, 0]
        self.last_aux_loss = self._compute_load_balancing_loss(
            router_probabilities,
            top1_expert_indices,
        )
        self.last_expert_usage = self._compute_top1_expert_usage(
            top1_expert_indices,
            dtype=router_probabilities.dtype,
        ).detach()
        return output.view(batch_size, token_count, hidden_dim)

    def _compute_load_balancing_loss(
        self,
        router_probabilities: torch.Tensor,
        top1_expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        tokens_per_expert = self._compute_top1_expert_usage(
            top1_expert_indices,
            dtype=router_probabilities.dtype,
        )
        prob_per_expert = router_probabilities.mean(dim=0)
        return self.num_experts * torch.sum(tokens_per_expert * prob_per_expert)

    def _compute_top1_expert_usage(
        self,
        top1_expert_indices: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expert_counts = torch.bincount(
            top1_expert_indices,
            minlength=self.num_experts,
        ).to(device=top1_expert_indices.device, dtype=dtype)
        return expert_counts / top1_expert_indices.numel()


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full',  # 'full' | 'ffn_only' | 'none'
        rank_moe_enable: bool = False,
        rank_moe_num_experts: int = 4,
        rank_moe_top_k: int = 1,
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode
        self.rank_moe_enable = rank_moe_enable

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        if rank_moe_enable:
            self.moe_ffn = SparseMoEFFN(
                d_model=d_model,
                hidden_mult=hidden_mult,
                dropout=dropout,
                num_experts=rank_moe_num_experts,
                top_k=rank_moe_top_k,
            )
        else:
            self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
            self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
            self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        if self.rank_moe_enable:
            Q_e = self.moe_ffn(x)
        else:
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost

    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        if self.mode == 'none' or not self.rank_moe_enable:
            return None
        return self.moe_ffn.last_aux_loss


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        rank_moe_enable: bool = False,
        rank_moe_num_experts: int = 4,
        rank_moe_top_k: int = 1,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
            rank_moe_enable=rank_moe_enable,
            rank_moe_num_experts=rank_moe_num_experts,
            rank_moe_top_k=rank_moe_top_k,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        target_anchor: Optional[torch.Tensor] = None,
        target_attn_bias_modules: Optional[nn.ModuleList] = None,
        target_attn_bias_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            target_anchor: Optional (B, D) candidate-aware target used by
                ``TargetAwareCrossAttentionBias`` to compute an additive
                cross-attention prior. Owned by ``PCVRHyFormer``.
            target_attn_bias_modules: Optional length-S ``nn.ModuleList``
                of per-domain ``TargetAwareCrossAttentionBias`` instances
                bound to this block.
            target_attn_bias_alpha: Optional (S,) learnable scalar
                tensor scaling the bias per domain (init=0 so the block
                degrades to vanilla cross-attention).

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Build target-aware cross-attention bias after sequence
        # evolution so its length matches the actual K/V tokens consumed
        # by cross-attention. This matters for compressing encoders
        # (e.g. LongerEncoder top-k) where L_i may shrink to top_k.
        use_target_attn_bias = (
            target_anchor is not None
            and target_attn_bias_modules is not None
            and target_attn_bias_alpha is not None
        )
        cross_attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None
        if use_target_attn_bias:
            cross_attn_bias_list = []
            for i in range(S):
                bias_i = target_attn_bias_modules[i](
                    target_anchor, next_seqs[i], next_masks[i])  # (B, L_i')
                bias_i = target_attn_bias_alpha[i] * bias_i
                cross_attn_bias_list.append(bias_i)

        # 3. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            bias_i = (cross_attn_bias_list[i]
                      if cross_attn_bias_list is not None else None)
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
                attn_score_bias=bias_i,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks

    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        return self.mixer.get_moe_aux_loss()


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class IntradayTimeFeatureEncoder(nn.Module):
    """Encodes time features within a single day."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.minute_of_day_embedding = nn.Embedding(1440, d_model)
        self.hour_of_day_embedding = nn.Embedding(24, d_model)
        self.daypart_embedding = nn.Embedding(4, d_model)

    def forward(
        self,
        calendar_time_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        minute_of_day_token = self.minute_of_day_embedding(calendar_time_features[:, 0])
        hour_of_day_token = self.hour_of_day_embedding(calendar_time_features[:, 1])
        daypart_token = self.daypart_embedding(calendar_time_features[:, 8])
        return minute_of_day_token, hour_of_day_token, daypart_token


class WeeklyTimeFeatureEncoder(nn.Module):
    """Encodes weekly time-position features."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.day_of_week_embedding = nn.Embedding(7, d_model)
        self.hour_of_week_embedding = nn.Embedding(168, d_model)
        self.weekend_flag_embedding = nn.Embedding(2, d_model)

    def forward(
        self,
        calendar_time_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        day_of_week_token = self.day_of_week_embedding(calendar_time_features[:, 2])
        hour_of_week_token = self.hour_of_week_embedding(calendar_time_features[:, 3])
        weekend_flag_token = self.weekend_flag_embedding(calendar_time_features[:, 7])
        return day_of_week_token, hour_of_week_token, weekend_flag_token


class AnnualTimeFeatureEncoder(nn.Module):
    """Encodes calendar date features within month and year cycles."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.day_of_month_embedding = nn.Embedding(32, d_model)
        self.month_of_year_embedding = nn.Embedding(13, d_model)
        self.day_of_year_embedding = nn.Embedding(367, d_model)

    def forward(
        self,
        calendar_time_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        day_of_month_token = self.day_of_month_embedding(calendar_time_features[:, 4])
        month_of_year_token = self.month_of_year_embedding(calendar_time_features[:, 5])
        day_of_year_token = self.day_of_year_embedding(calendar_time_features[:, 6])
        return day_of_month_token, month_of_year_token, day_of_year_token


class CalendarTimeFeatureEncoder(nn.Module):
    """Encodes enabled absolute calendar feature groups into one context token."""

    def __init__(
        self,
        d_model: int,
        enable_intraday_calendar_features: bool = True,
        enable_weekly_calendar_features: bool = True,
        enable_annual_calendar_features: bool = True,
        defer_output_projection: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.enable_intraday_calendar_features = enable_intraday_calendar_features
        self.enable_weekly_calendar_features = enable_weekly_calendar_features
        self.enable_annual_calendar_features = enable_annual_calendar_features
        self.enabled_feature_count = 3 * sum([
            enable_intraday_calendar_features,
            enable_weekly_calendar_features,
            enable_annual_calendar_features,
        ])
        if self.enabled_feature_count == 0:
            raise ValueError("At least one calendar feature group must be enabled")

        # minute, hour, day-of-week, hour-of-week, day, month, day-of-year,
        # weekend flag, daypart.
        self.minute_of_day_embedding = nn.Embedding(1440, d_model)
        self.hour_of_day_embedding = nn.Embedding(24, d_model)
        self.day_of_week_embedding = nn.Embedding(7, d_model)
        self.hour_of_week_embedding = nn.Embedding(168, d_model)
        self.day_of_month_embedding = nn.Embedding(32, d_model)
        self.month_of_year_embedding = nn.Embedding(13, d_model)
        self.day_of_year_embedding = nn.Embedding(367, d_model)
        self.weekend_flag_embedding = nn.Embedding(2, d_model)
        self.daypart_embedding = nn.Embedding(4, d_model)
        self.output_projection: Optional[nn.Sequential] = None
        if not defer_output_projection:
            self.build_output_projection()

    def build_output_projection(self) -> None:
        if self.output_projection is not None:
            return
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model * self.enabled_feature_count, self.d_model),
            nn.Dropout(0.1),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )

    def forward(self, calendar_time_features: torch.Tensor) -> torch.Tensor:
        if self.output_projection is None:
            raise RuntimeError("CalendarTimeFeatureEncoder output projection has not been built")
        calendar_token_parts = []
        if self.enable_intraday_calendar_features:
            minute_of_day_token = self.minute_of_day_embedding(calendar_time_features[:, 0])
            hour_of_day_token = self.hour_of_day_embedding(calendar_time_features[:, 1])
            daypart_token = self.daypart_embedding(calendar_time_features[:, 8])
            calendar_token_parts.extend([minute_of_day_token, hour_of_day_token])

        if self.enable_weekly_calendar_features:
            day_of_week_token = self.day_of_week_embedding(calendar_time_features[:, 2])
            hour_of_week_token = self.hour_of_week_embedding(calendar_time_features[:, 3])
            weekend_flag_token = self.weekend_flag_embedding(calendar_time_features[:, 7])
            calendar_token_parts.extend([day_of_week_token, hour_of_week_token])

        if self.enable_annual_calendar_features:
            day_of_month_token = self.day_of_month_embedding(calendar_time_features[:, 4])
            month_of_year_token = self.month_of_year_embedding(calendar_time_features[:, 5])
            day_of_year_token = self.day_of_year_embedding(calendar_time_features[:, 6])
            calendar_token_parts.extend([day_of_month_token, month_of_year_token, day_of_year_token])

        if self.enable_weekly_calendar_features:
            calendar_token_parts.append(weekend_flag_token)
        if self.enable_intraday_calendar_features:
            calendar_token_parts.append(daypart_token)

        calendar_tokens = torch.cat(calendar_token_parts, dim=-1)
        return self.output_projection(calendar_tokens).unsqueeze(1)


class UserDenseFeatureEncoder(nn.Module):
    """Projects dense user features while preserving the current physical layout."""

    sum_embedding_dim = 256
    lmf_embedding_dim = 320
    lmf_embedding_start = 568
    lmf_embedding_end = 888

    def __init__(self, user_dense_dim: int, d_model: int) -> None:
        super().__init__()
        statistical_feature_dim = user_dense_dim - self.sum_embedding_dim - self.lmf_embedding_dim
        self.statistical_feature_projection = nn.Sequential(
            nn.Linear(statistical_feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.sum_embedding_projection = nn.Sequential(
            nn.Linear(self.sum_embedding_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.lmf_embedding_projection = nn.Sequential(
            nn.Linear(self.lmf_embedding_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, user_dense_features: torch.Tensor) -> torch.Tensor:
        sum_embedding_features = user_dense_features[:, :self.sum_embedding_dim]
        statistical_features_before_lmf = user_dense_features[
            :, self.sum_embedding_dim:self.lmf_embedding_start
        ]
        lmf_embedding_features = user_dense_features[:, self.lmf_embedding_start:self.lmf_embedding_end]
        statistical_features_after_lmf = user_dense_features[:, self.lmf_embedding_end:]
        statistical_features = torch.cat([
            statistical_features_before_lmf,
            statistical_features_after_lmf,
        ], dim=1)

        statistical_token = self.statistical_feature_projection(statistical_features)
        sum_embedding_token = self.sum_embedding_projection(sum_embedding_features)
        lmf_embedding_token = self.lmf_embedding_projection(lmf_embedding_features)
        return F.silu(statistical_token + sum_embedding_token + lmf_embedding_token).unsqueeze(1)


class ContextInteractionFeatureBuilder(nn.Module):
    """Builds four-way interaction features between two context vectors."""

    def forward(
        self,
        primary_context: torch.Tensor,
        secondary_context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([
            primary_context,
            secondary_context,
            primary_context * secondary_context,
            torch.abs(primary_context - secondary_context),
        ], dim=-1)


class TokenListMeanPooler(nn.Module):
    """Pools a list of token groups into one context vector."""

    def forward(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tokens, dim=1).mean(dim=1)


class TargetAnchorProjector(nn.Module):
    """Builds a candidate-aware query anchor from static user/item tokens."""

    def __init__(self, d_model: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.token_pooler = TokenListMeanPooler()
        self.interaction_feature_builder = ContextInteractionFeatureBuilder()
        self.projection = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        user_tokens: List[torch.Tensor],
        item_tokens: List[torch.Tensor],
    ) -> torch.Tensor:
        user_context = self.token_pooler(user_tokens)
        item_context = self.token_pooler(item_tokens)
        target_anchor_features = self.interaction_feature_builder(item_context, user_context)
        return self.projection(target_anchor_features)


class TargetQueryProjector(nn.Module):
    """Projects the target anchor into the shared sequence-query space."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, target_anchor: torch.Tensor) -> torch.Tensor:
        return self.projection(target_anchor).unsqueeze(1)


class SequenceProjectionBank(nn.Module):
    """Stores one projection layer per sequence domain."""

    def __init__(self, d_model: int, num_sequences: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_sequences)
        ])

    def forward(self, sequence_tokens: torch.Tensor, domain_idx: int) -> torch.Tensor:
        return self.layers[domain_idx](sequence_tokens)


class SequenceContextNormBank(nn.Module):
    """Stores one context normalization layer per sequence domain."""

    def __init__(self, d_model: int, num_sequences: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_sequences)
        ])

    def forward(self, domain_context: torch.Tensor, domain_idx: int) -> torch.Tensor:
        return self.layers[domain_idx](domain_context)


class TopKSequenceAttention(nn.Module):
    """Computes target-aware attention over one sequence domain."""

    def __init__(self, d_model: int, top_k: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        sequence_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        activation_scores = (query * keys).sum(dim=-1) / math.sqrt(self.d_model)  # (B, L)
        activation_scores = activation_scores.masked_fill(sequence_padding_mask, float('-inf'))

        if self.top_k > 0 and self.top_k < activation_scores.shape[-1]:
            top_values, top_indices = torch.topk(activation_scores, k=self.top_k, dim=-1)
            filtered_scores = activation_scores.new_full(activation_scores.shape, float('-inf'))
            activation_scores = filtered_scores.scatter(-1, top_indices, top_values)

        attention_weights = F.softmax(activation_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        return torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)


class DomainGateScorer(nn.Module):
    """Scores how much each sequence domain should contribute."""

    def __init__(self, d_model: int, hidden_mult: int = 2) -> None:
        super().__init__()
        gate_hidden = max(d_model, d_model * hidden_mult)
        self.interaction_feature_builder = ContextInteractionFeatureBuilder()
        self.network = nn.Sequential(
            nn.Linear(4 * d_model, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, target_anchor: torch.Tensor, domain_context: torch.Tensor) -> torch.Tensor:
        domain_gate_features = self.interaction_feature_builder(target_anchor, domain_context)
        return self.network(domain_gate_features)


class DomainContextAggregator(nn.Module):
    """Aggregates domain contexts with softmax-normalized domain weights."""

    def forward(
        self,
        domain_contexts: List[torch.Tensor],
        domain_logits: List[torch.Tensor],
    ) -> torch.Tensor:
        contexts = torch.stack(domain_contexts, dim=1)  # (B, S, D)
        gate_logits = torch.cat(domain_logits, dim=-1)  # (B, S)
        domain_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)
        return (contexts * domain_weights).sum(dim=1)


class TargetResidualProjector(nn.Module):
    """Projects matched target/sequence context into a residual delta."""

    def __init__(self, d_model: int, hidden_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        delta_hidden = max(d_model, d_model * hidden_mult)
        self.interaction_feature_builder = ContextInteractionFeatureBuilder()
        self.projection = nn.Sequential(
            nn.Linear(4 * d_model, delta_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden, d_model),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, target_anchor: torch.Tensor, matched_context: torch.Tensor) -> torch.Tensor:
        residual_features = self.interaction_feature_builder(target_anchor, matched_context)
        return self.projection(residual_features)


class TargetAwareSequenceActivation(nn.Module):
    """Retrieves candidate-relevant history tokens via target-aware attention.
    Returns a residual delta for the main network.
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        top_k: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.top_k = top_k

        self.target_query_projection = nn.Linear(d_model, d_model)
        self.domain_key_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_sequences)
        ])
        self.domain_value_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_sequences)
        ])
        self.domain_context_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_sequences)
        ])

        gate_hidden = max(d_model, d_model * hidden_mult)
        self.domain_gate_network = nn.Sequential(
            nn.Linear(4 * d_model, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

        delta_hidden = max(d_model, d_model * hidden_mult)
        self.residual_delta_network = nn.Sequential(
            nn.Linear(4 * d_model, delta_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden, d_model),
        )
        nn.init.zeros_(self.residual_delta_network[-1].weight)
        nn.init.zeros_(self.residual_delta_network[-1].bias)

    def _activate_sequence_domain(
        self,
        target_anchor: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_padding_mask: torch.Tensor,
        domain_idx: int,
    ) -> torch.Tensor:
        query = self.target_query_projection(target_anchor).unsqueeze(1)
        keys = self.domain_key_projections[domain_idx](sequence_tokens)
        values = self.domain_value_projections[domain_idx](sequence_tokens)

        activation_scores = (query * keys).sum(dim=-1) / math.sqrt(self.d_model)
        activation_scores = activation_scores.masked_fill(sequence_padding_mask, float('-inf'))

        if self.top_k > 0 and self.top_k < activation_scores.shape[-1]:
            top_values, top_indices = torch.topk(activation_scores, k=self.top_k, dim=-1)
            filtered_scores = activation_scores.new_full(activation_scores.shape, float('-inf'))
            activation_scores = filtered_scores.scatter(-1, top_indices, top_values)

        attention_weights = F.softmax(activation_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        domain_context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        return self.domain_context_norms[domain_idx](domain_context)

    def forward(
        self,
        target_anchor: torch.Tensor,
        sequence_token_list: list,
        sequence_padding_masks: list,
    ) -> torch.Tensor:
        domain_contexts = []
        domain_logits = []

        for i in range(self.num_sequences):
            domain_context = self._activate_sequence_domain(
                target_anchor, sequence_token_list[i], sequence_padding_masks[i], i)
            domain_contexts.append(domain_context)
            gate_input = torch.cat([
                target_anchor,
                domain_context,
                target_anchor * domain_context,
                torch.abs(target_anchor - domain_context),
            ], dim=-1)
            domain_logits.append(self.domain_gate_network(gate_input))

        contexts = torch.stack(domain_contexts, dim=1)
        gate_logits = torch.cat(domain_logits, dim=-1)
        domain_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)
        matched_context = (contexts * domain_weights).sum(dim=1)

        delta_input = torch.cat([
            target_anchor,
            matched_context,
            target_anchor * matched_context,
            torch.abs(target_anchor - matched_context),
        ], dim=-1)
        return self.residual_delta_network(delta_input)


class TargetAwareSequenceReweighter(nn.Module):
    """DIN-style per-token reweighting of sequence tokens.

    For each domain, computes an importance gate per history position from
    the candidate target anchor and re-weights the original sequence tokens
    in-place. Output preserves the original sequence length and layout so
    that downstream modules (query generator, HyFormer blocks, optional
    additive DIN branch) can consume it transparently.

    Gate semantics: gate = 2 * sigmoid(MLP([t, h, t*h, |t-h|])) ∈ (0, 2),
    centered around 1.0 thanks to zero-init of the final gate layer, so the
    module starts as an identity mapping. Padding positions are zeroed.
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences

        gate_hidden = max(d_model, d_model * hidden_mult)
        self.target_query_projection = nn.Linear(d_model, d_model)
        self.domain_key_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_sequences)
        ])
        self.domain_gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * d_model, gate_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 1),
            )
            for _ in range(num_sequences)
        ])
        # Zero-init final gate layer so initial gate=1.0 (identity mapping)
        for gate_net in self.domain_gate_networks:
            nn.init.zeros_(gate_net[-1].weight)
            nn.init.zeros_(gate_net[-1].bias)

    def forward(
        self,
        target_anchor: torch.Tensor,
        sequence_token_list: list,
        sequence_padding_masks: list,
    ) -> list:
        """Re-weights every sequence domain by per-token target-aware gates.

        Args:
            target_anchor: (B, D) candidate-aware target representation.
            sequence_token_list: list of (B, L_i, D) tensors, one per domain.
            sequence_padding_masks: list of (B, L_i) bool masks, True at
                padding positions.

        Returns:
            List of (B, L_i, D) re-weighted tensors. Padding positions are
            forced to zero so downstream attention/pooling stays consistent.
        """
        target_query = self.target_query_projection(target_anchor)  # (B, D)
        reweighted_list = []
        for i in range(self.num_sequences):
            seq_tokens = sequence_token_list[i]  # (B, L, D)
            padding_mask = sequence_padding_masks[i]  # (B, L)
            B, L, D = seq_tokens.shape

            keys = self.domain_key_projections[i](seq_tokens)  # (B, L, D)
            target_expanded = target_query.unsqueeze(1).expand(B, L, D)  # (B, L, D)

            gate_features = torch.cat([
                target_expanded,
                keys,
                target_expanded * keys,
                torch.abs(target_expanded - keys),
            ], dim=-1)  # (B, L, 4D)

            gate_logits = self.domain_gate_networks[i](gate_features).squeeze(-1)  # (B, L)
            gate = 2.0 * torch.sigmoid(gate_logits)  # (B, L), centered around 1.0

            valid_mask = (~padding_mask).to(gate.dtype)  # (B, L)
            gate = gate * valid_mask

            reweighted = seq_tokens * gate.unsqueeze(-1)  # (B, L, D)
            reweighted_list.append(reweighted)
        return reweighted_list


class TargetAwareCrossAttentionBias(nn.Module):
    """Target-aware additive bias for HyFormer cross-attention.

    Each (HyFormer block, sequence domain) pair owns one independent
    instance. The module emits a (B, L) score from the candidate target
    anchor and the current sequence tokens, which is added to the
    cross-attention logits before softmax. Combined with a per-(block,
    domain) learnable scalar alpha initialised to 0 (see
    ``PCVRHyFormer.target_attn_bias_alpha``), the block degrades to a
    vanilla cross-attention at init; the model gradually amplifies the
    prior on layers/domains where it helps.

    Decoupled from existing Q/K/V projections inside
    ``CrossAttention`` so the prior is an *external* injection rather
    than a re-derivation from the same projections. Decoupled from
    ``TargetAwareSequenceActivation`` for the same reason: that branch
    consumes the static sequence tokens once after the HyFormer stack,
    while this module fires per block on the evolved sequence tokens.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.target_query_projection = nn.Linear(d_model, d_model, bias=False)
        self.sequence_key_projection = nn.Linear(d_model, d_model, bias=False)
        # Light LayerNorm prevents per-block scale drift of the inputs from
        # contaminating the bias magnitudes across the HyFormer stack.
        self.target_query_norm = nn.LayerNorm(d_model)
        self.sequence_key_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._inv_sqrt_d = float(d_model) ** -0.5

    def forward(
        self,
        target_anchor: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes an additive bias over the key positions.

        Args:
            target_anchor: (B, D) candidate-aware target representation.
            sequence_tokens: (B, L, D) sequence tokens consumed by the
                downstream cross-attention as keys / values.
            sequence_padding_mask: (B, L) bool, True at padding positions.

        Returns:
            (B, L) float tensor added to cross-attention logits before
            softmax. Padding positions receive a finite large negative
            (-1e4) so they are effectively suppressed; fully padded rows
            are forced to all-zero to avoid pushing softmax into NaN
            after a row-wide mask.
        """
        # (B, D) -> (B, 1, D)
        target_query = self.target_query_projection(
            self.target_query_norm(target_anchor)).unsqueeze(1)
        sequence_keys = self.sequence_key_projection(
            self.sequence_key_norm(sequence_tokens))  # (B, L, D)
        bias = (target_query * sequence_keys).sum(dim=-1) * self._inv_sqrt_d
        bias = self.dropout(bias)

        # -1e4 is representable in bf16 with exp(-1e4) ~ 0, so it behaves
        # like -inf for softmax without producing NaN under low precision.
        NEG_LARGE = -1e4
        bias = bias.masked_fill(sequence_padding_mask, NEG_LARGE)
        all_padded = sequence_padding_mask.all(dim=-1, keepdim=True)  # (B, 1)
        bias = torch.where(all_padded, torch.zeros_like(bias), bias)
        return bias


class HistoricalTemporalBiasInjector(nn.Module):
    """Adds an MLP-projected temporal residual onto every history token.

    Implements the *historical sequence time-perception* variant of the
    Time-Aware Gated Attention scheme (variant 1 in the paper):

    .. math::

        x_{i\\text{-}ta} = x_i + \\text{MLP}\\big(\\text{Concat}(x_i^{Hour}, x_i^{Day}, x_i^{Week}, \\dots)\\big)

    Although the figure in the paper highlights *Hour / Day / Week*, the
    underlying reference implementation actually fuses **eight** calendar
    fields (month-of-year, week-of-year, day-of-year, week-of-month,
    day-of-week, hour-of-day, weekday/weekend flag, day-part time slot).
    We keep the full 8-field fusion here to stay faithful to that reference
    implementation.

    For each field we keep an ``nn.Embedding`` lookup as well as a learned
    linear projection of a ``(sin, cos)`` cyclic encoding so that adjacent
    integer ids stay close in feature space. All field embeddings are
    concatenated and pushed through a small ``Linear → GELU → Linear`` block
    whose final layer is zero-initialised; this guarantees that the residual
    contribution starts at exactly zero and never destabilises the base token
    distribution during the first few optimiser steps.

    The forward pass tolerates ``temporal_ids=None`` (no-op) so that callers
    can wire the module in unconditionally without paying any cost when the
    feature is disabled.
    """

    # Each entry is ``(num_embeddings_including_padding_slot, cycle_period)``.
    # The padding slot at index 0 is reserved for masked / missing positions.
    # ``num_embeddings`` is therefore ``max_id + 1`` so 1-based ids never
    # index out of range.
    _TEMPORAL_FIELD_CONFIG: Dict[str, Tuple[int, int]] = {
        'month_of_year':  (13, 12),    # 1~12, period=12
        'week_of_year':   (54, 53),    # 1~53, period=53
        'day_of_year':    (367, 366),  # 1~366, period=366
        'week_of_month':  (6, 5),      # 1~5, period=5
        'day_of_week':    (8, 7),      # 1~7, period=7 (Mon=1..Sun=7)
        'hour_of_day':    (25, 24),    # 1~24, period=24
        'is_weekend':     (3, 2),      # 1=weekday, 2=weekend, period=2
        'time_period':    (8, 7),      # 1~7 daypart bins, period=7
    }

    def __init__(self, d_model: int, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.residual_scale = residual_scale
        self._field_names: Tuple[str, ...] = tuple(self._TEMPORAL_FIELD_CONFIG.keys())
        self._field_periods: Dict[str, int] = {
            name: period for name, (_, period) in self._TEMPORAL_FIELD_CONFIG.items()
        }

        self.id_lookups = nn.ModuleDict({
            name: nn.Embedding(vocab_with_pad, d_model, padding_idx=0)
            for name, (vocab_with_pad, _) in self._TEMPORAL_FIELD_CONFIG.items()
        })
        self.cyclic_projections = nn.ModuleDict({
            name: nn.Linear(2, d_model, bias=False)
            for name in self._field_names
        })
        self.bias_projector = nn.Sequential(
            nn.Linear(len(self._field_names) * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for emb in self.id_lookups.values():
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
        # Zero-init the final projection so the residual starts as a no-op.
        nn.init.zeros_(self.bias_projector[-1].weight)
        nn.init.zeros_(self.bias_projector[-1].bias)

    def forward(
        self,
        base_tokens: torch.Tensor,
        temporal_ids: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Return ``base_tokens`` with the temporal bias residual added in.

        Args:
            base_tokens: ``(B, L, d_model)`` sequence-token embeddings to bias.
            temporal_ids: optional mapping ``{field_name: (B, L) int64}`` whose
                keys must come from :attr:`_TEMPORAL_FIELD_CONFIG`. ``None``
                (or a dict with missing keys) yields a clean no-op so this
                module can be safely wired-in unconditionally.

        Returns:
            ``(B, L, d_model)`` tensor with the residual applied.
        """
        if temporal_ids is None:
            return base_tokens

        B, L, _ = base_tokens.shape
        per_field_embeddings: List[torch.Tensor] = []
        for name in self._field_names:
            ids = temporal_ids.get(name)
            if ids is None:
                per_field_embeddings.append(
                    base_tokens.new_zeros(B, L, self.d_model)
                )
                continue
            # Discrete lookup
            id_repr = self.id_lookups[name](ids)  # (B, L, d_model)
            # Cyclic sin/cos branch
            period = self._field_periods[name]
            angle = ids.to(base_tokens.dtype) * (2.0 * math.pi / period)
            cyc_encoding = torch.stack(
                [torch.sin(angle), torch.cos(angle)], dim=-1
            )  # (B, L, 2)
            cyc_repr = self.cyclic_projections[name](cyc_encoding)
            # Zero-out padding positions so the cyclic branch never leaks
            # information for masked items.
            valid_pos = (ids != 0).unsqueeze(-1).to(cyc_repr.dtype)
            cyc_repr = cyc_repr * valid_pos
            per_field_embeddings.append(id_repr + cyc_repr)

        concat_temporal_emb = torch.cat(per_field_embeddings, dim=-1)
        temporal_bias_residual = self.bias_projector(concat_temporal_emb)
        if self.residual_scale != 1.0:
            temporal_bias_residual = temporal_bias_residual * self.residual_scale
        return base_tokens + temporal_bias_residual


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        num_gap_buckets: int = 0,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        enable_intraday_calendar_features: bool = True,
        enable_weekly_calendar_features: bool = True,
        enable_annual_calendar_features: bool = True,
        seqContextwakeup: bool = True,
        enable_target_sequence_activation: Optional[bool] = None,
        target_activation_top_k: int = 32,
        target_activation_hidden_mult: int = 2,
        enable_seq_reweight: bool = False,
        seq_reweight_hidden_mult: int = 2,
        # Target-aware additive cross-attention bias inside every
        # HyFormer block. One TargetAwareCrossAttentionBias per
        # (block, domain) plus a (num_hyformer_blocks, num_sequences)
        # learnable scalar alpha initialised to 0 so the network
        # degrades to vanilla cross-attention at init.
        bias_attention_up: bool = False,
        target_attn_bias_dropout: float = 0.0,
        rank_moe_enable: bool = False,
        rank_moe_num_experts: int = 4,
        rank_moe_top_k: int = 1,
        # LMF gating branch (uses the LMF user-CVR embedding slice of
        # ``user_dense_feats`` to non-linearly gate all ns_tokens; the
        # resulting vector is added as a small residual to ``output``.)
        use_lmf_gating_branch: bool = False,
        lmf_gating_offset: int = 568,
        lmf_gating_length: int = 320,
        lmf_gating_dropout_rate: float = 0.1,
        lmf_gating_residual_scale: float = 0.05,
        lmf_gating_max_norm: float = 100.0,
        # Historical sequence time-perception (per-step Hour/Day/Week residual)
        enable_history_time_bias: bool = False,
        history_time_bias_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.num_gap_buckets = num_gap_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        if enable_target_sequence_activation is not None:
            seqContextwakeup = enable_target_sequence_activation
        self.seqContextwakeup = seqContextwakeup
        self.enable_target_sequence_activation = seqContextwakeup
        self.enable_seq_reweight = enable_seq_reweight
        self.bias_attention_up = bool(bias_attention_up)
        self.target_attn_bias_dropout = float(target_attn_bias_dropout)
        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,

            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_feature_encoder = UserDenseFeatureEncoder(
                user_dense_dim=user_dense_dim,
                d_model=d_model,
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== LMF gating branch ==================
        # Uses the LMF user-CVR embedding slice of ``user_dense_feats`` to build
        # a per-channel sigmoid gate over all ns_tokens, projects the gated
        # ns_tokens back to ``d_model``, and adds it as a small residual to
        # ``output`` before the classifier. The branch does NOT add new
        # ns_tokens, so ``num_ns`` and the ``d_model % T == 0`` constraint are
        # unaffected.
        self.use_lmf_gating_branch = use_lmf_gating_branch
        self.lmf_gating_offset = lmf_gating_offset
        self.lmf_gating_length = lmf_gating_length
        self.lmf_gating_dropout_rate = lmf_gating_dropout_rate
        self.lmf_gating_residual_scale = lmf_gating_residual_scale
        self.lmf_gating_max_norm = lmf_gating_max_norm
        if self.use_lmf_gating_branch:
            if not self.has_user_dense:
                raise ValueError(
                    "use_lmf_gating_branch=True requires user_dense_dim > 0")
            if lmf_gating_length <= 0:
                raise ValueError("lmf_gating_length must be positive")
            if lmf_gating_offset < 0 or lmf_gating_offset + lmf_gating_length > user_dense_dim:
                raise ValueError(
                    "lmf_gating slice must fit inside user_dense_feats: "
                    f"user_dense_dim={user_dense_dim}, "
                    f"offset={lmf_gating_offset}, length={lmf_gating_length}")
            self.lmf_gating_norm = nn.LayerNorm(lmf_gating_length)
            self.lmf_gating_dropout = nn.Dropout(lmf_gating_dropout_rate)
            self.lmf_gating_gate = nn.Sequential(
                nn.Linear(lmf_gating_length, d_model),
                nn.Sigmoid(),
            )
            self.lmf_gating_proj = nn.Sequential(
                nn.Linear(self.num_ns * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
        if num_gap_buckets > 0:
            self.gap_embedding = nn.Embedding(num_gap_buckets, d_model, padding_idx=0)

        # ================== Historical Sequence Time-Perception ==================
        # Optional residual that injects an MLP-projected (Hour, Day, Week)
        # signal onto each history token inside ``_embed_seq_domain``. Only the
        # first variant (historical sequence) of the Time-Aware Gated Attention
        # paper is wired up here; the anchor + gated-attention variants are
        # intentionally out of scope.
        self.enable_history_time_bias = enable_history_time_bias
        if self.enable_history_time_bias:
            self.history_time_bias_injector = HistoricalTemporalBiasInjector(
                d_model=d_model,
                residual_scale=history_time_bias_residual_scale,
            )

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                rank_moe_enable=rank_moe_enable,
                rank_moe_num_experts=rank_moe_num_experts,
                rank_moe_top_k=rank_moe_top_k,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== Target-aware Cross-Attention Bias ==================
        # One independent bias head per (HyFormer block, sequence domain)
        # so shallow and deep blocks can learn distinct target priors.
        # The per-(block, domain) scalar alpha starts at 0, so when
        # ``bias_attention_up=True`` the network is initially
        # equivalent to the unbiased cross-attention path and only
        # amplifies the prior on layers/domains where it helps.
        if self.bias_attention_up:
            self.target_attn_bias_modules = nn.ModuleList([
                nn.ModuleList([
                    TargetAwareCrossAttentionBias(
                        d_model=d_model,
                        dropout=self.target_attn_bias_dropout,
                    )
                    for _ in range(self.num_sequences)
                ])
                for _ in range(num_hyformer_blocks)
            ])
            self.target_attn_bias_alpha = nn.Parameter(
                torch.zeros(num_hyformer_blocks, self.num_sequences)
            )
        else:
            self.target_attn_bias_modules = None
            self.target_attn_bias_alpha = None

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

        self.calendar_time_feature_encoder = CalendarTimeFeatureEncoder(
            d_model=d_model,
            enable_intraday_calendar_features=enable_intraday_calendar_features,
            enable_weekly_calendar_features=enable_weekly_calendar_features,
            enable_annual_calendar_features=enable_annual_calendar_features,
            defer_output_projection=True,
        )
        # target_anchor_projector is shared by the additive DIN branch
        # (seqContextwakeup) and the per-token reweighter (enable_seq_reweight).
        if self.seqContextwakeup or self.enable_seq_reweight:
            self.target_anchor_projector = TargetAnchorProjector(
                d_model=d_model,
                hidden_mult=target_activation_hidden_mult,
            )
        if self.seqContextwakeup:
            self.target_sequence_activation = TargetAwareSequenceActivation(
                d_model=d_model,
                num_sequences=self.num_sequences,
                hidden_mult=target_activation_hidden_mult,
                dropout=dropout_rate,
                top_k=target_activation_top_k,
            )
        if self.enable_seq_reweight:
            self.target_sequence_reweighter = TargetAwareSequenceReweighter(
                d_model=d_model,
                num_sequences=self.num_sequences,
                hidden_mult=seq_reweight_hidden_mult,
                dropout=dropout_rate,
            )
        self.calendar_time_feature_encoder.build_output_projection()


    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0
        if self.num_gap_buckets > 0:
            nn.init.xavier_normal_(self.gap_embedding.weight.data)
            self.gap_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.num_gap_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        gap_bucket_ids: Optional[torch.Tensor] = None,
        history_time_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        if self.num_gap_buckets > 0:
            if gap_bucket_ids is None:
                gap_bucket_ids = seq.new_zeros(B, L, dtype=torch.long)
            token_emb = token_emb + self.gap_embedding(gap_bucket_ids)

        # Inject the (Hour, Day, Week) temporal-bias residual described in the
        # Time-Aware Gated Attention scheme; this is a no-op when the feature
        # is disabled or the caller didn't supply temporal ids.
        if self.enable_history_time_bias:
            token_emb = self.history_time_bias_injector(token_emb, history_time_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        target_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection.

        Args:
            target_anchor: Optional (B, D) candidate-aware target. When
                ``bias_attention_up=True`` this anchor is routed to
                every block to drive the per-block, per-domain
                ``TargetAwareCrossAttentionBias`` head.
        """
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        use_target_attn_bias = (
            self.bias_attention_up
            and self.target_attn_bias_modules is not None
            and self.target_attn_bias_alpha is not None
            and target_anchor is not None
        )

        for block_idx, block in enumerate(self.blocks):
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            block_target_bias_modules = (
                self.target_attn_bias_modules[block_idx]
                if use_target_attn_bias else None
            )
            block_target_bias_alpha = (
                self.target_attn_bias_alpha[block_idx]
                if use_target_attn_bias else None
            )

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                target_anchor=target_anchor if use_target_attn_bias else None,
                target_attn_bias_modules=block_target_bias_modules,
                target_attn_bias_alpha=block_target_bias_alpha,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def get_moe_aux_loss(self) -> torch.Tensor:
        aux_losses = []
        for block in self.blocks:
            aux_loss = block.get_moe_aux_loss()
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        if not aux_losses:
            return next(self.parameters()).new_tensor(0.0)
        return torch.stack(aux_losses).sum()

    @staticmethod
    def _merge_optional_context_token(
        base_token: torch.Tensor,
        optional_token: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        return [base_token] if optional_token is None else [base_token, optional_token]

    def _build_target_anchor(
        self,
        user_ns_tokens: torch.Tensor,
        item_ns_tokens: torch.Tensor,
        user_dense_context_token: Optional[torch.Tensor] = None,
        item_dense_context_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Builds the candidate target anchor for sequence activation."""
        user_tokens = self._merge_optional_context_token(user_ns_tokens, user_dense_context_token)
        item_tokens = self._merge_optional_context_token(item_ns_tokens, item_dense_context_token)
        return self.target_anchor_projector(user_tokens, item_tokens)

    def _build_non_sequence_tokens(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Builds static tokens used by the query generator and target activation."""
        calendar_context_token = self.calendar_time_feature_encoder(inputs.calendar_time_features)
        user_ns_tokens = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns_tokens = self.item_ns_tokenizer(inputs.item_int_feats)

        user_ns_tokens = user_ns_tokens + calendar_context_token

        ns_parts = [user_ns_tokens]
        user_dense_context_token = None
        if self.has_user_dense:
            user_dense_context_token = self.user_dense_feature_encoder(inputs.user_dense_feats)
            ns_parts.append(user_dense_context_token)
        ns_parts.append(item_ns_tokens)
        if self.has_item_dense:
            item_dense_context_token = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_context_token)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)
        return ns_tokens, user_ns_tokens, item_ns_tokens, user_dense_context_token

    def _embed_sequence_inputs(
        self,
        inputs: ModelInput,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Embeds every sequence domain and builds its padding mask."""
        sequence_token_list = []
        sequence_padding_masks = []
        seq_gap_buckets = inputs.seq_gap_buckets or {}
        seq_history_time_ids = inputs.seq_history_time_ids or {}
        for domain in self.seq_domains:
            domain_tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                seq_gap_buckets.get(domain),
                history_time_ids=seq_history_time_ids.get(domain),
            )
            sequence_token_list.append(domain_tokens)
            padding_mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            sequence_padding_masks.append(padding_mask)
        return sequence_token_list, sequence_padding_masks

    def _build_lmf_gating_vector(
        self,
        inputs: ModelInput,
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Builds a residual vector by gating ``ns_tokens`` with the LMF slice.

        Pipeline:
            1. Take ``user_dense_feats[:, offset:offset+length]`` (the LMF
               user-CVR embedding) and sanitise it with ``nan_to_num``.
            2. Drop entire rows whose post-sanitisation L2 norm exceeds
               ``lmf_gating_max_norm`` (cheap outlier guard).
            3. LayerNorm + Linear + Sigmoid → ``[B, 1, d_model]`` gate.
            4. ``interacted = ns_tokens * gate`` (broadcast over num_ns).
            5. Flatten and project back to ``d_model`` via a SiLU MLP.
            6. Multiply by the validity mask so invalid rows contribute 0.
        """
        lmf_values = inputs.user_dense_feats[
            :, self.lmf_gating_offset:self.lmf_gating_offset + self.lmf_gating_length]
        lmf_values = lmf_values.to(dtype=ns_tokens.dtype)
        lmf_values = torch.nan_to_num(lmf_values, nan=0.0, posinf=0.0, neginf=0.0)

        norm_valid = (
            lmf_values.norm(p=2, dim=1, keepdim=True) <= self.lmf_gating_max_norm
        ).to(dtype=ns_tokens.dtype)

        if self.training and self.lmf_gating_dropout_rate > 0:
            lmf_values = self.lmf_gating_dropout(lmf_values)

        gate = self.lmf_gating_gate(self.lmf_gating_norm(lmf_values)).unsqueeze(1)
        interacted = ns_tokens * gate
        lmf_vector = self.lmf_gating_proj(interacted.reshape(interacted.shape[0], -1))
        return lmf_vector * norm_valid

    def _compute_feature_output(
        self,
        inputs: ModelInput,
        apply_dropout: bool,
    ) -> torch.Tensor:
        """Computes the final pre-classifier representation."""
        # 1. NS tokens: grouped projection
        ns_tokens, user_ns_tokens, item_ns_tokens, user_dense_context_token = (
            self._build_non_sequence_tokens(inputs)
        )

        # 2. Embed each sequence domain (dynamic)
        sequence_token_list, sequence_padding_masks = self._embed_sequence_inputs(inputs)

        # 3. (Optional) Build target_anchor once, shared by:
        #    - pre-query-generator reweighter (enable_seq_reweight)
        #    - additive DIN branch after the HyFormer stack (seqContextwakeup)
        target_anchor = None
        if self.seqContextwakeup or self.enable_seq_reweight:
            target_anchor = self._build_target_anchor(
                user_ns_tokens, item_ns_tokens, user_dense_context_token, None)

        # 3b. (Optional) Build the pure item-side target for the
        #     target-aware cross-attention bias. Mean-pool item NS tokens
        #     to a single (B, D) candidate representation. Kept separate
        #     from ``target_anchor`` so the bias path stays a clean item
        #     prior over key positions, while ``target_anchor`` keeps
        #     mixing user×item interaction for the other branches.
        bias_target = None
        if self.bias_attention_up:
            bias_target = item_ns_tokens.mean(dim=1)  # (B, D)

        # 4. (Optional) DIN-style per-token reweighting of sequence tokens
        #    BEFORE query generation. The re-weighted sequences also flow into
        #    the HyFormer blocks and (if enabled) the additive DIN branch.
        if self.enable_seq_reweight:
            sequence_token_list = self.target_sequence_reweighter(
                target_anchor, sequence_token_list, sequence_padding_masks)

        # 5. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        query_token_list = self.query_generator(ns_tokens, sequence_token_list, sequence_padding_masks)

        # 6. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            query_token_list, ns_tokens, sequence_token_list, sequence_padding_masks,
            apply_dropout=apply_dropout,
            target_anchor=bias_target,
        )
        if self.seqContextwakeup:
            output = output + self.target_sequence_activation(
                target_anchor, sequence_token_list, sequence_padding_masks)

        # 7. (Optional) LMF gating residual: a small additive correction driven
        #    by the LMF user-CVR embedding slice. Does not alter ns_tokens or
        #    the existing user_dense_token forward path.
        if self.use_lmf_gating_branch:
            output = output + self.lmf_gating_residual_scale * self._build_lmf_gating_vector(
                inputs, ns_tokens)
        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        output = self._compute_feature_output(inputs, apply_dropout=self.training)
        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        output = self._compute_feature_output(inputs, apply_dropout=False)
        logits = self.clsfier(output)
        return logits, output
