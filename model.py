"""Hybrid Mamba-Transformer architecture for mechanistic interpretability.

Layer layout is set by HybridConfig.layer_pattern, a string of 'M'
(SimpleMambaBlock, SSM with explicit sequential scan) and 'A'
(StandardAttentionBlock), e.g. "MAMA", "AMAM", "MMMM", "AAAA". The two block
types are identical in every architecture, so differences between
architectures come only from which block sits where.

Design constraints (from project spec):
  - SimpleMambaBlock uses an explicit Python for-loop over time steps so that
    h_t is materialised at every position and can be patched or probed.
  - All critical tensor shapes are annotated inline.
  - No custom CUDA kernels.

Cache keys returned by run_with_cache():
  Mamba layers:
    'h'        : (B, L, D_inner, N)  — SSM hidden state after each step
    'delta'    : (B, L, D_inner)     — discretisation step sizes Δ_t
    'B_t'      : (B, L, N)           — input projection (write gate)
    'C_t'      : (B, L, N)           — output projection (read gate)
    'x_act'    : (B, L, D_inner)     — post-activation SSM input
    'z'        : (B, L, D_inner)     — multiplicative gate
    'block_out': (B, L, D_inner)     — post-gate pre-out_proj
    'resid_out': (B, L, D)           — full block output (post-residual)

  Attention layers:
    'Q'        : (B, L, H, d_h)     — query heads
    'K'        : (B, L, H, d_h)     — key heads
    'V'        : (B, L, H, d_h)     — value heads
    'attn_w'   : (B, H, L, L)       — softmax attention weights
    'attn_out' : (B, L, D)          — attention output (pre-residual-add)
    'mlp_out'  : (B, L, D)          — MLP output (pre-residual-add)
    'resid_out': (B, L, D)          — full block output (post-residual)
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    vocab_size: int = 64
    d_model: int = 128
    n_layers: int = 4       # ignored if layer_pattern is given
    d_state: int = 16       # SSM state dimension N
    d_conv: int = 4         # depthwise conv kernel width
    expand: int = 2         # Mamba inner-dim multiplier
    dt_rank: int = 0        # 0 → auto: max(1, d_model // 16)
    n_heads: int = 4        # attention heads
    mlp_ratio: int = 2      # attention MLP width = mlp_ratio * d_model. 2 keeps
                            # an attention block (~132k params at d_model=128)
                            # close to a Mamba block (~117k).

    # Architecture pattern: string of 'M' (Mamba) / 'A' (Attention).
    # None → default alternating "MA" repeated (backwards compatible).
    # Examples: "MAMA", "AMAM", "MMMM", "AAAA".
    layer_pattern: Optional[str] = None

    # Position information for the attention layers:
    #   'rope'    — rotary embeddings on Q/K inside every attention layer.
    #               Default: every architecture then uses the *same* attention
    #               block, and relative positions carry over to longer
    #               sequences.
    #   'learned' — learned absolute embeddings added to the token embedding.
    #               CAVEAT: cannot be evaluated beyond max_seq_len, and
    #               positions never seen in training are untrained.
    #   'none'    — no explicit position signal (attention only gets what the
    #               causal mask and any earlier Mamba layer leak).
    pos_encoding: str = "rope"
    # Bias on the Q/K projections. With RoPE it gives every query and key a
    # token-independent component, which is what a purely position-based
    # attention pattern (e.g. 'attend to the previous token') is built from.
    qk_bias: bool = True
    max_seq_len: int = 512  # only used by pos_encoding='learned'

    def __post_init__(self):
        if self.layer_pattern is None:
            assert self.n_layers % 2 == 0, "n_layers must be even for default MA pattern"
            self.layer_pattern = "MA" * (self.n_layers // 2)
        self.layer_pattern = self.layer_pattern.upper()
        assert set(self.layer_pattern) <= {"M", "A"}, "layer_pattern must contain only 'M'/'A'"
        self.n_layers = len(self.layer_pattern)
        assert self.d_model % self.n_heads == 0
        assert self.pos_encoding in ("rope", "learned", "none")
        if self.pos_encoding == "rope":
            assert (self.d_model // self.n_heads) % 2 == 0, "RoPE needs an even d_head"
        if self.dt_rank == 0:
            self.dt_rank = max(1, self.d_model // 16)

    @property
    def d_inner(self) -> int:                  # Mamba inner dimension D_inner
        return int(self.expand * self.d_model)

    @property
    def d_head(self) -> int:                   # per-head dimension d_h
        return self.d_model // self.n_heads

    def layer_type(self, layer_idx: int) -> str:
        return "mamba" if self.layer_pattern[layer_idx] == "M" else "attn"


# ---------------------------------------------------------------------------
# SimpleMambaBlock
# ---------------------------------------------------------------------------

class SimpleMambaBlock(nn.Module):
    """
    One Mamba layer with pre-norm, depthwise conv, explicit SSM scan, and
    multiplicative gate.

    The SSM scan is an explicit Python loop (`for t in range(L)`) so that
    h_t is materialised and patchable at every position.

    Args (forward):
        x              : (B, L, D)
        h_override     : (B, L, D_inner, N)  — clean h to splice post-step
        h_override_mask: (L,) bool            — which positions to override
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        D = config.d_model
        Di = config.d_inner          # D_inner = expand * D
        N = config.d_state           # SSM state dim
        R = config.dt_rank

        self.d_inner = Di
        self.d_state = N
        self.dt_rank = R

        # Pathway ablations (see mamba_ablation). A Mamba block can move
        # information across positions in exactly two ways:
        #   memory — the recurrent state h_{t-1} → h_t
        #   conv   — the depthwise conv, which sees the previous d_conv-1 tokens
        self.ablate_memory = False   # True → h_t forgets h_{t-1} (Ā = 0)
        self.ablate_conv = False     # True → conv keeps only the current token

        self.norm = nn.LayerNorm(D)

        # Input projection: D → 2 * D_inner (SSM branch + gate branch)
        self.in_proj = nn.Linear(D, 2 * Di, bias=False)   # (D, 2·Di)

        # Depthwise causal conv over the SSM branch
        self.conv1d = nn.Conv1d(
            Di, Di, config.d_conv,
            bias=True, padding=config.d_conv - 1, groups=Di,
        )

        # SSM input projections: D_inner → [Δ_raw | B_t | C_t]
        self.x_proj = nn.Linear(Di, R + 2 * N, bias=False)   # (Di, R+2N)

        # Δ projection: R → D_inner  (with bias for stable init)
        self.dt_proj = nn.Linear(R, Di, bias=True)

        # A: log-parameterised (D_inner, N), initialised 1..N per row
        A_init = torch.arange(1, N + 1, dtype=torch.float32).repeat(Di, 1)  # (Di, N)
        self.A_log = nn.Parameter(torch.log(A_init))

        # D: skip-connection scalar per channel
        self.D = nn.Parameter(torch.ones(Di))

        # Output projection: D_inner → D
        self.out_proj = nn.Linear(Di, D, bias=False)

        self._init_dt()

    def _init_dt(self):
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt0 = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))

    # ------------------------------------------------------------------
    # Selective scan (explicit loop — interpretability-first)
    # ------------------------------------------------------------------

    def _scan(
        self,
        x_act: torch.Tensor,             # (B, L, D_inner)
        delta: torch.Tensor,             # (B, L, D_inner)
        B_t: torch.Tensor,               # (B, L, N)
        C_t: torch.Tensor,               # (B, L, N)
        h_override: Optional[torch.Tensor],       # (B, L, D_inner, N) or None
        h_override_mask: Optional[torch.Tensor],  # (L,) bool or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            y       : (B, L, D_inner)
            h_cache : (B, L, D_inner, N)
        """
        B, L, Di = x_act.shape
        N = self.d_state

        # A: negative definite (Di, N)
        A = -torch.exp(self.A_log.float())

        # Discretise via zero-order hold
        # dA[b,t,d,n] = exp( delta[b,t,d] * A[d,n] )
        dA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))   # (B, L, Di, N)
        # Write term dB_t ⊙ x_t, with dB[b,t,d,n] = delta[b,t,d] * B_t[b,t,n]
        # (Euler approx)
        dBx = torch.einsum("bld,bln->bldn", delta * x_act, B_t)   # (B, L, Di, N)
        if self.ablate_memory:
            dA = torch.zeros_like(dA)     # h_t = dB_t ⊙ x_t: nothing carried over

        # Split into per-step views once. Indexing dA[:, t] inside the loop
        # would make the backward pass allocate a full (B, L, Di, N) tensor at
        # every step, which dominates training time.
        dA_steps = dA.unbind(dim=1)                               # L × (B, Di, N)
        dBx_steps = dBx.unbind(dim=1)                             # L × (B, Di, N)
        patch_steps: set[int] = set()
        if h_override is not None and h_override_mask is not None:
            patch_steps = set(h_override_mask.nonzero().flatten().tolist())

        h = x_act.new_zeros(B, Di, N)                             # (B, Di, N)
        h_list: list[torch.Tensor] = []

        for t in range(L):
            # Recurrence: h_t = dA_t ⊙ h_{t-1} + dB_t ⊙ x_t
            h = torch.addcmul(dBx_steps[t], dA_steps[t], h)       # (B, Di, N)

            # Patch h_t with clean activation if requested
            if t in patch_steps:
                h = h_override[:, t]                               # (B, Di, N)

            h_list.append(h)                                       # store (B, Di, N)

        h_cache = torch.stack(h_list, dim=1)                       # (B, L, Di, N)
        # Output: y_t = C_t · h_t  (the D · x_t skip is added by the caller)
        y = torch.einsum("bldn,bln->bld", h_cache, C_t)            # (B, L, Di)
        return y, h_cache

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                          # (B, L, D)
        h_override: Optional[torch.Tensor] = None,          # (B, L, Di, N)
        h_override_mask: Optional[torch.Tensor] = None,     # (L,) bool
    ) -> tuple[torch.Tensor, dict]:

        B, L, D = x.shape
        residual = x                               # (B, L, D)  — residual branch

        x_n = self.norm(x)                        # (B, L, D)

        # Expand and split
        xz = self.in_proj(x_n)                    # (B, L, 2·Di)
        x_ssm, z = xz.chunk(2, dim=-1)            # each (B, L, Di)

        # Causal depthwise conv (trim right-padding). With padding d_conv-1,
        # the LAST kernel tap multiplies the current token and the earlier
        # taps multiply the previous d_conv-1 tokens.
        conv_w = self.conv1d.weight                # (Di, 1, d_conv)
        if self.ablate_conv:
            conv_w = conv_w.clone()
            conv_w[..., :-1] = 0.0                 # drop the previous-token taps
        x_conv = F.conv1d(
            x_ssm.transpose(1, 2),                 # (B, Di, L)
            conv_w, self.conv1d.bias,
            padding=self.conv1d.padding[0], groups=self.d_inner,
        )[:, :, :L].transpose(1, 2)               # (B, L, Di)
        x_act = F.silu(x_conv)                    # (B, L, Di)

        # Project to [Δ_raw | B_t | C_t]
        x_dbl = self.x_proj(x_act)                # (B, L, R+2N)
        delta_raw = x_dbl[..., :self.dt_rank]                        # (B, L, R)
        B_t = x_dbl[..., self.dt_rank: self.dt_rank + self.d_state]  # (B, L, N)
        C_t = x_dbl[..., self.dt_rank + self.d_state:]               # (B, L, N)

        delta = F.softplus(self.dt_proj(delta_raw))   # (B, L, Di)

        # Selective scan
        y, h_cache = self._scan(x_act, delta, B_t, C_t, h_override, h_override_mask)
        # y: (B, L, Di),  h_cache: (B, L, Di, N)

        y = y + self.D * x_act                    # (B, L, Di)  — skip connection
        block_out = y * F.silu(z)                 # (B, L, Di)  — multiplicative gate

        out = self.out_proj(block_out) + residual  # (B, L, D)

        cache = {
            "h":         h_cache,    # (B, L, Di, N)
            "delta":     delta,      # (B, L, Di)
            "B_t":       B_t,        # (B, L, N)
            "C_t":       C_t,        # (B, L, N)
            "x_act":     x_act,      # (B, L, Di)
            "z":         z,          # (B, L, Di)
            "block_out": block_out,  # (B, L, Di)
            "resid_out": out,        # (B, L, D)
        }
        return out, cache


# ---------------------------------------------------------------------------
# StandardAttentionBlock
# ---------------------------------------------------------------------------

class StandardAttentionBlock(nn.Module):
    """
    Standard pre-norm causal multi-head attention + MLP block.

    K and V tensors are explicitly separated and can be overridden per-position
    to support cross-architecture activation patching. K is cached and
    overridden *before* the rotary rotation, which is a fixed function of
    position, so splicing K at a position keeps that position's rotation.

    Args (forward):
        x                 : (B, L, D)
        K_override        : (B, L, H, d_h)  — clean K to splice in
        V_override        : (B, L, H, d_h)  — clean V to splice in
        KV_mask           : (L,) bool        — which positions to override K/V
        attn_out_override : (B, L, D)        — clean attention output
        attn_out_mask     : (L,) bool        — which positions to override it
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        D = config.d_model
        H = config.n_heads
        d_h = config.d_head

        self.n_heads = H
        self.d_head = d_h
        self.scale = d_h ** -0.5
        self.use_rope = config.pos_encoding == "rope"

        # Attention sublayer
        self.norm1 = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D, bias=config.qk_bias)    # (D, D)
        self.k_proj = nn.Linear(D, D, bias=config.qk_bias)    # (D, D)
        self.v_proj = nn.Linear(D, D, bias=False)    # (D, D)
        self.out_proj = nn.Linear(D, D, bias=False)  # (D, D)

        # MLP sublayer
        self.norm2 = nn.LayerNorm(D)
        d_ff = config.mlp_ratio * D
        self.mlp = nn.Sequential(
            nn.Linear(D, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, D, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,                          # (B, L, D)
        K_override: Optional[torch.Tensor] = None,   # (B, L, H, d_h)
        V_override: Optional[torch.Tensor] = None,   # (B, L, H, d_h)
        KV_mask: Optional[torch.Tensor] = None,      # (L,) bool
        attn_out_override: Optional[torch.Tensor] = None,   # (B, L, D)
        attn_out_mask: Optional[torch.Tensor] = None,       # (L,) bool
    ) -> tuple[torch.Tensor, dict]:

        B, L, D = x.shape
        H, d_h = self.n_heads, self.d_head
        residual_attn = x                             # (B, L, D)

        # --- Attention sublayer ---
        x_n1 = self.norm1(x)                         # (B, L, D)

        # Project and reshape to heads
        Q = self.q_proj(x_n1).view(B, L, H, d_h)    # (B, L, H, d_h)
        K = self.k_proj(x_n1).view(B, L, H, d_h)    # (B, L, H, d_h)
        V = self.v_proj(x_n1).view(B, L, H, d_h)    # (B, L, H, d_h)

        # Splice in clean K / V at masked positions (activation patching)
        if KV_mask is not None:
            if K_override is not None:
                K = K.clone()
                K[:, KV_mask] = K_override[:, KV_mask]  # (B, L, H, d_h)
            if V_override is not None:
                V = V.clone()
                V[:, KV_mask] = V_override[:, KV_mask]  # (B, L, H, d_h)

        # Attention: (B, H, L, L)
        Q_t = Q.permute(0, 2, 1, 3)                  # (B, H, L, d_h)
        K_t = K.permute(0, 2, 1, 3)                  # (B, H, L, d_h)
        V_t = V.permute(0, 2, 1, 3)                  # (B, H, L, d_h)
        if self.use_rope:
            Q_t, K_t = self._rope(Q_t), self._rope(K_t)

        scores = (Q_t @ K_t.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask[None, None], float("-inf"))
        attn_w = F.softmax(scores, dim=-1)            # (B, H, L, L)

        # Weighted sum over values
        attn_vals = attn_w @ V_t                      # (B, H, L, d_h)
        attn_concat = attn_vals.permute(0, 2, 1, 3).reshape(B, L, D)  # (B, L, D)
        attn_out = self.out_proj(attn_concat)         # (B, L, D)

        # Splice in the clean attention output *before* the residual add, so
        # this block's MLP also sees the patched value.
        if attn_out_override is not None and attn_out_mask is not None:
            attn_out = attn_out.clone()
            attn_out[:, attn_out_mask] = attn_out_override[:, attn_out_mask]

        x = residual_attn + attn_out                  # (B, L, D)

        # --- MLP sublayer ---
        x_n2 = self.norm2(x)                          # (B, L, D)
        mlp_out = self.mlp(x_n2)                      # (B, L, D)
        out = x + mlp_out                             # (B, L, D)

        cache = {
            "Q":        Q,          # (B, L, H, d_h)
            "K":        K,          # (B, L, H, d_h)  — post-override
            "V":        V,          # (B, L, H, d_h)  — post-override
            "attn_w":   attn_w,     # (B, H, L, L)
            "attn_out": attn_out,   # (B, L, D)   — post-override
            "mlp_out":  mlp_out,    # (B, L, D)
            "resid_out": out,       # (B, L, D)
        }
        return out, cache

    @staticmethod
    def _rope(x: torch.Tensor) -> torch.Tensor:
        """Rotary position embedding: rotate each (even, odd) pair of the head
        dimension by an angle proportional to the position. x: (B, H, L, d_h)."""
        L, d_h = x.shape[-2], x.shape[-1]
        inv_freq = 1.0 / (10000 ** (
            torch.arange(0, d_h, 2, device=x.device, dtype=x.dtype) / d_h
        ))                                             # (d_h/2,)
        ang = torch.arange(L, device=x.device, dtype=x.dtype)[:, None] * inv_freq
        cos, sin = ang.cos(), ang.sin()                # (L, d_h/2)
        x1, x2 = x[..., 0::2], x[..., 1::2]            # (B, H, L, d_h/2)
        return torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        ).flatten(-2)                                  # (B, H, L, d_h)


# ---------------------------------------------------------------------------
# HybridLM
# ---------------------------------------------------------------------------

class HybridLM(nn.Module):
    """
    Mamba / attention LM with the layer order given by config.layer_pattern.

    Central hypothesis under test (for the "MAMA" hybrid): Mamba (L0) compresses the [A, B] bigram
    into h_t and writes it to the residual stream; the subsequent Attention
    (L1) reads this out via its K/V projections, implementing a
    cross-architecture induction circuit.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.max_seq_len, config.d_model)
            if config.pos_encoding == "learned" else None
        )

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if config.layer_type(i) == "mamba":
                self.layers.append(SimpleMambaBlock(config))
            else:
                self.layers.append(StandardAttentionBlock(config))

        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        std = self.config.d_model ** -0.5
        nn.init.normal_(self.embedding.weight, std=std)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=std)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SimpleMambaBlock):
                nn.init.normal_(layer.in_proj.weight, std=std)
                nn.init.normal_(layer.out_proj.weight,
                                 std=std / math.sqrt(2 * self.config.n_layers))
            else:
                for proj in [layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj]:
                    nn.init.normal_(proj.weight, std=std)

    # ------------------------------------------------------------------

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)          # (B, L, D)
        if self.pos_emb is not None:
            L = input_ids.shape[1]
            assert L <= self.config.max_seq_len, (
                f"seq_len {L} exceeds max_seq_len {self.config.max_seq_len} "
                "(pos_encoding='learned' model cannot extrapolate)"
            )
            pos = torch.arange(L, device=input_ids.device)
            x = x + self.pos_emb(pos)[None]    # (B, L, D)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard forward. Returns logits (B, L, V)."""
        x = self._embed(input_ids)             # (B, L, D)
        for layer in self.layers:
            x, _ = layer(x)
        return self.lm_head(self.norm_f(x))    # (B, L, V)

    @torch.no_grad()
    def run_with_cache(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Full forward pass storing all intermediate activations.

        Returns:
            logits : (B, L, V)
            caches : list[dict] of length n_layers
                     Mamba layers → keys: h, delta, B_t, C_t, x_act, z, block_out, resid_out
                     Attn layers  → keys: Q, K, V, attn_w, attn_out, mlp_out, resid_out
        """
        x = self._embed(input_ids)             # (B, L, D)
        caches: list[dict] = []
        for layer in self.layers:
            x, cache = layer(x)
            caches.append(cache)
        logits = self.lm_head(self.norm_f(x))  # (B, L, V)
        return logits, caches

    @property
    def A_matrices(self) -> list[tuple[int, torch.Tensor]]:
        """Return [(layer_idx, A_matrix)] for all Mamba layers."""
        result = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SimpleMambaBlock):
                A = -torch.exp(layer.A_log.detach())   # (Di, N)
                result.append((i, A))
        return result


# ---------------------------------------------------------------------------
# Pathway ablation
# ---------------------------------------------------------------------------

@contextmanager
def mamba_ablation(
    model: HybridLM,
    layers: Optional[list[int]] = None,   # Mamba layer indices; None → all
    memory: bool = False,
    conv: bool = False,
):
    """
    Temporarily switch off cross-position pathways in the given Mamba layers.

      memory=True → the SSM state is not carried between positions
      conv=True   → the depthwise conv sees only the current token

    With both on, the block is a purely per-token function (like an MLP), so
    it cannot move any information between positions.
    """
    blocks = [
        layer for i, layer in enumerate(model.layers)
        if isinstance(layer, SimpleMambaBlock) and (layers is None or i in layers)
    ]
    for b in blocks:
        b.ablate_memory, b.ablate_conv = memory, conv
    try:
        yield model
    finally:
        for b in blocks:
            b.ablate_memory, b.ablate_conv = False, False
