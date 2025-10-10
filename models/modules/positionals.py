from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# --- Postional Encoding Methods -------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al.).

    Adds a fixed, precomputed sinusoidal encoding to the input sequence and applies dropout.

    Args:
        d_model: Embedding/model dimension.
        dropout: Dropout probability applied after adding the encoding.
        max_len: Maximum supported sequence length.

    Shapes:
        x: (B, S, d_model)

    Returns:
        (B, S, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)  # (1, S, D)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # even dims
        pe[0, :, 1::2] = torch.cos(position * div_term)  # odd dims

        # Kept as a buffer so it moves with the module; not a parameter.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to `x` and apply dropout."""
        # Slice to current sequence length and cast to x's dtype/device for safety
        pe = self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)
        x = x + pe
        return self.dropout(x)

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Implements RoPE as in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864)

    Args:
        dim: Feature dimension on which to apply rotation; must be even.
        max_seq_len: Maximum supported sequence length.
        base: Base for frequency calculation.

    Shapes:
        x: (B, S, ..., dim)  — rotation is applied to the last dimension.

    Returns:
        Tensor with RoPE applied, same shape as input.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        half_dim = dim // 2

        # Frequencies: base^{-i/half_dim}, i = 0..half_dim-1
        freq = torch.pow(base, -torch.arange(half_dim, dtype=torch.float32) / half_dim)  # (Dh,)

        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (S,)
        angles = torch.outer(positions, 1.0 / freq)  # (S, Dh)

        cos = torch.cos(angles)  # (S, Dh)
        sin = torch.sin(angles)  # (S, Dh)

        # Register as buffers so they move with the module
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply rotary embeddings to the last dimension of `x`."""
        if x.size(-1) % 2 != 0:
            raise ValueError(f"Last dimension must be even for RoPE, got {x.size(-1)}")

        seq_len = x.size(1)
        Dh = x.size(-1) // 2

        # Select needed slice and cast/broadcast to input dtype/device
        cos = self.cos[:seq_len, :Dh].to(dtype=x.dtype, device=x.device)  # (S, Dh)
        sin = self.sin[:seq_len, :Dh].to(dtype=x.dtype, device=x.device)  # (S, Dh)

        # Build broadcastable shapes: (1, S, 1, ..., 1, Dh)
        # Number of extra dims between S and last dim:
        extra = x.dim() - 3  # e.g., B,S,D -> 0; B,S,H,D -> 1
        view_shape = [1, seq_len] + [1] * extra + [Dh]
        cos = cos.view(*view_shape)
        sin = sin.view(*view_shape)

        # Split last dim into pairs (..., Dh, 2)
        x_shape = x.shape
        x = x.view(*x_shape[:-1], Dh, 2)
        x1, x2 = x[..., 0], x[..., 1]  # even/odd parts

        # Apply rotation:
        # [x1'; x2'] = [ x1 * cos - x2 * sin ; x1 * sin + x2 * cos ]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

        # Restore original shape
        return rotated.view(*x_shape)