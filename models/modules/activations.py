from __future__ import annotations

from typing import Callable

import torch.nn.functional as F
from torch import Tensor, nn


# --- Feed-Forward Network Variants for Transformer Layers ----------------------------------

class FFN(nn.Module):
    """Vanilla feed-forward network used in Transformer blocks.

    Structure:
        x -> Linear(d_in -> d_ffn) -> GELU -> Dropout -> Linear(d_ffn -> d_out)

    Args:
        d_in: Input dimension.
        d_ffn: Hidden dimension (expansion).
        d_out: Output dimension.
        dropout: Dropout probability applied after the activation.
    """

    def __init__(self, d_in: int, d_ffn: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        # NOTE: kept d_ffn as an attribute because some external code references it.
        self.d_ffn = d_ffn

        self.w1 = nn.Linear(d_in, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation: Callable[[Tensor], Tensor] = F.gelu

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class GEGLUFFN(nn.Module):
    """GEGLU feed-forward network (Gated GELU).

    Structure:
        gate = GELU(W_g x)
        value = W_v x
        y = (value * gate) -> Dropout -> W_o -> Dropout

    Args:
        d_in: Input dimension.
        d_ffn: Hidden (pre-output) dimension.
        d_out: Output dimension.
        dropout: Dropout probability applied before and after the output projection.
    """

    def __init__(self, d_in: int, d_ffn: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.d_ffn = d_ffn

        self.w_value = nn.Linear(d_in, d_ffn)   # value projection
        self.w_gate = nn.Linear(d_in, d_ffn)    # gate projection
        self.w_out = nn.Linear(d_ffn, d_out)    # output projection (from d_ffn!)
        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value = self.w_value(x)
        gate = F.gelu(self.w_gate(x))
        x = value * gate
        x = self.dropout_in(x)
        x = self.w_out(x)
        return self.dropout_out(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020).

    Structure:
        gate = SiLU(W_g x * beta)
        value = W_v x
        y = (value * gate) -> Dropout -> W_o -> Dropout

    Args:
        d_in: Input dimension.
        d_ffn: Hidden (pre-output) dimension.
        d_out: Output dimension.
        beta: Scaling factor applied to the gate pre-activation.
        dropout: Dropout probability applied before and after the output projection.

    Notes:
        - Includes a `gated_tap` nn.Identity hook point to make it easy to register
          forward hooks and inspect the gated product without changing code.
    """

    def __init__(
        self,
        d_in: int,
        d_ffn: int,
        d_out: int,
        beta: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_ffn = d_ffn
        self.beta = beta

        self.w_value = nn.Linear(d_in, d_ffn)
        self.w_gate = nn.Linear(d_in, d_ffn)
        self.w_out = nn.Linear(d_ffn, d_out)
        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

        # Tap for hooking/inspection
        self.gated_tap = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        value = self.w_value(x)
        gate = F.silu(self.w_gate(x) * self.beta)
        x = value * gate
        self.gated_tap(x)  # hook point
        x = self.dropout_in(x)
        x = self.w_out(x)
        return self.dropout_out(x)