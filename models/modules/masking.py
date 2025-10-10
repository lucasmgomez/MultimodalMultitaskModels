from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


# --- General masking ---------------------------------------------------------

def get_padding_mask(
    sequences: Tensor, padding_token: Tensor, device: str | torch.device
) -> Tensor:
    """Return a (B, S) boolean mask where True means the position is NOT padding.

    Compares `sequences` against `padding_token` across all trailing dimensions
    and keeps positions where every element differs from the padding token.

    Args:
        sequences: Tensor of shape (B, S, ...).
        padding_token: Tensor broadcastable to the trailing dims of `sequences`.
        device: Target device for the resulting mask.

    Returns:
        (B, S) boolean Tensor where True indicates a non-padding position.
    """
    padding_token = padding_token.reshape(1, 1, *padding_token.shape)
    mask = torch.ne(sequences, padding_token).to(device)
    return mask.all(dim=tuple(range(2, sequences.ndim)))


# --- FlexAttention masking ---------------------------------------------------


def create_padding_mask(pads: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """Factory for a FlexAttention-compatible padding mask predicate.

    Given a (B, S) boolean pad tensor where True indicates padding, returns a
    function with signature (b, h, q_idx, kv_idx) -> bool mask that allows
    attention only when both query and key/value positions are not padding.

    Args:
        pads: (B, S) boolean Tensor where True = padding.

    Returns:
        A callable (b, h, q_idx, kv_idx) -> Tensor[bool] for FlexAttention.
    """

    def padding(b: int, h: int, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return ~pads[b, q_idx] & ~pads[b, kv_idx]

    return padding


def causal(b: int, h: int, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    """FlexAttention-compatible causal predicate (allow q >= k).
    
    Args:
        b: Batch index (unused).
        h: Head index (unused).
        q_idx: (N,) Tensor of query indices.
        kv_idx: (M,) Tensor of key/value indices.

    Returns:
        (N, M) boolean Tensor where True indicates allowed attention.
    """
    return q_idx >= kv_idx


# --- Vanilla (SDPA) attention masking ---------------------------------------


def create_causal_mask(sz: int, device: str | torch.device) -> Tensor:
    """Create a lower-triangular (causal) mask of shape (S, S) with 1.0 on/under diag.

    Note:
        This returns a float mask (0/1) to match common SDPA expectations;
        callers can convert to bool if needed.

    Args:
        sz: Size of the square mask (S).
        device: Target device for the resulting mask.   
    
    Returns:
        (S, S) float Tensor with 1.0 on and below the main diagonal, 0.0 elsewhere.
    """
    return torch.tril(torch.ones(sz, sz, dtype=torch.float32), diagonal=0).to(device)


def create_causal_padding_mask(
    causal_mask: Tensor, padding_mask: Tensor, n_head: int | None = None
) -> Tensor:
    """Combine padding and causal masks for SDPA.

    Args:
        causal_mask: (S, S) float/bool causal mask (typically lower-triangular).
        padding_mask: (B, S) boolean mask where True = padding.
        n_head: If provided, expand the result to (B, H, S, S).

    Returns:
        (B, S, S) or (B, H, S, S) mask tensor matching SDPA expectations.
    """
    B, S = padding_mask.shape

    # 1) Expand padding mask: (B, S) -> (B, 1, S) -> (B, S, S)
    pad_mask = padding_mask.unsqueeze(1).expand(-1, S, -1)  # (B, S, S)

    # 2) Expand causal mask: (S, S) -> (B, S, S)
    causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # (B, S, S)

    # 3) Combine (elementwise AND if bool; multiplication if float/bool)
    attn_mask = pad_mask * causal_mask  # (B, S, S)

    if n_head is not None:
        # 4) Optionally expand to (B, H, S, S)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, n_head, -1, -1)

    return attn_mask