from __future__ import annotations

from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention


# --- Attention Modules -------------------------------------------------------

class MultiheadFlexAttention(nn.Module):
    """Multi-head self-attention using torch.flex_attention.

    Args:
        d_in: Input feature dimension.
        d_out: Output feature dimension; must be divisible by `n_heads`.
        n_heads: Number of attention heads.
        bias: Whether to use Q/K/V biases.

    Shapes:
        x: (B, S, d_in)
        attn_mask: BlockMask compatible with flex_attention.

    Returns:
        Tensor of shape (B, S, d_out).
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int, bias: bool = False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.d_out = d_out

        self.in_proj = nn.Linear(d_in, 3 * d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.flex_attention = flex_attention

    def forward(self, x: Tensor, attn_mask: BlockMask) -> Tensor:
        """Apply multi-head attention.

        Args:
            x: (B, S, d_in)
            attn_mask: BlockMask for flex_attention.

        Returns:
            (B, S, d_out)
        """
        batch_size, max_seq_len, _ = x.shape

        # Create stacked qkv via input projection
        qkv = self.in_proj(x)  # (B, S, 3 * d_out)

        # Split qkv and divide d_out into heads
        qkv = qkv.view(batch_size, max_seq_len, 3, self.n_heads, self.d_head)  # (B, S, 3, H, Dh)

        # Permute shape of qkv for flex_attention
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, Dh)

        # Get queries, keys, values
        queries, keys, values = qkv  # each: (B, H, S, Dh)

        # Calculate attention via flex_attention
        attn = self.flex_attention(queries, keys, values, block_mask=attn_mask)  # (B, H, S, Dh)

        # Merge heads into d_out
        attn = attn.transpose(1, 2).contiguous().view(batch_size, max_seq_len, self.d_out)

        # Output projection
        return self.out_proj(attn)


class MultiheadAttention(nn.Module):
    """Multi-head self-attention using scaled_dot_product_attention (SDPA).

    Args:
        d_in: Input feature dimension.
        d_out: Output feature dimension; must be divisible by `n_heads`.
        n_heads: Number of attention heads.
        bias: Whether to use Q/K/V biases.

    Shapes:
        x: (B, S, d_in)
        attn_mask: (B or 1, H or 1, S, S) or broadcastable equivalent.

    Returns:
        Tensor of shape (B, S, d_out).
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int, bias: bool = False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.d_out = d_out

        self.in_proj = nn.Linear(d_in, 3 * d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Apply multi-head attention.

        Args:
            x: (B, S, d_in)
            attn_mask: (B or 1, H or 1, S, S) or broadcastable.

        Returns:
            (B, S, d_out)
        """
        batch_size, max_seq_len, _ = x.shape

        # Create stacked qkv via input projection
        qkv = self.in_proj(x)  # (B, S, 3 * d_out)

        # Split qkv and divide d_out into heads
        qkv = qkv.view(batch_size, max_seq_len, 3, self.n_heads, self.d_head)  # (B, S, 3, H, Dh)

        # Permute shape of qkv for attention
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, Dh)

        # Get queries, keys, values
        queries, keys, values = qkv  # each: (B, H, S, Dh)

        # Calculate attention via scaled_dot_product_attention
        attn = nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask)

        # Merge heads into d_out
        attn = attn.transpose(1, 2).contiguous().view(batch_size, max_seq_len, self.d_out)

        # Output projection
        return self.out_proj(attn)