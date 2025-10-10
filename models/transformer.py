from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import ModuleList
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
)

from models.modules.masking import (
    causal,
    create_causal_mask,
    create_causal_padding_mask,
    create_padding_mask,
    get_padding_mask,
)
from models.modules.positionals import PositionalEncoding
from models.modules.attentions import MultiheadFlexAttention, MultiheadAttention

_IMAGE_SPATIAL = 7  # spatial size used in flattening (7x7)


# --- Transformer Block -------------------------------------------------------

class TFBlock(nn.Module):
    """Transformer block: LN → MHA → Dropout → Add → LN → MLP → Dropout → Add.

    Args:
        d_in: Input feature dimension.
        d_model: Model feature dimension.
        n_heads: Number of attention heads.
        attn_cls: Attention module class to instantiate (MultiheadFlexAttention or MultiheadAttention).
        moe: If True, use Mixture-of-Experts feedforward.
        n_experts: Number of experts (when `moe=True`).
        top_k: Number of experts to select per token (when `moe=True`).
        ffn: Feedforward module (MLP) used when `moe=False`; also cloned per expert when `moe=True`.
        dropout: Dropout probability.
        layer_norm_eps: Epsilon for LayerNorm.
        bias: Whether to use Q/K/V biases in attention.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        attn_cls: type[nn.Module],
        moe: bool,
        n_experts: int,
        top_k: int,
        ffn: nn.Module,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()

        # MultiheadAttention (either MultiheadFlexAttention or MultiheadAttention)
        self.mha = attn_cls(d_in, d_model, n_heads, bias=bias)

        # LayerNorms
        self.layernorm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Feedforward network (MLP / MoE)
        self.moe = moe
        self.top_k = top_k
        if moe:
            self.gate = nn.Linear(d_model, n_experts, bias=False)
            self.experts = _get_clones(ffn, n_experts)
        else:
            self.ffn = ffn
            # TODO: register_hook on FFN if needed.

        # Attention dropout
        self.dropout_attn = nn.Dropout(dropout)

    def self_attention(self, x: Tensor, attn_mask: BlockMask | Tensor) -> Tensor:
        return self.dropout_attn(self.mha(x, attn_mask))

    def feedforward(self, x: Tensor) -> Tensor:
        if self.moe:
            batch, seq_len, d_model = x.shape
            N = batch * seq_len

            # Flatten to (N, d_model) for fast per-token processing
            tokens = x.reshape(-1, d_model)

            # Compute expert gate logits and take top-k weights
            gate_logits = self.gate(tokens)  # (N, n_experts)
            topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)  # (N, top_k)

            # Initialize combined output
            combined = tokens.new_zeros(tokens.shape)  # (N, d_model)

            # For each expert, gather tokens, run MLP, weight & scatter-add
            for e_i, expert in enumerate(self.experts):
                mask = topk_idx == e_i  # (N, top_k)
                if not mask.any():  # no tokens route through this expert
                    continue

                # Per-token weight for this expert
                w = (mask.float() * topk_weights).sum(dim=1)  # (N,)
                sel = w > 0  # tokens that route through this expert

                inp = tokens[sel]  # (⊂N, d_model)
                out = expert(inp) * w[sel].unsqueeze(1)  # (⊂N, d_model)
                combined[sel] += out

            x = combined.view(batch, seq_len, d_model)  # (B, S, D)
        else:
            x = self.ffn(x)  # vanilla feedforward
        return x

    def forward(self, x: Tensor, attn_mask: BlockMask | Tensor) -> Tensor:
        # Pre-attention LN
        x = self.layernorm1(x)

        # Self-attention + residual
        x = x + self.self_attention(x, attn_mask)

        # Pre-FFN LN
        x = self.layernorm2(x)

        # Feedforward + residual
        x = x + self.feedforward(x)
        return x


# --- Model: FlexATF ----------------------------------------------------------

class FlexATF(nn.Module):
    """Action Transformer model for visual working-memory tasks using MultiheadFlexAttention.

    Args:
        hidden_dim: Hidden feature dimension.
        device: Torch device.
        imgm_dim: Image encoder channel dimension (input to Conv3d).
        insm_dim: Instruction embedding dimension.
        n_head: Number of attention heads per block.
        blocks: Number of Transformer blocks.
        output_dim: Number of output classes.
        max_tokens: Maximum number of instruction tokens.
        max_images: Maximum number of image tokens.
        dropout: Dropout probability.
        ffn: Feedforward module used in TF blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        device: str | torch.device,
        imgm_dim: int,
        insm_dim: int,
        n_head: int,
        blocks: int,
        output_dim: int,
        max_tokens: int,
        max_images: int,
        dropout: float,
        ffn: nn.Module,
    ):
        super().__init__()

        # Device & sequence meta
        self.device = device
        self.max_images = max_images
        self.max_tokens = max_tokens
        self.insm_dim = insm_dim

        # Convolutional layer
        self.cnnlayer = nn.Conv3d(in_channels=imgm_dim, out_channels=hidden_dim, kernel_size=1)

        # Model dimensions
        self.input_dim = hidden_dim * _IMAGE_SPATIAL * _IMAGE_SPATIAL
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Linear layers (instructions, in, & out)
        self.ins_hidden = nn.Linear(insm_dim, hidden_dim)
        self.image_hidden = nn.Linear(self.input_dim, hidden_dim)
        self.hidden2output = nn.Linear(self.hidden_dim, self.output_dim)

        # Layer norms
        self.ln_image = nn.LayerNorm(self.input_dim)
        self.ln_ins = nn.LayerNorm(insm_dim)

        # Positional encodings
        self.image_pos_emb = PositionalEncoding(hidden_dim, dropout, max_images)
        self.ins_pos_emb = PositionalEncoding(hidden_dim, dropout, max_tokens)

        # Transformer
        self.tf_block = TFBlock(
            d_in=hidden_dim,
            d_model=hidden_dim,
            n_heads=n_head,
            attn_cls=MultiheadFlexAttention,
            moe=False,
            n_experts=0,
            top_k=0,
            ffn=ffn,
            dropout=dropout,
        ).to(device)
        self.transformer = _get_clones(self.tf_block, blocks)

        # Compile cnnlayer
        self.cnnlayer = torch.compile(self.cnnlayer)

        # NOTE: torch.compile on transformer blocks may be unstable; keep disabled.
        # for i in range(len(self.transformer)):
        #     self.transformer[i] = torch.compile(self.transformer[i])

        # Pre-compile the create_block_mask function
        self.create_block_mask = torch.compile(create_block_mask)

    def create_masks(
        self,
        images: Tensor,
        image_pad: Tensor,
        ins_acts: Tensor,
        ins_pad: Tensor,
    ) -> BlockMask:
        """Create a causal + padding BlockMask for flex attention.

        Args:
            images: (B, S_img, C, H, W) image sequence (pre-conv).
            image_pad: (B, S_img) padding (1 = pad).
            ins_acts: (B, S_txt, D) instruction sequence.
            ins_pad: (B, S_txt) padding (1 = pad).

        Returns:
            BlockMask suitable for flex_attention over concatenated sequence.
        """
        bs = images.shape[0]
        seq_len = images.shape[1] + ins_acts.shape[1]

        # Padding masks
        image_pad_mask = get_padding_mask(images, image_pad, self.device)
        ins_pad_mask = get_padding_mask(ins_acts, ins_pad, self.device)

        # Combine padding masks
        pad_mask = torch.cat((ins_pad_mask, image_pad_mask), dim=1)

        # Causal + padding mask
        padding_mask = create_padding_mask(pad_mask)
        combined_mask = and_masks(causal, padding_mask)

        causal_padding_mask = create_block_mask(
            combined_mask,
            B=bs,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=self.device,
        )
        return causal_padding_mask

    def forward(self, image_embs: Tensor, instruction_embs: Tensor, causal_pad_mask: BlockMask) -> Tensor:
        """Forward pass.

        Args:
            image_embs: (B, S_img, C, H, W) image embeddings before Conv3d.
            instruction_embs: (B, S_txt, D) instruction embeddings.
            causal_pad_mask: BlockMask from `create_masks`.

        Returns:
            (B, S_img, output_dim) — logits for image positions only.
        """
        bs = image_embs.shape[0]

        # Conv on image sequence: (B, S_img, C, H, W) → (B, C_out, S_img, H, W) → (B, S_img, C_out, H, W)
        images_out = self.cnnlayer(image_embs.transpose(1, 2)).transpose(1, 2)

        # Flatten spatial dims
        images_out = images_out.reshape(bs, self.max_images, -1)  # (B, S_img, hidden_dim * 7 * 7)

        # LN + projection
        images_out = self.image_hidden(self.ln_image(images_out))
        ins_out = self.ins_hidden(self.ln_ins(instruction_embs))

        # Positional encodings
        images_out = self.image_pos_emb(images_out)
        ins_out = self.ins_pos_emb(ins_out)

        # Concatenate tokens (instructions first)
        hidden_x = torch.cat((ins_out, images_out), dim=1)

        # Transformer encoder
        for block in self.transformer:
            hidden_x = block(hidden_x, attn_mask=causal_pad_mask)

        # Output head
        out = self.hidden2output(hidden_x)

        # Return only image positions
        return out[:, self.max_tokens :, :]


# --- Model: ATF (SDPA attention) ---------------------------------------------

class ATF(nn.Module):
    """Action Transformer model for visual working-memory tasks using MultiheadAttention (SDPA).

    Args:
        hidden_dim: Hidden feature dimension.
        device: Torch device.
        imgm_dim: Image encoder channel dimension (input to Conv3d).
        insm_dim: Instruction embedding dimension.
        n_head: Number of attention heads per block.
        blocks: Number of Transformer blocks.
        output_dim: Number of output classes.
        max_tokens: Maximum number of instruction tokens.
        max_images: Maximum number of image tokens.
        moe: If True, use Mixture-of-Experts FFN.
        n_experts: Number of experts (when `moe=True`).
        top_k: Number of experts to select per token (when `moe=True`).
        ffn: Feedforward module used in TF blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        device: str | torch.device,
        imgm_dim: int,
        insm_dim: int,
        n_head: int,
        blocks: int,
        output_dim: int,
        max_tokens: int,
        max_images: int,
        moe: bool,
        n_experts: int,
        top_k: int,
        ffn: nn.Module,
        dropout: float,
    ):
        super().__init__()

        # Device & sequence meta
        self.device = device
        self.max_images = max_images
        self.max_tokens = max_tokens
        self.insm_dim = insm_dim

        # Convolutional layer
        self.cnnlayer = nn.Conv3d(in_channels=imgm_dim, out_channels=hidden_dim, kernel_size=1)

        # Model dimensions
        self.input_dim = hidden_dim * _IMAGE_SPATIAL * _IMAGE_SPATIAL
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Linear layers (instructions, in, & out)
        self.ins_hidden = nn.Linear(insm_dim, hidden_dim)
        self.image_hidden = nn.Linear(self.input_dim, hidden_dim)
        self.hidden2output = nn.Linear(self.hidden_dim, self.output_dim)

        # Layer norms
        self.ln_image = nn.LayerNorm(self.input_dim)
        self.ln_ins = nn.LayerNorm(insm_dim)

        # Positional encodings
        self.image_pos_emb = PositionalEncoding(hidden_dim, dropout, max_images)
        self.ins_pos_emb = PositionalEncoding(hidden_dim, dropout, max_tokens)

        # Transformer
        self.n_head = n_head
        self.blocks = blocks
        self.tf_block = TFBlock(
            d_in=hidden_dim,
            d_model=hidden_dim,
            n_heads=n_head,
            attn_cls=MultiheadAttention,
            moe=moe,
            n_experts=n_experts,
            top_k=top_k,
            ffn=ffn,
            dropout=dropout,
        ).to(device)
        self.transformer = _get_clones(self.tf_block, blocks)

        # Compile cnnlayer
        self.cnnlayer = torch.compile(self.cnnlayer, mode="reduce-overhead")

        # Compile transformer blocks (optional; may be unstable depending on PyTorch version)
        for i in range(len(self.transformer)):
            self.transformer[i] = torch.compile(self.transformer[i])

    def create_masks(
        self, images: Tensor, image_pad: Tensor, ins_acts: Tensor, ins_pad: Tensor
    ) -> Tensor:
        """Create a causal + padding mask tensor for SDPA.

        Args:
            images: (B, S_img, C, H, W) image sequence (pre-conv).
            image_pad: (B, S_img) padding (1 = pad).
            ins_acts: (B, S_txt, D) instruction sequence.
            ins_pad: (B, S_txt) padding (1 = pad).

        Returns:
            Attention mask tensor suitable for SDPA with shape broadcastable to
            (B, H, S, S).
        """
        seq_len = images.shape[1] + ins_acts.shape[1]

        # Padding masks
        image_pad_mask = get_padding_mask(images, image_pad, self.device)
        ins_pad_mask = get_padding_mask(ins_acts, ins_pad, self.device)

        # Combine padding masks
        pad_mask = torch.cat((ins_pad_mask, image_pad_mask), dim=1)

        # Causal mask
        causal_mask = create_causal_mask(seq_len, self.device)

        # Convert to bool
        pad_mask = pad_mask.to(torch.bool)
        causal_mask = causal_mask.to(torch.bool)

        # Combine causal and padding masks
        causal_padding_mask = create_causal_padding_mask(
            causal_mask, pad_mask, self.n_head
        )

        return causal_padding_mask

    def forward(self, image_embs: Tensor, instruction_embs: Tensor, causal_pad_mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            image_embs: (B, S_img, C, H, W) image embeddings before Conv3d.
            instruction_embs: (B, S_txt, D) instruction embeddings.
            causal_pad_mask: SDPA mask tensor broadcastable to (B, H, S, S).

        Returns:
            (B, S_img, output_dim) — logits for image positions only.
        """
        bs = image_embs.shape[0]
        n_tokens = instruction_embs.shape[1]

        # Conv on image sequence
        images_out = self.cnnlayer(image_embs.transpose(1, 2)).transpose(1, 2)

        # Flatten spatial dims
        images_out = images_out.reshape(bs, self.max_images, -1)

        # LN + projection
        images_out = self.image_hidden(self.ln_image(images_out))
        ins_out = self.ins_hidden(self.ln_ins(instruction_embs))

        # Positional encodings
        images_out = self.image_pos_emb(images_out)
        ins_out = self.ins_pos_emb(ins_out)

        # Concatenate tokens (instructions first)
        hidden_x = torch.cat((ins_out, images_out), dim=1)

        # Transformer
        for block in self.transformer:
            hidden_x = block(hidden_x, attn_mask=causal_pad_mask)

        # Output head
        out = self.hidden2output(hidden_x)

        # Return only image positions
        return out[:, n_tokens:, :]


# --- Helper Utilities --------------------------------------------------------

def _get_clones(module: nn.Module, N: int) -> ModuleList:
    """Deep-copy a module N times into a ModuleList."""
    return ModuleList([copy.deepcopy(module) for _ in range(N)])