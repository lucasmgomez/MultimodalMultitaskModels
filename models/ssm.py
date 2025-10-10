from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import Mamba2Config, Mamba2Model

from models.modules.masking import get_padding_mask
from models.modules.positionals import PositionalEncoding

_IMAGE_SPATIAL = 7  # spatial size used in flattening (7x7)


# --- Model: ASSM (Mamba2) -------------------------------------------------------

class ASSM(nn.Module):
    """ Action State-space model built on Mamba2 blocks for visual working-memory tasks.

    Args:
        hidden_dim: Hidden feature dimension.
        n_head: Number of independent kernels for selective scan.
        n_groups: Number of channel groups for selective scan.
        d_state: Dimension of the state representation.
        d_conv: Convolution kernel size for selective scan.
        dt_rank: Rank of the dynamic tokenization (passed via config).
        ffl_expand: Expansion factor for feedforward layers.
        device: Torch device.
        imgm_dim: Channels of the image encoder (input to Conv3d).
        insm_dim: Instruction embedding dimension.
        blocks: Number of Mamba layers (Transformer depth analog).
        output_dim: Number of output classes.
        max_tokens: Maximum number of instruction tokens.
        max_images: Maximum number of image tokens.
        dropout: Dropout probability for positional encodings and projections.

    Notes:
        - `dt_rank` is currently configured as `"auto"` to match the existing behavior.
        - `attention_mask` passed to `Mamba2Model` mirrors the original implementation.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        n_groups: int,
        d_state: int,
        d_conv: int,
        dt_rank: int,
        ffl_expand: int,
        device: str | torch.device,
        imgm_dim: int,
        insm_dim: int,
        blocks: int,
        output_dim: int,
        max_tokens: int,
        max_images: int,
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

        # Mamba config & model
        self.blocks = blocks
        self.mamba_cfg = Mamba2Config(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_heads=n_head,
            n_groups=n_groups,
            head_dim=hidden_dim * ffl_expand // n_head,
            state_size=d_state,
            conv_kernel=d_conv,
            time_step_rank="auto",  # keeps original behavior; ignores dt_rank arg
            expand=ffl_expand,
            num_hidden_layers=blocks,
            pad_token_id=0,
            rms_norm=True,
            use_cache=False,
            output_hidden_states=False,
        )
        self.mamba = Mamba2Model(self.mamba_cfg)

        # Compile cnnlayer
        self.cnnlayer = torch.compile(self.cnnlayer)

        # NOTE: You can try uncommenting and compiling, but huggingface implementation of Mamba2 seems to have issues
        # self.mamba = torch.compile(self.mamba)

    def create_masks(
        self, images: Tensor, image_pad: Tensor, ins_acts: Tensor, ins_pad: Tensor
    ) -> Tensor:
        """Create a padding mask for the concatenated instruction+image sequence.

        Args:
            images: (B, S_img, C, H, W) image sequence (pre-conv).
            image_pad: (B, S_img) padding (1 = pad).
            ins_acts: (B, S_txt, D) instruction sequence.
            ins_pad: (B, S_txt) padding (1 = pad).

        Returns:
            (B, S_txt + S_img) boolean padding mask.
        """
        # Make padding masks for images and instructions
        image_pad_mask = get_padding_mask(images, image_pad, self.device)
        ins_pad_mask = get_padding_mask(ins_acts, ins_pad, self.device)

        # Convert to boolean
        image_pad_mask = image_pad_mask.to(torch.bool)
        ins_pad_mask = ins_pad_mask.to(torch.bool)

        # Combine padding masks (instructions first)
        pad_mask = torch.cat((ins_pad_mask, image_pad_mask), dim=1)
        return pad_mask

    def forward(self, image_embs: Tensor, instruction_embs: Tensor, pad_mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            image_embs: (B, S_img, C, H, W) image embeddings before Conv3d.
            instruction_embs: (B, S_txt, D) instruction embeddings.
            pad_mask: (B, S_txt + S_img) boolean padding mask.

        Returns:
            (B, S_img, output_dim) — logits for image positions only.
        """
        bs = image_embs.shape[0]
        n_tokens = instruction_embs.shape[1]

        # Conv on image sequence: (B, S_img, C, H, W) -> (B, C_out, S_img, H, W) -> (B, S_img, C_out, H, W)
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
        hidden = torch.cat((ins_out, images_out), dim=1)

        # Mamba forward
        hidden = self.mamba(inputs_embeds=hidden, attention_mask=pad_mask).last_hidden_state

        # Output head
        out = self.hidden2output(hidden)

        # Return only image positions
        return out[:, n_tokens:, :]