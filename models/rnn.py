from __future__ import annotations

import torch
from torch import Tensor, nn

_IMAGE_SPATIAL = 7  # spatial size used in flattening (7x7)


# --- Model: ARNN (LSTM) -------------------------------------------------------

class ARNN(nn.Module):
    """Recurrent Action model (LSTM) for visual working-memory tasks.

    Args:
        hidden_dim: Hidden/state dimension of the model and LSTM.
        device: Torch device.
        imgm_dim: Channels of the image encoder (input to Conv3d).
        insm_dim: Instruction embedding dimension.
        blocks: Number of stacked LSTM layers.
        output_dim: Number of output classes.
        max_images: Maximum number of image tokens.
        dropout: Dropout between LSTM layers (passed to LSTM constructor).

    Shapes:
        image_embs: (B, S_img, C, H, W)
        instruction_embs: (B, S_txt, D_ins)
        returns: (logits, hidden)
            logits: (B, S_txt + S_img, output_dim)
            hidden: (B, S_txt + S_img, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        device: str | torch.device,
        imgm_dim: int,
        insm_dim: int,
        blocks: int,
        output_dim: int,
        max_images: int,
        dropout: float,
    ):
        super().__init__()

        # Device & sequence meta
        self.device = device
        self.max_images = max_images
        self.insm_dim = insm_dim

        # Convolutional layer (1x1x1) to map image channels -> hidden_dim
        self.cnnlayer = nn.Conv3d(in_channels=imgm_dim, out_channels=hidden_dim, kernel_size=1)

        # Dimensions
        self.input_dim = hidden_dim * _IMAGE_SPATIAL * _IMAGE_SPATIAL
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projections (instructions, images, output)
        self.ins_hidden = nn.Linear(insm_dim, hidden_dim)
        self.image_hidden = nn.Linear(self.input_dim, hidden_dim)
        self.hidden2output = nn.Linear(self.hidden_dim, self.output_dim)

        # Layer norms
        self.ln_image = nn.LayerNorm(self.input_dim)
        self.ln_ins = nn.LayerNorm(insm_dim)

        # LSTM stack
        self.blocks = blocks
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=self.blocks,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
            proj_size=0,
        )

        # Compile conv (safe)
        self.cnnlayer = torch.compile(self.cnnlayer)

        # Compile LSTM (may vary by PyTorch/version/kernel; keep if it works for you)
        self.lstm = torch.compile(self.lstm)

    def forward(self, image_embs: Tensor, instruction_embs: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            image_embs: (B, S_img, C, H, W)
            instruction_embs: (B, S_txt, D_ins)

        Returns:
            logits: (B, S_txt + S_img, output_dim)
            hidden: (B, S_txt + S_img, hidden_dim)
        """
        bs = image_embs.shape[0]

        # Image path: (B, S_img, C, H, W) -> (B, C_out, S_img, H, W) -> (B, S_img, C_out, H, W)
        images_out = self.cnnlayer(image_embs.transpose(1, 2)).transpose(1, 2)

        # Flatten spatial dims -> (B, S_img, hidden_dim * 7 * 7)
        images_out = images_out.reshape(bs, self.max_images, -1)

        # Normalize + project to model dim
        images_out = self.image_hidden(self.ln_image(images_out))
        ins_out = self.ins_hidden(self.ln_ins(instruction_embs))

        # Concatenate tokens: instructions first
        hidden = torch.cat((ins_out, images_out), dim=1)

        # LSTM over sequence
        hidden, _ = self.lstm(hidden)

        # Output head
        out = self.hidden2output(hidden)

        return out, hidden