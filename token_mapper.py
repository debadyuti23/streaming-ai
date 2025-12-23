import torch
import torch.nn as nn

class TokenMapper(nn.Module):
    """
    Maps encoder tokens (e.g., VideoMAE last_hidden_state) into cross-attention
    context tokens for the video generator UNet.
    Expected input: (batch, seq_len, src_dim)
    Output: (batch, num_cond_tokens, ctx_dim)
    """
    def __init__(
        self,
        src_dim: int,
        ctx_dim: int,
        num_cond_tokens: int = 1,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_cond_tokens = num_cond_tokens
        self.src_dim = src_dim
        self.ctx_dim = ctx_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = nn.Sequential(
            nn.LayerNorm(src_dim),
            nn.Linear(src_dim, hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool across sequence dimension to a fixed number of conditional tokens
        self.sequence_pool = nn.AdaptiveAvgPool1d(num_cond_tokens)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, ctx_dim),
        )

    def forward(self, tokens):
        """
        tokens: (B, N, src_dim)
        returns: (B, num_cond_tokens, ctx_dim)
        """
        # align tokens with module device/dtype to avoid device mismatch
        #print("tokens shape:", tokens.shape)
        #print("src_dim:", self.src_dim)
        #print("ctx_dim:", self.ctx_dim)
        #print("num_cond_tokens:", self.num_cond_tokens)
        #print("hidden_dim:", self.hidden_dim)
        #print("num_layers:", self.num_layers)
        #print("num_heads:", self.num_heads)
        if tokens.dim() != 3 or tokens.size(-1) != self.src_dim:
            raise ValueError(f"Expected tokens of shape (B, N, {self.src_dim}), got {tuple(tokens.shape)}")

        x = self.input_proj(tokens)          # (B, N, hidden_dim)
        x = self.encoder(x)                  # (B, N, hidden_dim)

        # Pool across sequence length N -> num_cond_tokens
        x = x.transpose(1, 2)                # (B, hidden_dim, N)
        x = self.sequence_pool(x)            # (B, hidden_dim, num_cond_tokens)
        x = x.transpose(1, 2)                # (B, num_cond_tokens, hidden_dim)

        cond_ctx = self.output_proj(x)       # (B, num_cond_tokens, ctx_dim)
        return cond_ctx