import math

import torch
from torch import nn


class SoftbombTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_tracks=512,
        use_positional_encoding=False,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.position = nn.Parameter(torch.zeros(1, max_tracks + 1, d_model))
        else:
            self.register_parameter("position", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        if self.position is not None:
            nn.init.normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        tokens = self.input_projection(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask.bool()], dim=1)
        if self.position is not None:
            tokens = tokens + self.position[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        logits = self.head(encoded[:, 0]).squeeze(-1)
        return logits


def build_model_from_config(config, input_dim, max_tracks):
    model_config = dict(config["training"]["model"])
    return SoftbombTransformer(input_dim=input_dim, max_tracks=max_tracks, **model_config)

