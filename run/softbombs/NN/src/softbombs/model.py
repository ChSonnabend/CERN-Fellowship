from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch import nn


def sdpa_backend_context(attention_backend):
    backend = str(attention_backend or "auto").lower()
    if backend == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return nullcontext()

    backend_map = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "flash_attention": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    if backend not in backend_map:
        raise ValueError(f"Unknown attention_backend '{attention_backend}'")
    return sdpa_kernel(backend_map[backend])


class FlashSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        attention_backend="auto",
        allow_attention_fallback=True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.attention_backend = attention_backend
        self.allow_attention_fallback = bool(allow_attention_fallback)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = float(dropout)
        self._backend_fallback_warned = False

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        mask = mask.bool()
        attn_mask = None if bool(mask.all().item()) else mask[:, None, None, :]
        dropout_p = self.dropout_p if self.training else 0.0
        attention_backend = self.attention_backend if query.is_cuda else "auto"
        try:
            with sdpa_backend_context(attention_backend):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        except RuntimeError as exc:
            if str(attention_backend).lower() == "auto":
                raise
            if not self._backend_fallback_warned:
                print(f"Falling back to automatic SDPA backend because '{attention_backend}' failed")
                self._backend_fallback_warned = True
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.dropout(self.out_projection(attended))

    def attention_weights(self, x, mask):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, _ = qkv.unbind(dim=0)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        mask = mask.bool()
        if not bool(mask.all().item()):
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        attention_backend="auto",
        allow_attention_fallback=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = FlashSelfAttention(
            d_model,
            n_heads,
            dropout,
            attention_backend,
            allow_attention_fallback=allow_attention_fallback,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = x + self.self_attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def attention_weights(self, x, mask):
        return self.self_attention.attention_weights(self.norm1(x), mask)


class FlashTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward,
        dropout,
        attention_backend="auto",
        allow_attention_fallback=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout,
                    attention_backend,
                    allow_attention_fallback=allow_attention_fallback,
                )
                for _ in range(int(n_layers))
            ]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CrossAttentionCompressionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.latent_norm = nn.LayerNorm(d_model)
        self.track_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, latents, tracks, track_mask):
        query = self.latent_norm(latents)
        key_value = self.track_norm(tracks)
        attended, _ = self.cross_attn(
            query,
            key_value,
            key_value,
            key_padding_mask=~track_mask.bool(),
            need_weights=False,
        )
        latents = latents + self.dropout(attended)
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class TrackAttentionCompressor(nn.Module):
    def __init__(
        self,
        d_model,
        n_latents=256,
        n_heads=8,
        n_layers=1,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.n_latents = int(n_latents)
        self.latents = nn.Parameter(torch.zeros(1, self.n_latents, d_model))
        self.layers = nn.ModuleList(
            [
                CrossAttentionCompressionLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(int(n_layers))
            ]
        )

    def reset_parameters(self):
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, tracks, track_mask):
        latents = self.latents.expand(tracks.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, tracks, track_mask)
        latent_mask = torch.ones(
            (tracks.shape[0], self.n_latents),
            dtype=torch.bool,
            device=tracks.device,
        )
        return latents, latent_mask


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
        attention_backend="auto",
        allow_attention_fallback=True,
        track_compressor=None,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        compressor_config = dict(track_compressor or {})
        compressor_enabled = bool(compressor_config.pop("enabled", False))
        self.track_compressor = None
        encoder_token_capacity = max_tracks
        if compressor_enabled:
            n_latents = int(compressor_config.pop("n_latents", 256))
            self.track_compressor = TrackAttentionCompressor(
                d_model=d_model,
                n_latents=n_latents,
                n_heads=int(compressor_config.pop("n_heads", n_heads)),
                n_layers=int(compressor_config.pop("n_layers", 1)),
                dim_feedforward=int(compressor_config.pop("dim_feedforward", dim_feedforward)),
                dropout=float(compressor_config.pop("dropout", dropout)),
            )
            if compressor_config:
                unknown = ", ".join(sorted(compressor_config))
                raise ValueError(f"Unknown track_compressor config keys: {unknown}")
            encoder_token_capacity = n_latents

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.position = nn.Parameter(torch.zeros(1, encoder_token_capacity + 1, d_model))
        else:
            self.register_parameter("position", None)

        self.encoder = FlashTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_backend=attention_backend,
            allow_attention_fallback=allow_attention_fallback,
        )
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
        if self.track_compressor is not None:
            self.track_compressor.reset_parameters()
        if self.position is not None:
            nn.init.normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encoder_inputs(self, x, mask):
        mask = mask.bool()
        max_valid_tracks = int(mask.sum(dim=1).max().item())
        if max_valid_tracks < mask.shape[1]:
            x = x[:, :max_valid_tracks, :]
            mask = mask[:, :max_valid_tracks]
        tokens = self.input_projection(x)
        token_mask = mask
        if self.track_compressor is not None:
            tokens, token_mask = self.track_compressor(tokens, token_mask)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((token_mask.shape[0], 1), dtype=torch.bool, device=token_mask.device)
        full_mask = torch.cat([cls_mask, token_mask], dim=1)
        if self.position is not None:
            tokens = tokens + self.position[:, : tokens.shape[1], :]
        return tokens, full_mask

    def forward(self, x, mask):
        tokens, full_mask = self.encoder_inputs(x, mask)
        encoded = self.encoder(tokens, full_mask)
        logits = self.head(encoded[:, 0]).squeeze(-1)
        return logits


def build_model_from_config(config, input_dim, max_tracks):
    model_config = dict(config["training"]["model"])
    return SoftbombTransformer(input_dim=input_dim, max_tracks=max_tracks, **model_config)
