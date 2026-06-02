import torch
import torch.nn as nn


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        return self.dropout(x + self.pos_embedding) if self.dropout else x + self.pos_embedding


class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, num_heads, dropout=attn_dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        if self.dropout:
            x = self.dropout(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super().__init__()
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(emb_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.pos_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.norm(x)
