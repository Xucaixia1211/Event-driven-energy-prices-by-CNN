"""Neural benchmark classifiers used in the empirical comparison."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResNetTypeCNNBlock(nn.Module):
    """One residual block for the ResNet-type CNN benchmark."""

    def __init__(self, channels: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ResNetTypeCNNBinaryClassifier(nn.Module):
    """ResNet-type CNN used as an empirical benchmark."""

    def __init__(
        self,
        input_length: int,
        channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_length < 1:
            raise ValueError("input_length must be positive.")
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResNetTypeCNNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


class MLPBinaryClassifier(nn.Module):
    """Dense neural benchmark with a configurable number of hidden layers."""

    def __init__(
        self,
        input_length: int,
        hidden_layers: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_length
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.net(x)


class BlockSparseLinear(nn.Module):
    """Linear layer with a fixed block-diagonal parameter mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_blocks: int = 4,
        bias: bool = True,
        block_bandwidth: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        in_block_ids = torch.div(
            torch.arange(in_features) * n_blocks,
            in_features,
            rounding_mode="floor",
        )
        out_block_ids = torch.div(
            torch.arange(out_features) * n_blocks,
            out_features,
            rounding_mode="floor",
        )
        mask = (
            torch.abs(out_block_ids[:, None] - in_block_ids[None, :])
            <= block_bandwidth
        ).float()
        self.register_buffer("mask", mask)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class BlockSparseMLPBinaryClassifier(nn.Module):
    """MLP benchmark whose hidden maps use fixed block-sparse masks."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        n_blocks: int = 4,
        dropout: float = 0.2,
        block_bandwidth: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_length
        for _ in range(hidden_layers):
            layers.extend(
                [
                    BlockSparseLinear(
                        in_dim,
                        hidden_dim,
                        n_blocks=n_blocks,
                        block_bandwidth=block_bandwidth,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.net(x)


class LSTMBinaryClassifier(nn.Module):
    """LSTM benchmark followed by a binary classifier."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        x_sequence = x.transpose(1, 2)
        _, (hidden_state, _) = self.lstm(x_sequence)
        return self.classifier(hidden_state[-1])


class NumericFeatureTokenizer(nn.Module):
    """Map scalar time-point features to tokens for the FT-Transformer."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls_token = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls_token, tokens], dim=1)


class FTTransformerBinaryClassifier(nn.Module):
    """Feature-tokenizer Transformer benchmark for fixed-length windows."""

    def __init__(
        self,
        input_length: int,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(input_length, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])
