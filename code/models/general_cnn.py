"""General parallel-branch CNN components used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn


PoolingKind = Literal["none", "max", "adaptive_max", "average", "adaptive_average"]


@dataclass(frozen=True)
class ConvBlockConfig:
    """Configuration for one convolution, activation and pooling block."""

    out_channels: int
    kernel_size: int
    padding: int | str = 0
    pooling: PoolingKind = "none"
    pool_size: int = 2
    batch_normalisation: bool = False


class ConvPoolBranch1D(nn.Module):
    """A configurable one-dimensional convolution and pooling branch."""

    def __init__(
        self,
        input_channels: int,
        blocks: Sequence[ConvBlockConfig],
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive.")
        if not blocks:
            raise ValueError("A convolutional branch must contain at least one block.")

        layers: list[nn.Module] = []
        current_channels = int(input_channels)
        for block in blocks:
            if block.out_channels < 1 or block.kernel_size < 1:
                raise ValueError("Channel counts and kernel sizes must be positive.")
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=block.out_channels,
                        kernel_size=block.kernel_size,
                        padding=block.padding,
                    ),
                    nn.ReLU(),
                ]
            )
            if block.batch_normalisation:
                layers.append(nn.BatchNorm1d(block.out_channels))

            if block.pooling == "max":
                layers.append(nn.MaxPool1d(block.pool_size))
            elif block.pooling == "adaptive_max":
                layers.append(nn.AdaptiveMaxPool1d(1))
            elif block.pooling == "average":
                layers.append(nn.AvgPool1d(block.pool_size))
            elif block.pooling == "adaptive_average":
                layers.append(nn.AdaptiveAvgPool1d(1))
            elif block.pooling != "none":
                raise ValueError(f"Unsupported pooling operation: {block.pooling!r}")
            current_channels = block.out_channels

        self.layers = nn.Sequential(*layers)
        self.output_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReLUClassifierHead(nn.Module):
    """A ReLU multilayer head ending in unnormalised output scores."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("Head dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            if hidden_dim < 1:
                raise ValueError("Hidden dimensions must be positive.")
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class AffineHead(nn.Module):
    """An affine head for a pooled feature vector."""

    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class IdentityHead(nn.Module):
    """Return pooled features, optionally squeezing a scalar feature."""

    def __init__(self, squeeze_scalar: bool = False) -> None:
        super().__init__()
        self.squeeze_scalar = bool(squeeze_scalar)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.squeeze_scalar and features.shape[1] == 1:
            return features[:, 0]
        return features


class GeneralCNN(nn.Module):
    """Combine one or more feature branches with a common classifier head."""

    def __init__(self, branches: Sequence[nn.Module], head: nn.Module) -> None:
        super().__init__()
        if not branches:
            raise ValueError("GeneralCNN requires at least one branch.")
        self.branches = nn.ModuleList(branches)
        self.head = head

    @staticmethod
    def _as_feature_matrix(branch_output: torch.Tensor) -> torch.Tensor:
        if branch_output.ndim == 1:
            return branch_output.unsqueeze(1)
        if branch_output.ndim < 1:
            raise ValueError("A branch must retain the batch dimension.")
        return branch_output.flatten(start_dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        branch_features = [
            self._as_feature_matrix(branch(x)) for branch in self.branches
        ]
        batch_sizes = {features.shape[0] for features in branch_features}
        if len(batch_sizes) != 1:
            raise ValueError("All branches must return the same batch size.")
        return torch.cat(branch_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))
