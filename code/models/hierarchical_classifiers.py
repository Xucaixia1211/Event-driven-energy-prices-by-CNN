"""CNN classifiers used at each stage of the hierarchical application."""

from __future__ import annotations

from dataclasses import dataclass

from .general_cnn import (
    ConvBlockConfig,
    ConvPoolBranch1D,
    GeneralCNN,
    ReLUClassifierHead,
)


@dataclass(frozen=True)
class HierarchicalCNNConfig:
    """Configuration used by the empirical Stage 1 and Stage 2 classifiers."""

    input_length: int
    input_channels: int = 1
    channel_sizes: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (3, 3, 3)
    poolings: tuple[str, ...] = ("max", "max", "adaptive_max")
    head_hidden_dims: tuple[int, ...] = (64,)
    num_classes: int = 2
    dropout: float = 0.3
    batch_normalisation: bool = True


def build_hierarchical_cnn(
    input_length: int,
    num_classes: int = 2,
    dropout: float = 0.3,
    input_channels: int = 1,
    channel_sizes: tuple[int, ...] = (32, 64, 128),
    kernel_sizes: tuple[int, ...] = (3, 3, 3),
    poolings: tuple[str, ...] = ("max", "max", "adaptive_max"),
    head_hidden_dims: tuple[int, ...] = (64,),
    batch_normalisation: bool = True,
) -> GeneralCNN:
    """Build the empirical CNN by supplying architecture parameters."""
    config = HierarchicalCNNConfig(
        input_length=input_length,
        input_channels=input_channels,
        channel_sizes=tuple(channel_sizes),
        kernel_sizes=tuple(kernel_sizes),
        poolings=tuple(poolings),
        head_hidden_dims=tuple(head_hidden_dims),
        num_classes=num_classes,
        dropout=dropout,
        batch_normalisation=batch_normalisation,
    )
    if config.input_length < 1:
        raise ValueError("input_length must be positive.")
    n_blocks = len(config.channel_sizes)
    if len(config.kernel_sizes) != n_blocks or len(config.poolings) != n_blocks:
        raise ValueError(
            "channel_sizes, kernel_sizes and poolings must have equal lengths."
        )

    blocks = [
        ConvBlockConfig(
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            pooling=pooling,
            pool_size=2,
            batch_normalisation=config.batch_normalisation,
        )
        for out_channels, kernel_size, pooling in zip(
            config.channel_sizes,
            config.kernel_sizes,
            config.poolings,
            strict=True,
        )
    ]
    branch = ConvPoolBranch1D(config.input_channels, blocks)
    head = ReLUClassifierHead(
        input_dim=branch.output_channels,
        hidden_dims=config.head_hidden_dims,
        output_dim=config.num_classes,
        dropout=config.dropout,
    )
    return GeneralCNN(branches=[branch], head=head)
