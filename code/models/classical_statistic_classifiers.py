"""Fixed CNN classifiers based on the classical statistics in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .general_cnn import GeneralCNN, IdentityHead


StatisticName = Literal[
    "range",
    "drawup",
    "drawdown",
    "slope",
    "realised_volatility",
    "ar",
]


@dataclass(frozen=True)
class ClassicalStatisticCNNConfig:
    """Parameters selecting one fixed classical-statistic CNN."""

    statistic: StatisticName
    input_length: int
    bound: float = 1.0
    approximation_level: int = 9


def _freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _slope_contrast_coefficients(input_length: int, split: int) -> torch.Tensor:
    time = torch.arange(1, input_length + 1, dtype=torch.float64)
    left_time = time[:split]
    right_time = time[split:]
    left_centred = left_time - left_time.mean()
    right_centred = right_time - right_time.mean()
    coefficients = torch.zeros(input_length, dtype=torch.float64)
    coefficients[:split] = left_centred / left_centred.square().sum()
    coefficients[split:] = -right_centred / right_centred.square().sum()
    return coefficients


class _RangeBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[1.0]], [[-1.0]]]))
        _freeze(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = F.relu(self.conv(x.unsqueeze(1)))
        positive_score = response[:, 0, :] - response[:, 1, :]
        negative_score = response[:, 1, :] - response[:, 0, :]
        statistic = positive_score.amax(dim=1) + negative_score.amax(dim=1)
        return statistic.unsqueeze(1)


class _DrawBranch(nn.Module):
    def __init__(self, input_length: int, direction: Literal["up", "down"]) -> None:
        super().__init__()
        sign = 1.0 if direction == "up" else -1.0
        filters = []
        for lag in range(1, input_length):
            conv = nn.Conv1d(1, 1, kernel_size=lag + 1, bias=False)
            weight = torch.zeros(1, 1, lag + 1)
            weight[0, 0, 0] = -sign
            weight[0, 0, lag] = sign
            with torch.no_grad():
                conv.weight.copy_(weight)
            filters.append(_freeze(conv))
        self.filters = nn.ModuleList(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x.unsqueeze(1)
        branch_maxima = [
            F.relu(conv(series)).amax(dim=2).squeeze(1) for conv in self.filters
        ]
        statistic = torch.stack(branch_maxima, dim=1).amax(dim=1)
        return statistic.unsqueeze(1)


class _SlopeChangeBranch(nn.Module):
    def __init__(self, input_length: int) -> None:
        super().__init__()
        coefficients = [
            _slope_contrast_coefficients(input_length, split)
            for split in range(3, input_length - 1)
        ]
        filters = torch.stack(
            coefficients + [-coefficient for coefficient in coefficients]
        )
        self.conv = nn.Conv1d(
            1,
            filters.shape[0],
            kernel_size=input_length,
            bias=False,
            dtype=torch.float64,
        )
        with torch.no_grad():
            self.conv.weight.copy_(filters.unsqueeze(1))
        _freeze(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = F.relu(self.conv(x.to(torch.float64).unsqueeze(1)))
        statistic = response.squeeze(-1).amax(dim=1).to(x.dtype)
        return statistic.unsqueeze(1)


def _tent_map(values: torch.Tensor) -> torch.Tensor:
    return (
        2.0 * F.relu(values)
        - 4.0 * F.relu(values - 0.5)
        + 2.0 * F.relu(values - 1.0)
    )


def _square_relu_approximation(
    values: torch.Tensor,
    bound: float,
    approximation_level: int,
) -> torch.Tensor:
    if bound <= 0:
        raise ValueError("bound must be positive.")
    if approximation_level < 0:
        raise ValueError("approximation_level must be nonnegative.")
    scaled = torch.abs(values) / bound
    tent_value = scaled
    approximation = scaled.clone()
    for level in range(1, approximation_level + 1):
        tent_value = _tent_map(tent_value)
        approximation = approximation - tent_value / (4.0**level)
    return (bound**2) * approximation


class _RealisedVolatilityBranch(nn.Module):
    def __init__(self, bound: float, approximation_level: int) -> None:
        super().__init__()
        self.bound = float(bound)
        self.approximation_level = int(approximation_level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        increments = x[:, 1:] - x[:, :-1]
        statistic = _square_relu_approximation(
            increments,
            2.0 * self.bound,
            self.approximation_level,
        ).sum(dim=1)
        return statistic.unsqueeze(1)


class _ARBranch(nn.Module):
    def __init__(self, bound: float, approximation_level: int) -> None:
        super().__init__()
        self.bound = float(bound)
        self.approximation_level = int(approximation_level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = x[:, :-1]
        right = x[:, 1:]
        product = 0.25 * (
            _square_relu_approximation(
                left + right, 2.0 * self.bound, self.approximation_level
            )
            - _square_relu_approximation(
                left - right, 2.0 * self.bound, self.approximation_level
            )
        )
        left_square = _square_relu_approximation(
            left, self.bound, self.approximation_level
        )
        statistic = (product - left_square).sum(dim=1)
        return statistic.unsqueeze(1)


def build_classical_statistic_cnn(
    statistic: StatisticName,
    input_length: int,
    bound: float = 1.0,
    approximation_level: int = 9,
) -> GeneralCNN:
    """Build one fixed classical-statistic CNN from its defining parameters."""
    config = ClassicalStatisticCNNConfig(
        statistic=statistic,
        input_length=input_length,
        bound=bound,
        approximation_level=approximation_level,
    )
    if config.input_length < 1:
        raise ValueError("input_length must be positive.")

    if config.statistic == "range":
        branch: nn.Module = _RangeBranch()
    elif config.statistic in {"drawup", "drawdown"}:
        if config.input_length < 2:
            raise ValueError("Drawup and drawdown require input_length >= 2.")
        direction = "up" if config.statistic == "drawup" else "down"
        branch = _DrawBranch(config.input_length, direction)
    elif config.statistic == "slope":
        if config.input_length < 5:
            raise ValueError("Slope change requires input_length >= 5.")
        branch = _SlopeChangeBranch(config.input_length)
    elif config.statistic == "realised_volatility":
        if config.input_length < 2:
            raise ValueError("Realised volatility requires input_length >= 2.")
        branch = _RealisedVolatilityBranch(
            config.bound, config.approximation_level
        )
    elif config.statistic == "ar":
        if config.input_length < 2:
            raise ValueError("The AR statistic requires input_length >= 2.")
        branch = _ARBranch(config.bound, config.approximation_level)
    else:
        raise ValueError(f"Unsupported statistic: {config.statistic!r}")

    return GeneralCNN(
        branches=[branch],
        head=IdentityHead(squeeze_scalar=True),
    )


def rv_uniform_error_bound(
    input_length: int,
    bound: float,
    approximation_level: int,
) -> float:
    return (
        (input_length - 1)
        * (2.0 * bound) ** 2
        * (4.0 ** (-(approximation_level + 1)))
    )


def ar_uniform_error_bound(
    input_length: int,
    bound: float,
    approximation_level: int,
) -> float:
    return (
        (input_length - 1)
        * 3.0
        * (bound**2)
        * (4.0 ** (-(approximation_level + 1)))
    )
