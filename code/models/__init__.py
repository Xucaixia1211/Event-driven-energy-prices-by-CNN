"""Shared model components for the simulations and empirical analysis."""

from .hierarchical_classifiers import HierarchicalCNNConfig, build_hierarchical_cnn
from .general_cnn import (
    AffineHead,
    ConvBlockConfig,
    ConvPoolBranch1D,
    GeneralCNN,
    IdentityHead,
    ReLUClassifierHead,
)
from .classical_statistic_classifiers import (
    ClassicalStatisticCNNConfig,
    ar_uniform_error_bound,
    build_classical_statistic_cnn,
    rv_uniform_error_bound,
)

__all__ = [
    "AffineHead",
    "ConvBlockConfig",
    "ConvPoolBranch1D",
    "ClassicalStatisticCNNConfig",
    "GeneralCNN",
    "IdentityHead",
    "ReLUClassifierHead",
    "HierarchicalCNNConfig",
    "ar_uniform_error_bound",
    "build_classical_statistic_cnn",
    "build_hierarchical_cnn",
    "rv_uniform_error_bound",
]
