"""Utility functions for the proportion module."""

import torch


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Computes the binary entropy of a given probability.

    Args:
        p (torch.Tensor): Probability.

    Returns:
        torch.Tensor: Binary entropy.
    """
    return -p * torch.log2(p + 1e-8) - (1 - p) * torch.log2(1 - p + 1e-8)
