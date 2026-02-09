"""Shared test fixtures for all model tests."""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_protein():
    """A 10-residue synthetic protein with backbone coords."""
    L = 10
    torch.manual_seed(42)
    coords_CA = torch.randn(L, 3) * 3
    coords_N = coords_CA + torch.randn(L, 3) * 0.3
    coords_C = coords_CA + torch.randn(L, 3) * 0.3
    coords_O = coords_C + torch.randn(L, 3) * 0.2
    return {
        "coords_N": coords_N,
        "coords_CA": coords_CA,
        "coords_C": coords_C,
        "coords_O": coords_O,
        "sequence": torch.zeros(L, dtype=torch.long),
        "mask": torch.ones(L, dtype=torch.bool),
    }


@pytest.fixture
def sample_protein_20():
    """A 20-residue synthetic protein."""
    L = 20
    torch.manual_seed(123)
    coords_CA = torch.randn(L, 3) * 5
    coords_N = coords_CA + torch.randn(L, 3) * 0.3
    coords_C = coords_CA + torch.randn(L, 3) * 0.3
    coords_O = coords_C + torch.randn(L, 3) * 0.2
    return {
        "coords_N": coords_N,
        "coords_CA": coords_CA,
        "coords_C": coords_C,
        "coords_O": coords_O,
        "sequence": torch.randint(0, 20, (L,)),
        "mask": torch.ones(L, dtype=torch.bool),
    }
