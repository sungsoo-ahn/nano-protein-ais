"""Tests for ProteinMPNN model."""

import pytest
import torch

from proteinmpnn.model import (
    ProteinMPNN,
    build_knn_graph,
    compute_edge_features,
    compute_node_features,
    create_decoding_mask,
    rbf_encode,
)


@pytest.fixture
def model():
    return ProteinMPNN(
        hidden_dim=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=10,
        num_rbf=8,
        dropout=0.0,
    )


@pytest.fixture
def protein():
    """20-residue synthetic protein."""
    L = 20
    torch.manual_seed(42)
    coords_CA = torch.randn(L, 3)
    coords_N = coords_CA + torch.randn(L, 3) * 0.3
    coords_C = coords_CA + torch.randn(L, 3) * 0.3
    coords_O = coords_C + torch.randn(L, 3) * 0.2
    mask = torch.ones(L, dtype=torch.bool)
    mask[-2:] = False
    return {
        "coords_N": coords_N,
        "coords_CA": coords_CA,
        "coords_C": coords_C,
        "coords_O": coords_O,
        "sequence": torch.randint(0, 20, (L,)),
        "mask": mask,
    }


def test_build_knn_graph_shape(protein):
    edge_index, edge_dist = build_knn_graph(protein["coords_CA"], k=5, mask=protein["mask"])
    assert edge_index.shape[0] == 2
    assert edge_dist.shape[0] == edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]
    assert (src != dst).all()


def test_rbf_encode_range():
    d = torch.tensor([0.0, 5.0, 10.0, 20.0])
    rbf = rbf_encode(d, num_rbf=16, max_dist=20.0)
    assert rbf.shape == (4, 16)
    assert rbf.min() >= 0.0
    assert rbf.max() <= 1.0


def test_compute_node_features(protein):
    feats = compute_node_features(protein["coords_N"], protein["coords_CA"], protein["coords_C"])
    assert feats.shape == (20, 15)
    assert torch.isfinite(feats).all()


def test_compute_edge_features(protein):
    edge_index, _ = build_knn_graph(protein["coords_CA"], k=10, mask=protein["mask"])
    feats = compute_edge_features(
        protein["coords_N"],
        protein["coords_CA"],
        protein["coords_C"],
        protein["coords_O"],
        edge_index,
        num_rbf=8,
    )
    E = edge_index.shape[1]
    expected_dim = 16 * 8 + 65 + 3
    assert feats.shape == (E, expected_dim)
    assert torch.isfinite(feats).all()


def test_decoder_causal_mask():
    L = 10
    order = torch.randperm(L)
    mask = create_decoding_mask(order)
    assert mask.shape == (L, L)
    first_pos = order[0].item()
    assert mask[first_pos].sum() == 1
    last_pos = order[L - 1].item()
    assert mask[last_pos].sum() == L


def test_full_model_forward(model, protein):
    model.eval()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["logits"].shape == (1, 20, 20)
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["loss"].ndim == 0
    assert outputs["loss"].item() > 0


def test_model_param_count():
    small = ProteinMPNN(hidden_dim=32, num_encoder_layers=2, num_decoder_layers=2)
    assert 50_000 <= small.count_parameters() <= 2_000_000
    prod = ProteinMPNN()
    assert 1_000_000 <= prod.count_parameters() <= 10_000_000


def test_loss_is_finite(model, protein):
    model.train()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    outputs = model(batch)
    assert torch.isfinite(outputs["loss"]).all()


def test_design_generates_valid_aa(model, protein):
    model.eval()
    seqs = model.design(
        protein["coords_N"],
        protein["coords_CA"],
        protein["coords_C"],
        protein["coords_O"],
        protein["mask"],
        temperature=0.5,
        num_samples=3,
    )
    assert seqs.shape == (3, 20)
    assert (seqs >= 0).all()
    assert (seqs < 20).all()


def test_batch_processing(model, protein):
    model.eval()
    batch = {k: v.unsqueeze(0).repeat(2, *([1] * v.dim())) for k, v in protein.items()}
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["logits"].shape[0] == 2
    assert torch.isfinite(outputs["loss"]).all()
