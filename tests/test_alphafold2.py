"""Tests for AlphaFold2 model."""

import pytest
import torch

from alphafold2.model import (
    AlphaFold2,
    EvoformerBlock,
    InvariantPointAttention,
    MSARowAttentionWithPairBias,
    OuterProductMean,
    RigidTransform,
    TriangularAttention,
    TriangularMultiplicativeUpdate,
    fape_loss,
)


@pytest.fixture
def model():
    return AlphaFold2(
        c_m=32,
        c_z=16,
        c_s=32,
        num_evoformer_blocks=1,
        evoformer_heads=4,
        num_structure_layers=1,
        ipa_heads=4,
        n_qk_points=2,
        n_v_points=2,
        num_dist_bins=16,
        num_plddt_bins=10,
    )


@pytest.fixture
def protein():
    L = 15
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
        "sequence": torch.randint(0, 20, (L,)),
        "mask": torch.ones(L, dtype=torch.bool),
    }


def test_msa_row_attention_shape():
    L, c_m, c_z = 10, 32, 16
    attn = MSARowAttentionWithPairBias(c_m, c_z, n_heads=4)
    msa = torch.randn(1, L, c_m)
    pair = torch.randn(L, L, c_z)
    out = attn(msa, pair)
    assert out.shape == (1, L, c_m)
    assert torch.isfinite(out).all()


def test_outer_product_mean_shape():
    L, c_m, c_z = 10, 32, 16
    opm = OuterProductMean(c_m, c_z, c_hidden=8)
    out = opm(torch.randn(1, L, c_m))
    assert out.shape == (L, L, c_z)
    assert torch.isfinite(out).all()


def test_triangular_multiplicative_update():
    L, c_z = 10, 16
    for mode in ("outgoing", "incoming"):
        tri = TriangularMultiplicativeUpdate(c_z, c_z, mode=mode)
        out = tri(torch.randn(L, L, c_z))
        assert out.shape == (L, L, c_z)
        assert torch.isfinite(out).all()


def test_triangular_attention():
    L, c_z = 10, 16
    for mode in ("starting", "ending"):
        tri = TriangularAttention(c_z, n_heads=4, mode=mode)
        out = tri(torch.randn(L, L, c_z))
        assert out.shape == (L, L, c_z)
        assert torch.isfinite(out).all()


def test_evoformer_block_shape():
    L, c_m, c_z = 10, 32, 16
    block = EvoformerBlock(c_m, c_z, n_heads=4)
    msa, pair = block(torch.randn(1, L, c_m), torch.randn(L, L, c_z))
    assert msa.shape == (1, L, c_m)
    assert pair.shape == (L, L, c_z)
    assert torch.isfinite(msa).all()
    assert torch.isfinite(pair).all()


def test_ipa_shape():
    L, c_s, c_z = 10, 32, 16
    ipa = InvariantPointAttention(c_s, c_z, n_heads=4, n_qk_points=2, n_v_points=2)
    out = ipa(torch.randn(L, c_s), torch.randn(L, L, c_z), RigidTransform.identity((L,)))
    assert out.shape == (L, c_s)
    assert torch.isfinite(out).all()


def test_fape_zero_for_perfect():
    L = 10
    frames = RigidTransform.identity((L,))
    coords = torch.randn(L, 3)
    loss = fape_loss(frames, coords, frames, coords)
    assert loss.item() < 1e-3


def test_fape_positive_for_imperfect():
    L = 10
    frames = RigidTransform.identity((L,))
    true_coords = torch.randn(L, 3)
    pred_coords = true_coords + torch.randn(L, 3) * 2.0
    loss = fape_loss(frames, pred_coords, frames, true_coords)
    assert loss.item() > 0.1
    assert torch.isfinite(loss)


def test_full_model_forward(model, protein):
    model.eval()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    with torch.no_grad():
        outputs = model(batch)
    L = protein["coords_CA"].shape[0]
    assert outputs["coords"].shape == (1, L, 3)
    assert torch.isfinite(outputs["loss"])
    assert outputs["loss"].item() > 0


def test_model_param_count(model):
    assert 50_000 <= model.count_parameters() <= 5_000_000
    prod = AlphaFold2()
    assert 1_000_000 <= prod.count_parameters() <= 50_000_000


def test_loss_is_finite(model, protein):
    model.train()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    outputs = model(batch)
    assert torch.isfinite(outputs["loss"])
    assert not torch.isnan(outputs["loss"])
