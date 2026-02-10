"""Tests for AlphaFold3 model."""

import pytest
import torch

from alphafold3.model import (
    AlphaFold3,
    AttentionWithPairBias,
    DiffusionModule,
    EDMNoiseSchedule,
    PairformerBlock,
    TriangularAttention,
    TriangularMultiplicativeUpdate,
    diffusion_loss,
)


@pytest.fixture
def model():
    return AlphaFold3(
        c_s=64,
        c_z=16,
        c_atom=64,
        num_pairformer_blocks=1,
        pairformer_heads=4,
        num_diffusion_blocks=1,
        diffusion_heads=4,
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


def test_triangular_multiplicative_update():
    B, L, c_z = 2, 10, 16
    for mode in ("outgoing", "incoming"):
        tri = TriangularMultiplicativeUpdate(c_z, c_z, mode=mode)
        out = tri(torch.randn(B, L, L, c_z))
        assert out.shape == (B, L, L, c_z)
        assert torch.isfinite(out).all()


def test_triangular_attention():
    B, L, c_z = 2, 10, 16
    for mode in ("starting", "ending"):
        tri = TriangularAttention(c_z, n_heads=4, mode=mode)
        out = tri(torch.randn(B, L, L, c_z))
        assert out.shape == (B, L, L, c_z)
        assert torch.isfinite(out).all()


def test_attention_with_pair_bias_shape():
    B, L, c_s, c_z = 2, 10, 64, 16
    attn = AttentionWithPairBias(c_s, c_z, n_heads=4)
    single = torch.randn(B, L, c_s)
    pair = torch.randn(B, L, L, c_z)
    out = attn(single, pair)
    assert out.shape == (B, L, c_s)
    assert torch.isfinite(out).all()


def test_pairformer_block_shape():
    B, L, c_s, c_z = 2, 10, 64, 16
    block = PairformerBlock(c_s, c_z, n_heads=4)
    single, pair = block(torch.randn(B, L, c_s), torch.randn(B, L, L, c_z))
    assert single.shape == (B, L, c_s)
    assert pair.shape == (B, L, L, c_z)
    assert torch.isfinite(single).all()
    assert torch.isfinite(pair).all()


def test_edm_noise_schedule():
    schedule = EDMNoiseSchedule()
    sigma = torch.tensor(1.0)
    assert schedule.c_skip(sigma).item() > 0
    assert schedule.c_out(sigma).item() > 0
    assert schedule.c_in(sigma).item() > 0
    # loss weight should be positive
    assert schedule.loss_weight(sigma).item() > 0
    # sample_sigma should return positive values
    sigmas = schedule.sample_sigma(100, torch.device("cpu"))
    assert (sigmas > 0).all()
    # schedule should go from high to low
    sched = schedule.sample_schedule(10, torch.device("cpu"))
    assert sched[0] > sched[-1]
    assert sched[-1] == 0.0


def test_diffusion_module_shape():
    B, L, c_s, c_z, c_atom = 2, 10, 64, 16, 64
    module = DiffusionModule(c_s=c_s, c_z=c_z, c_atom=c_atom, n_blocks=1, n_heads=4)
    x_noisy = torch.randn(B, L, 3)
    sigma = torch.ones(B)
    single = torch.randn(B, L, c_s)
    pair = torch.randn(B, L, L, c_z)
    x_denoised = module(x_noisy, sigma, single, pair)
    assert x_denoised.shape == (B, L, 3)
    assert torch.isfinite(x_denoised).all()


def test_diffusion_loss_zero_for_perfect():
    B, L = 2, 10
    x_true = torch.randn(B, L, 3)
    sigma = torch.ones(B)
    loss = diffusion_loss(x_true, x_true, sigma)
    assert loss.item() < 1e-5


def test_diffusion_loss_positive_for_imperfect():
    B, L = 2, 10
    x_true = torch.randn(B, L, 3)
    x_pred = x_true + torch.randn(B, L, 3) * 2.0
    sigma = torch.ones(B)
    loss = diffusion_loss(x_pred, x_true, sigma)
    assert loss.item() > 0.1
    assert torch.isfinite(loss)


def test_full_model_forward(model, protein):
    model.eval()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    with torch.no_grad():
        outputs = model(batch)
    assert torch.isfinite(outputs["loss"])
    assert outputs["loss"].item() > 0


def test_model_param_count(model):
    assert 50_000 <= model.count_parameters() <= 5_000_000
    prod = AlphaFold3()
    assert 1_000_000 <= prod.count_parameters() <= 50_000_000


def test_loss_is_finite(model, protein):
    model.train()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    outputs = model(batch)
    assert torch.isfinite(outputs["loss"])
    assert not torch.isnan(outputs["loss"])
