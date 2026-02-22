"""Tests for Polymer Diffusion (DDPM) model."""

import pytest
import torch

from polymer_diffusion.model import (
    Denoiser,
    NoiseSchedule,
    PolymerDiffusion,
    SinusoidalEmbedding,
    radius_of_gyration,
)


@pytest.fixture
def model():
    return PolymerDiffusion(data_dim=24, time_dim=16, hidden_dim=32, num_steps=20)


@pytest.fixture
def polymer_batch():
    torch.manual_seed(42)
    return {"coords": torch.randn(4, 24)}


def test_noise_schedule_shapes():
    ns = NoiseSchedule(T=200)
    assert ns.betas.shape == (200,)
    assert ns.alphas.shape == (200,)
    assert ns.alpha_bars.shape == (200,)
    assert ns.sqrt_alpha_bars.shape == (200,)
    assert ns.sqrt_one_minus_alpha_bars.shape == (200,)
    # alpha_bars should be decreasing
    assert (ns.alpha_bars[1:] <= ns.alpha_bars[:-1]).all()


def test_add_noise_preserves_shape():
    ns = NoiseSchedule(T=50)
    x0 = torch.randn(4, 24)
    t = torch.randint(0, 50, (4,))
    noise = torch.randn_like(x0)
    x_t = ns.add_noise(x0, t, noise)
    assert x_t.shape == (4, 24)
    assert torch.isfinite(x_t).all()


def test_sinusoidal_embedding_shape():
    emb = SinusoidalEmbedding(dim=64)
    t = torch.tensor([0, 10, 50, 199])
    out = emb(t)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_denoiser_output_shape():
    denoiser = Denoiser(data_dim=24, time_dim=16, hidden_dim=32)
    x_t = torch.randn(4, 24)
    t = torch.randint(0, 200, (4,))
    out = denoiser(x_t, t)
    assert out.shape == (4, 24)
    assert torch.isfinite(out).all()


def test_forward_returns_loss(model, polymer_batch):
    model.eval()
    with torch.no_grad():
        outputs = model(polymer_batch)
    assert "loss" in outputs
    assert "noise_pred" in outputs
    assert "noise_true" in outputs
    assert outputs["noise_pred"].shape == (4, 24)
    assert outputs["noise_true"].shape == (4, 24)


def test_loss_is_finite(model, polymer_batch):
    model.train()
    outputs = model(polymer_batch)
    assert torch.isfinite(outputs["loss"])


def test_sample_shape(model):
    model.eval()
    samples = model.sample(8)
    assert samples.shape == (8, 12, 2)
    assert torch.isfinite(samples).all()


def test_param_count():
    small = PolymerDiffusion(hidden_dim=32, time_dim=16)
    assert small.count_parameters() > 0
    full = PolymerDiffusion()  # default 256 hidden
    assert 100_000 <= full.count_parameters() <= 500_000


def test_radius_of_gyration():
    coords = torch.randn(5, 12, 2)
    rg = radius_of_gyration(coords)
    assert rg.shape == (5,)
    assert (rg >= 0).all()
    assert torch.isfinite(rg).all()
