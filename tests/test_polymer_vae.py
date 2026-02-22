"""Tests for Polymer VAE model."""

import pytest
import torch

from polymer_vae.model import Decoder, Encoder, PolymerVAE, radius_of_gyration, vae_loss


@pytest.fixture
def model():
    return PolymerVAE(input_dim=24, hidden_dim=32, latent_dim=2, beta=0.01)


@pytest.fixture
def polymer_batch():
    torch.manual_seed(42)
    return {"coords": torch.randn(4, 24)}


def test_encoder_output_shapes():
    enc = Encoder(input_dim=24, hidden_dim=32, latent_dim=2)
    x = torch.randn(4, 24)
    mu, sigma = enc(x)
    assert mu.shape == (4, 2)
    assert sigma.shape == (4, 2)
    assert (sigma > 0).all(), "sigma must be positive (softplus)"


def test_decoder_output_shape():
    dec = Decoder(latent_dim=2, hidden_dim=32, output_dim=24)
    z = torch.randn(4, 2)
    out = dec(z)
    assert out.shape == (4, 24)


def test_forward_returns_loss(model, polymer_batch):
    model.eval()
    with torch.no_grad():
        outputs = model(polymer_batch)
    assert "loss" in outputs
    assert "recon_loss" in outputs
    assert "kl_loss" in outputs
    assert "x_recon" in outputs
    assert "mu" in outputs
    assert "sigma" in outputs
    assert outputs["x_recon"].shape == (4, 24)


def test_loss_is_finite(model, polymer_batch):
    model.train()
    outputs = model(polymer_batch)
    assert torch.isfinite(outputs["loss"])
    assert torch.isfinite(outputs["recon_loss"])
    assert torch.isfinite(outputs["kl_loss"])


def test_vae_loss_components():
    x = torch.randn(8, 24)
    x_recon = x + 0.1 * torch.randn_like(x)
    mu = torch.randn(8, 2)
    sigma = torch.ones(8, 2)
    total, recon, kl = vae_loss(x, x_recon, mu, sigma)
    assert recon.item() >= 0
    assert kl.item() >= 0
    assert torch.isfinite(total)


def test_sample_shape(model):
    model.eval()
    samples = model.sample(8)
    assert samples.shape == (8, 12, 2)
    assert torch.isfinite(samples).all()


def test_reparameterize_stochastic(model):
    mu = torch.zeros(10, 2)
    sigma = torch.ones(10, 2)
    z1 = model.reparameterize(mu, sigma)
    z2 = model.reparameterize(mu, sigma)
    # Two calls should give different results (stochastic)
    assert not torch.allclose(z1, z2)


def test_param_count():
    small = PolymerVAE(hidden_dim=32, latent_dim=2)
    assert small.count_parameters() > 0
    full = PolymerVAE()  # default 256 hidden
    assert 100_000 <= full.count_parameters() <= 500_000


def test_radius_of_gyration():
    coords = torch.randn(5, 12, 2)
    rg = radius_of_gyration(coords)
    assert rg.shape == (5,)
    assert (rg >= 0).all()
    assert torch.isfinite(rg).all()
