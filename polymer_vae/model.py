"""Polymer VAE — variational autoencoder for 2D bead-spring polymer conformations.

Self-contained implementation: Encoder, Decoder, beta-VAE loss, sampling.
Input: 12 beads in 2D (flattened to 24D). Latent space: 2D.

Total parameters: ~250K with default config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM = 24  # 12 beads * 2D
HIDDEN_DIM = 256
LATENT_DIM = 2
BETA = 0.01  # KL weight for beta-VAE


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        sigma = F.softplus(self.fc_sigma(h))  # ensure positive
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def vae_loss(x, x_recon, mu, sigma, beta=BETA):
    """Beta-VAE loss = MSE reconstruction + beta * KL divergence."""
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    # KL(N(mu, sigma^2) || N(0, 1))
    kl_loss = -0.5 * torch.mean(1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2))
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PolymerVAE(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        beta=BETA,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.beta = beta
        self.latent_dim = latent_dim

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, batch):
        x = batch["coords"]  # [B, 24]
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_recon = self.decoder(z)
        loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, sigma, self.beta)
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "x_recon": x_recon,
            "mu": mu,
            "sigma": sigma,
        }

    @torch.no_grad()
    def sample(self, n_samples, device=None):
        """Sample from the prior and decode. Returns [n_samples, 12, 2]."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        x = self.decoder(z)
        return x.view(n_samples, 12, 2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation utility
# ---------------------------------------------------------------------------


def radius_of_gyration(coords):
    """Compute radius of gyration for each conformation.

    Args:
        coords: [batch, 12, 2] bead positions
    Returns:
        [batch] Rg values
    """
    com = coords.mean(dim=1, keepdim=True)  # [batch, 1, 2]
    displacements = coords - com
    rg = (displacements**2).sum(dim=-1).mean(dim=-1).sqrt()
    return rg
