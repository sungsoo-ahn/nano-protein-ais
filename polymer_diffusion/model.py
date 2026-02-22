"""Polymer Diffusion — DDPM for 2D bead-spring polymer conformations.

Self-contained implementation: noise schedule, sinusoidal embeddings,
denoiser network, forward/reverse diffusion. Input: 12 beads in 2D (24D).

Total parameters: ~250K with default config.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIM = 24  # 12 beads * 2D
TIME_DIM = 64
HIDDEN_DIM = 256
T = 200  # diffusion timesteps
BETA_START = 1e-4
BETA_END = 0.02


# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------


class NoiseSchedule:
    """Precomputed DDPM linear noise schedule."""

    def __init__(self, T=T, beta_start=BETA_START, beta_end=BETA_END):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)

    def add_noise(self, x0, t, noise):
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise.

        Args:
            x0: [B, D] clean data
            t: [B] integer timesteps
            noise: [B, D] standard Gaussian noise
        Returns:
            x_t: [B, D] noised data
        """
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1).to(x0.device)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1).to(x0.device)
        return sqrt_ab * x0 + sqrt_omab * noise


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=TIME_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B] integer timesteps → [B, dim] embeddings."""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


class Denoiser(nn.Module):
    """MLP that predicts noise epsilon given (x_t, t)."""

    def __init__(self, data_dim=DATA_DIM, time_dim=TIME_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t, t):
        """
        Args:
            x_t: [B, D] noised data
            t: [B] integer timesteps
        Returns:
            eps_pred: [B, D] predicted noise
        """
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PolymerDiffusion(nn.Module):
    def __init__(
        self,
        data_dim=DATA_DIM,
        time_dim=TIME_DIM,
        hidden_dim=HIDDEN_DIM,
        num_steps=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
    ):
        super().__init__()
        self.denoiser = Denoiser(data_dim, time_dim, hidden_dim)
        self.schedule = NoiseSchedule(num_steps, beta_start, beta_end)
        self.num_steps = num_steps

    def forward(self, batch):
        """Training forward: sample random timestep, add noise, predict noise.

        Args:
            batch: dict with "coords" key, shape [B, 24]
        Returns:
            dict with "loss", "noise_pred", "noise_true"
        """
        x0 = batch["coords"]
        B = x0.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.add_noise(x0, t, noise)
        noise_pred = self.denoiser(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss, "noise_pred": noise_pred, "noise_true": noise}

    @torch.no_grad()
    def sample(self, n_samples, device=None):
        """DDPM reverse sampling. Returns [n_samples, 12, 2]."""
        if device is None:
            device = next(self.parameters()).device
        x = torch.randn(n_samples, DATA_DIM, device=device)

        for t_idx in reversed(range(self.num_steps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            eps_pred = self.denoiser(x, t)

            beta = self.schedule.betas[t_idx].to(device)
            sqrt_alpha = self.schedule.sqrt_alphas[t_idx].to(device)
            sqrt_omab = self.schedule.sqrt_one_minus_alpha_bars[t_idx].to(device)

            # mu_theta = (x_t - beta/sqrt(1-alpha_bar) * eps_pred) / sqrt(alpha)
            x = (x - (beta / sqrt_omab) * eps_pred) / sqrt_alpha

            # Add noise except at t=0
            if t_idx > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

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
    com = coords.mean(dim=1, keepdim=True)
    displacements = coords - com
    rg = (displacements**2).sum(dim=-1).mean(dim=-1).sqrt()
    return rg
