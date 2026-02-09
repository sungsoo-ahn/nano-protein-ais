"""AlphaFold3 — structure prediction: amino acid sequence → 3D backbone coordinates.

Self-contained implementation with Pairformer (single+pair representation) and
diffusion-based structure module using EDM/Karras preconditioning.

Core ideas: Pairformer replaces Evoformer (drops MSA axis entirely), coordinate
diffusion replaces IPA-based frame refinement.

Total parameters: ~30M with default config.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_S = 256  # Single representation dim
C_Z = 64  # Pair representation dim
C_ATOM = 256  # Diffusion transformer dim
NUM_PAIRFORMER_BLOCKS = 4
PAIRFORMER_HEADS = 4
NUM_DIFFUSION_BLOCKS = 4
DIFFUSION_HEADS = 8
NUM_DIST_BINS = 64
NUM_PLDDT_BINS = 50

# EDM / Karras preconditioning
SIGMA_DATA = 16.0  # Data std (Angstroms)
P_MEAN = -1.2  # Log-normal mean for sigma sampling
P_STD = 1.2  # Log-normal std
SIGMA_MIN = 0.0004
SIGMA_MAX = 160.0
NUM_SAMPLE_STEPS = 50

# Loss weights
DIFFUSION_WEIGHT = 4.0
DISTOGRAM_WEIGHT = 0.03
PLDDT_WEIGHT = 0.01

# ---------------------------------------------------------------------------
# Geometry (inlined)
# ---------------------------------------------------------------------------


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: axis-angle [...,3] -> rotation matrix [...,3,3]."""
    angle = axis_angle.norm(dim=-1, keepdim=True).unsqueeze(-1)
    axis = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)
    K = skew_symmetric(axis)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    return eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


class RigidTransform:
    """Rigid body transformation T = (R, t) in SE(3)."""

    def __init__(self, rots: torch.Tensor, trans: torch.Tensor):
        self.rots = rots  # [..., 3, 3]
        self.trans = trans  # [..., 3]

    @classmethod
    def identity(cls, batch_shape: tuple, device="cpu", dtype=torch.float32):
        rots = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).clone()
        trans = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
        return cls(rots, trans)

    def compose(self, other):
        new_rots = self.rots @ other.rots
        new_trans = (self.rots @ other.trans.unsqueeze(-1)).squeeze(-1) + self.trans
        return RigidTransform(new_rots, new_trans)

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return (self.rots @ points.unsqueeze(-1)).squeeze(-1) + self.trans

    def invert(self):
        inv_rots = self.rots.transpose(-1, -2)
        inv_trans = -(inv_rots @ self.trans.unsqueeze(-1)).squeeze(-1)
        return RigidTransform(inv_rots, inv_trans)

    def detach(self):
        return RigidTransform(self.rots.detach(), self.trans.detach())


def compute_local_frame(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor) -> RigidTransform:
    x = C - CA
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    v = N - CA
    z = torch.cross(x, v, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)
    return RigidTransform(R, CA)


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)
    return diff.norm(dim=-1)


# ---------------------------------------------------------------------------
# Pairformer (replaces Evoformer — no MSA axis)
# ---------------------------------------------------------------------------


class TriangularMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z: int, c_hidden: int, mode: str = "outgoing"):
        super().__init__()
        self.mode = mode
        self.ln = nn.LayerNorm(c_z)
        self.left_proj = nn.Linear(c_z, c_hidden)
        self.right_proj = nn.Linear(c_z, c_hidden)
        self.left_gate = nn.Linear(c_z, c_hidden)
        self.right_gate = nn.Linear(c_z, c_hidden)
        self.output_gate = nn.Linear(c_z, c_z)
        self.output_proj = nn.Linear(c_hidden, c_z)
        self.final_norm = nn.LayerNorm(c_hidden)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        z = self.ln(pair)
        left = self.left_proj(z) * torch.sigmoid(self.left_gate(z))
        right = self.right_proj(z) * torch.sigmoid(self.right_gate(z))
        if self.mode == "outgoing":
            out = torch.einsum("ikc,jkc->ijc", left, right)
        else:
            out = torch.einsum("kic,kjc->ijc", left, right)
        out = self.output_proj(self.final_norm(out))
        return pair + torch.sigmoid(self.output_gate(pair)) * out


class TriangularAttention(nn.Module):
    def __init__(self, c_z: int, n_heads: int = 4, mode: str = "starting"):
        super().__init__()
        self.c_z, self.n_heads, self.mode = c_z, n_heads, mode
        self.head_dim = c_z // n_heads
        self.ln = nn.LayerNorm(c_z)
        self.to_q = nn.Linear(c_z, c_z, bias=False)
        self.to_k = nn.Linear(c_z, c_z, bias=False)
        self.to_v = nn.Linear(c_z, c_z, bias=False)
        self.bias_proj = nn.Linear(c_z, n_heads, bias=False)
        self.to_out = nn.Linear(c_z, c_z)
        self.gate = nn.Linear(c_z, c_z)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        L = pair.shape[0]
        if self.mode == "ending":
            pair = pair.transpose(0, 1).contiguous()
        z = self.ln(pair)
        q = self.to_q(z).view(L, L, self.n_heads, self.head_dim)
        k = self.to_k(z).view(L, L, self.n_heads, self.head_dim)
        v = self.to_v(z).view(L, L, self.n_heads, self.head_dim)
        attn = torch.einsum("ijhd,ikhd->hijk", q, k) / (self.head_dim**0.5)
        attn = attn + self.bias_proj(z).permute(2, 0, 1).unsqueeze(1)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hijk,ikhd->ijhd", attn, v).reshape(L, L, self.c_z)
        result = pair + torch.sigmoid(self.gate(pair)) * self.to_out(out)
        if self.mode == "ending":
            result = result.transpose(0, 1).contiguous()
        return result


class Transition(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion)
        self.linear2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(F.relu(self.linear1(self.ln(x))))


class AttentionWithPairBias(nn.Module):
    """Self-attention on single rep [L, C_S] with pair bias from [L, L, C_Z]."""

    def __init__(self, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        self.c_s, self.n_heads = c_s, n_heads
        self.head_dim = c_s // n_heads
        self.ln_s = nn.LayerNorm(c_s)
        self.ln_z = nn.LayerNorm(c_z)
        self.to_q = nn.Linear(c_s, c_s, bias=False)
        self.to_k = nn.Linear(c_s, c_s, bias=False)
        self.to_v = nn.Linear(c_s, c_s, bias=False)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.to_out = nn.Linear(c_s, c_s)
        self.gate = nn.Linear(c_s, c_s)

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        L = single.shape[0]
        s = self.ln_s(single)
        z = self.ln_z(pair)
        q = self.to_q(s).view(L, self.n_heads, self.head_dim)
        k = self.to_k(s).view(L, self.n_heads, self.head_dim)
        v = self.to_v(s).view(L, self.n_heads, self.head_dim)
        attn = torch.einsum("ihd,jhd->hij", q, k) / (self.head_dim**0.5)
        attn = attn + self.pair_bias(z).permute(2, 0, 1)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hij,jhd->ihd", attn, v).reshape(L, self.c_s)
        return single + torch.sigmoid(self.gate(s)) * self.to_out(out)


class PairformerBlock(nn.Module):
    """Pairformer block: (single, pair) → (single, pair).

    Pair stack: tri_out → tri_in → tri_attn_start → tri_attn_end → pair_transition
    Single stack: attention_with_pair_bias → single_transition
    """

    def __init__(self, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        # Pair stack
        self.tri_out = TriangularMultiplicativeUpdate(c_z, c_z, "outgoing")
        self.tri_in = TriangularMultiplicativeUpdate(c_z, c_z, "incoming")
        self.tri_attn_start = TriangularAttention(c_z, n_heads, "starting")
        self.tri_attn_end = TriangularAttention(c_z, n_heads, "ending")
        self.pair_transition = Transition(c_z)
        # Single stack
        self.attn = AttentionWithPairBias(c_s, c_z, n_heads)
        self.single_transition = Transition(c_s)

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.tri_out(pair)
        pair = self.tri_in(pair)
        pair = self.tri_attn_start(pair)
        pair = self.tri_attn_end(pair)
        pair = self.pair_transition(pair)
        single = self.attn(single, pair)
        single = self.single_transition(single)
        return single, pair


# ---------------------------------------------------------------------------
# EDM Noise Schedule (Karras et al.)
# ---------------------------------------------------------------------------


class EDMNoiseSchedule:
    """Karras preconditioning for diffusion on R^3 coordinates."""

    def __init__(
        self,
        sigma_data: float = SIGMA_DATA,
        p_mean: float = P_MEAN,
        p_std: float = P_STD,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
    ):
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (sigma**2 + self.sigma_data**2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma + 1e-8)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def sample_sigma(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample training noise levels from log-normal."""
        log_sigma = self.p_mean + self.p_std * torch.randn(n, device=device)
        return log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    def sample_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Karras sigma schedule for inference (rho=7)."""
        rho = 7.0
        inv_rho = 1.0 / rho
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        sigmas = (
            self.sigma_max**inv_rho + steps * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** rho
        sigmas[-1] = 0.0
        return sigmas


# ---------------------------------------------------------------------------
# Diffusion Module (replaces Structure Module)
# ---------------------------------------------------------------------------


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal embedding of noise level."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=sigma.device) / half)
        args = sigma.unsqueeze(-1) * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)


class AdaLN(nn.Module):
    """Adaptive LayerNorm: LN(x) * (1 + scale) + shift."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)
        if cond.dim() == 1:
            scale_shift = scale_shift.unsqueeze(0).expand(x.shape[0], -1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.ln(x) * (1 + scale) + shift


class DiffusionTransformerBlock(nn.Module):
    """Self-attention + pair bias + AdaLN time conditioning + FFN."""

    def __init__(self, c_atom: int, c_z: int, n_heads: int):
        super().__init__()
        self.c_atom, self.n_heads = c_atom, n_heads
        self.head_dim = c_atom // n_heads
        self.adaln1 = AdaLN(c_atom, c_atom)
        self.to_q = nn.Linear(c_atom, c_atom, bias=False)
        self.to_k = nn.Linear(c_atom, c_atom, bias=False)
        self.to_v = nn.Linear(c_atom, c_atom, bias=False)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.to_out = nn.Linear(c_atom, c_atom)
        self.adaln2 = AdaLN(c_atom, c_atom)
        self.ffn = nn.Sequential(
            nn.Linear(c_atom, c_atom * 4),
            nn.GELU(),
            nn.Linear(c_atom * 4, c_atom),
        )

    def forward(self, x: torch.Tensor, pair: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        L = x.shape[0]
        # Self-attention with pair bias and AdaLN
        h = self.adaln1(x, time_cond)
        q = self.to_q(h).view(L, self.n_heads, self.head_dim)
        k = self.to_k(h).view(L, self.n_heads, self.head_dim)
        v = self.to_v(h).view(L, self.n_heads, self.head_dim)
        attn = torch.einsum("ihd,jhd->hij", q, k) / (self.head_dim**0.5)
        attn = attn + self.pair_bias(pair).permute(2, 0, 1)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hij,jhd->ihd", attn, v).reshape(L, self.c_atom)
        x = x + self.to_out(out)
        # FFN with AdaLN
        x = x + self.ffn(self.adaln2(x, time_cond))
        return x


class DiffusionModule(nn.Module):
    """Denoising network for R^3 coordinate diffusion.

    Applies EDM preconditioning:
        x_denoised = c_skip(σ) * x_noisy + c_out(σ) * F(c_in(σ) * x_noisy, ...)
    """

    def __init__(
        self,
        c_s: int = C_S,
        c_z: int = C_Z,
        c_atom: int = C_ATOM,
        n_blocks: int = NUM_DIFFUSION_BLOCKS,
        n_heads: int = DIFFUSION_HEADS,
        sigma_data: float = SIGMA_DATA,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.schedule = EDMNoiseSchedule(sigma_data=sigma_data)

        self.coord_proj = nn.Linear(3, c_atom)
        self.single_cond_proj = nn.Linear(c_s, c_atom)
        self.time_embed = FourierTimeEmbedding(c_atom)
        self.time_mlp = nn.Sequential(
            nn.Linear(c_atom, c_atom * 4),
            nn.GELU(),
            nn.Linear(c_atom * 4, c_atom),
        )
        self.blocks = nn.ModuleList(
            [DiffusionTransformerBlock(c_atom, c_z, n_heads) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(c_atom, 3)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        single: torch.Tensor,
        pair: torch.Tensor,
    ) -> torch.Tensor:
        """Predict denoised coordinates with EDM preconditioning.

        Args:
            x_noisy: [L, 3] noisy coordinates
            sigma: [] scalar noise level
            single: [L, C_S] single representation from Pairformer
            pair: [L, L, C_Z] pair representation from Pairformer
        Returns:
            x_denoised: [L, 3] denoised coordinates
        """
        c_skip = self.schedule.c_skip(sigma)
        c_out = self.schedule.c_out(sigma)
        c_in = self.schedule.c_in(sigma)

        # Network input
        x_scaled = c_in * x_noisy
        h = self.coord_proj(x_scaled) + self.single_cond_proj(single)

        # Time conditioning
        time_emb = self.time_embed(self.schedule.c_noise(sigma))
        time_cond = self.time_mlp(time_emb)

        for block in self.blocks:
            h = block(h, pair, time_cond)

        F_out = self.output_proj(h)

        # EDM preconditioning
        x_denoised = c_skip * x_noisy + c_out * F_out
        return x_denoised


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class DistogramHead(nn.Module):
    def __init__(self, c_z: int, num_bins: int = NUM_DIST_BINS):
        super().__init__()
        self.linear = nn.Linear(c_z, num_bins)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.linear(pair)


class PLDDTHead(nn.Module):
    def __init__(self, c_s: int, num_bins: int = NUM_PLDDT_BINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, num_bins),
        )

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        return self.net(single)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def diffusion_loss(
    x_denoised: torch.Tensor,
    x_true: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor | None = None,
    sigma_data: float = SIGMA_DATA,
) -> torch.Tensor:
    """Weighted MSE: λ(σ) * ||x_denoised - x_0||²."""
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    sq_error = ((x_denoised - x_true) ** 2).sum(-1)  # [L]
    if mask is not None:
        return weight * (sq_error * mask.float()).sum() / mask.sum().clamp(min=1)
    return weight * sq_error.mean()


def distogram_loss(
    pred_logits: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_bins: int = NUM_DIST_BINS,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
) -> torch.Tensor:
    diff = true_coords.unsqueeze(0) - true_coords.unsqueeze(1)
    true_dist = torch.sqrt((diff**2).sum(-1) + 1e-8)
    bin_edges = torch.linspace(min_dist, max_dist, num_bins + 1, device=true_dist.device)
    true_bins = torch.bucketize(true_dist, bin_edges[1:]).clamp(0, num_bins - 1)
    L = pred_logits.shape[0]
    flat_loss = F.cross_entropy(
        pred_logits.reshape(-1, num_bins),
        true_bins.reshape(-1),
        reduction="none",
    )
    loss = flat_loss.reshape(L, L)
    if mask is not None:
        mask_2d = mask.unsqueeze(0) & mask.unsqueeze(1)
        return (loss * mask_2d.float()).sum() / mask_2d.sum().clamp(min=1)
    return loss.mean()


def plddt_loss(
    pred_logits: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_bins: int = NUM_PLDDT_BINS,
) -> torch.Tensor:
    diff = (pred_coords - true_coords).norm(dim=-1)
    lddt = torch.clamp(1.0 - diff / 15.0, 0.0, 1.0)
    true_bins = (lddt * (num_bins - 1)).long().clamp(0, num_bins - 1)
    loss = F.cross_entropy(pred_logits, true_bins, reduction="none")
    if mask is not None:
        return (loss * mask.float()).sum() / mask.sum().clamp(min=1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Input Embedding
# ---------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    """Embed sequence → (single [L, C_S], pair [L, L, C_Z]). No MSA axis."""

    def __init__(self, c_s: int, c_z: int):
        super().__init__()
        self.single_proj = nn.Linear(21, c_s)
        self.left_single = nn.Linear(21, c_z)
        self.right_single = nn.Linear(21, c_z)
        self.relpos = nn.Embedding(65, c_z)

    def forward(
        self,
        sequence: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = sequence.shape[0]
        one_hot = torch.zeros(L, 21, device=sequence.device)
        one_hot.scatter_(1, sequence.clamp(0, 20).unsqueeze(1), 1.0)
        single = self.single_proj(one_hot)
        left, right = self.left_single(one_hot), self.right_single(one_hot)
        pair = left[:, None, :] + right[None, :, :]
        d = torch.clamp(residue_index[:, None] - residue_index[None, :] + 32, 0, 64).long()
        return single, pair + self.relpos(d)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AlphaFold3(nn.Module):
    """AlphaFold3 protein structure prediction model.

    Single-sequence mode. Predicts backbone coordinates from sequence using
    Pairformer trunk + coordinate diffusion.
    Default: 4 pairformer blocks, 4 diffusion blocks = ~30M params.
    """

    def __init__(
        self,
        c_s: int = C_S,
        c_z: int = C_Z,
        c_atom: int = C_ATOM,
        num_pairformer_blocks: int = NUM_PAIRFORMER_BLOCKS,
        pairformer_heads: int = PAIRFORMER_HEADS,
        num_diffusion_blocks: int = NUM_DIFFUSION_BLOCKS,
        diffusion_heads: int = DIFFUSION_HEADS,
        num_dist_bins: int = NUM_DIST_BINS,
        num_plddt_bins: int = NUM_PLDDT_BINS,
        sigma_data: float = SIGMA_DATA,
        diffusion_weight: float = DIFFUSION_WEIGHT,
        distogram_weight: float = DISTOGRAM_WEIGHT,
        plddt_weight: float = PLDDT_WEIGHT,
    ):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.distogram_weight = distogram_weight
        self.plddt_weight = plddt_weight
        self.num_dist_bins = num_dist_bins
        self.num_plddt_bins = num_plddt_bins
        self.sigma_data = sigma_data

        self.input_embedding = InputEmbedding(c_s, c_z)
        self.pairformer_blocks = nn.ModuleList(
            [PairformerBlock(c_s, c_z, pairformer_heads) for _ in range(num_pairformer_blocks)]
        )
        self.diffusion_module = DiffusionModule(
            c_s, c_z, c_atom, num_diffusion_blocks, diffusion_heads, sigma_data
        )
        self.distogram_head = DistogramHead(c_z, num_dist_bins)
        self.plddt_head = PLDDTHead(c_s, num_plddt_bins)
        self.noise_schedule = EDMNoiseSchedule(sigma_data=sigma_data)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            batch: dict with coords_N/CA/C [B,L,3], sequence [B,L], mask [B,L].
        Returns:
            dict with loss (scalar), diffusion_loss (scalar).
        """
        B = batch["sequence"].shape[0]
        all_losses, all_diff_losses = [], []

        for b in range(B):
            seq, mask = batch["sequence"][b], batch["mask"][b]
            true_CA = batch["coords_CA"][b]
            L = seq.shape[0]
            device = seq.device

            # Pairformer trunk
            single, pair = self.input_embedding(seq, torch.arange(L, device=device))
            for block in self.pairformer_blocks:
                single, pair = block(single, pair)

            # Sample sigma and noise coordinates
            sigma = self.noise_schedule.sample_sigma(1, device).squeeze(0)
            eps = torch.randn_like(true_CA)
            x_noisy = true_CA + sigma * eps

            # Diffusion module predicts denoised coords
            x_denoised = self.diffusion_module(x_noisy, sigma, single, pair)

            # Losses
            loss_diff = diffusion_loss(x_denoised, true_CA, sigma, mask, self.sigma_data)

            dist_logits = self.distogram_head(pair)
            loss_dist = distogram_loss(dist_logits, true_CA, mask, self.num_dist_bins)

            plddt_logits = self.plddt_head(single)
            loss_plddt = plddt_loss(
                plddt_logits, x_denoised.detach(), true_CA, mask, self.num_plddt_bins
            )

            total = (
                self.diffusion_weight * loss_diff
                + self.distogram_weight * loss_dist
                + self.plddt_weight * loss_plddt
            )
            all_losses.append(total)
            all_diff_losses.append(loss_diff)

        return {
            "loss": torch.stack(all_losses).mean(),
            "diffusion_loss": torch.stack(all_diff_losses).mean(),
        }

    @torch.no_grad()
    def predict(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict structure from sequence via iterative denoising.

        Returns coords [L,3] and plddt [L].
        """
        self.eval()
        L = sequence.shape[0]
        device = sequence.device

        # Pairformer trunk
        single, pair = self.input_embedding(sequence, torch.arange(L, device=device))
        for block in self.pairformer_blocks:
            single, pair = block(single, pair)

        # Iterative denoising (Euler ODE sampler)
        schedule = self.noise_schedule
        sigmas = schedule.sample_schedule(NUM_SAMPLE_STEPS, device)
        x = sigmas[0] * torch.randn(L, 3, device=device)

        for i in range(len(sigmas) - 1):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]
            x_denoised = self.diffusion_module(x, sigma_i, single, pair)
            # Euler step: score = (x - D) / sigma^2, dx = score * (sigma_next - sigma_i)
            # Simplified: x = x + (sigma_next - sigma_i) / sigma_i * (x - x_denoised)
            if sigma_i > 0:
                d = (x - x_denoised) / sigma_i
                x = x + (sigma_next - sigma_i) * d

        # pLDDT
        plddt_logits = self.plddt_head(single)
        plddt = torch.softmax(plddt_logits, dim=-1)
        bins = torch.linspace(0, 1, self.num_plddt_bins, device=device)
        plddt_score = (plddt * bins).sum(dim=-1)

        return {"coords": x, "plddt": plddt_score}
