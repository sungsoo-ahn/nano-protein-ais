"""AlphaFold2 — structure prediction: amino acid sequence → 3D backbone frames.

Self-contained implementation with Pairformer (single+pair representation) and
SE(3) frame diffusion using product SO(3) x R(3) noise schedules.

Core ideas: Pairformer replaces Evoformer (drops MSA axis entirely), SE(3) frame
diffusion with IGSO3 rotations + cosine DDPM translations, FAPE loss.

All modules expect batched inputs with a leading B dimension.

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
C_ATOM = 256  # Denoiser dim
NUM_PAIRFORMER_BLOCKS = 4
PAIRFORMER_HEADS = 4
NUM_DENOISE_BLOCKS = 4
DENOISE_HEADS = 8
NUM_PLDDT_BINS = 50
NUM_SAMPLE_STEPS = 100

# SO(3) diffusion
SO3_SIGMA_MAX = 1.5

# R(3) diffusion
R3_SIGMA_MAX = 10.0

# Loss weights
FAPE_WEIGHT = 0.1
TRANS_WEIGHT = 1.0
ROT_WEIGHT = 0.5
PLDDT_WEIGHT = 0.01
FAPE_CLAMP = 10.0

# Ideal backbone geometry (Angstroms / radians)
IDEAL_N_CA_DIST = 1.458
IDEAL_CA_C_DIST = 1.523
IDEAL_N_CA_C_ANGLE = 1.937  # ~111 degrees

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


def backbone_from_frames(frames: RigidTransform):
    """Reconstruct N, CA, C from SE(3) frames + ideal bond geometry."""
    coords_CA = frames.trans
    # C along local x-axis
    c_local = torch.zeros_like(coords_CA)
    c_local[..., 0] = IDEAL_CA_C_DIST
    coords_C = frames.apply(c_local)
    # N in local x-y plane at ideal angle
    angle = IDEAL_N_CA_C_ANGLE
    n_local = torch.zeros_like(coords_CA)
    n_local[..., 0] = -IDEAL_N_CA_DIST * math.cos(math.pi - angle)
    n_local[..., 1] = IDEAL_N_CA_DIST * math.sin(math.pi - angle)
    coords_N = frames.apply(n_local)
    return coords_N, coords_CA, coords_C


# ---------------------------------------------------------------------------
# Pairformer (replaces Evoformer — no MSA axis)
# All modules expect batched input: single [B,L,C_S], pair [B,L,L,C_Z]
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
        """pair: [B, L, L, c_z]"""
        z = self.ln(pair)
        left = self.left_proj(z) * torch.sigmoid(self.left_gate(z))
        right = self.right_proj(z) * torch.sigmoid(self.right_gate(z))
        if self.mode == "outgoing":
            out = torch.einsum("bikc,bjkc->bijc", left, right)
        else:
            out = torch.einsum("bkic,bkjc->bijc", left, right)
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
        """pair: [B, L, L, c_z]"""
        B, L = pair.shape[0], pair.shape[1]
        if self.mode == "ending":
            pair = pair.transpose(1, 2).contiguous()
        z = self.ln(pair)
        q = self.to_q(z).view(B, L, L, self.n_heads, self.head_dim)
        k = self.to_k(z).view(B, L, L, self.n_heads, self.head_dim)
        v = self.to_v(z).view(B, L, L, self.n_heads, self.head_dim)
        attn = torch.einsum("bijhd,bikhd->bhijk", q, k) / (self.head_dim**0.5)
        # bias: [B,L,L,H] -> [B,H,L,1,L] to broadcast over j
        attn = attn + self.bias_proj(z).permute(0, 3, 1, 2).unsqueeze(2)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhijk,bikhd->bijhd", attn, v).reshape(B, L, L, self.c_z)
        result = pair + torch.sigmoid(self.gate(pair)) * self.to_out(out)
        if self.mode == "ending":
            result = result.transpose(1, 2).contiguous()
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
    """Self-attention on single rep [B, L, C_S] with pair bias from [B, L, L, C_Z]."""

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
        """single: [B, L, c_s], pair: [B, L, L, c_z]"""
        B, L = single.shape[:2]
        s = self.ln_s(single)
        z = self.ln_z(pair)
        q = self.to_q(s).view(B, L, self.n_heads, self.head_dim)
        k = self.to_k(s).view(B, L, self.n_heads, self.head_dim)
        v = self.to_v(s).view(B, L, self.n_heads, self.head_dim)
        attn = torch.einsum("bihd,bjhd->bhij", q, k) / (self.head_dim**0.5)
        attn = attn + self.pair_bias(z).permute(0, 3, 1, 2)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(B, L, self.c_s)
        return single + torch.sigmoid(self.gate(s)) * self.to_out(out)


class PairformerBlock(nn.Module):
    """Pairformer block: (single, pair) -> (single, pair).

    Pair stack: tri_out -> tri_in -> tri_attn_start -> tri_attn_end -> pair_transition
    Single stack: attention_with_pair_bias -> single_transition
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
# SE(3) Diffusion: SO(3) x R(3) product diffusion
# ---------------------------------------------------------------------------


def sample_igso3(shape: tuple, sigma, device=torch.device("cpu")) -> torch.Tensor:
    """Sample from Isotropic Gaussian on SO(3) via axis-angle."""
    if isinstance(sigma, torch.Tensor):
        s = sigma
        while s.dim() < len(shape):
            s = s.unsqueeze(-1)
        omega = torch.abs(torch.randn(*shape, device=device) * s)
    else:
        omega = torch.abs(torch.randn(*shape, device=device) * sigma)
    axis = torch.randn(*shape, 3, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    return axis_angle_to_rotation_matrix(axis * omega.unsqueeze(-1))


class SO3Diffuser:
    """SO(3) diffusion with linear sigma schedule and IGSO3 noise."""

    def __init__(self, sigma_max: float = SO3_SIGMA_MAX):
        self.sigma_max = sigma_max

    def sigma(self, t):
        return t * self.sigma_max

    def forward_marginal(self, rots_0, t):
        sigma = self.sigma(t)
        noise_rots = sample_igso3(rots_0.shape[:-2], sigma, rots_0.device)
        rots_t = torch.einsum("...ij,...jk->...ik", noise_rots, rots_0)
        return rots_t, noise_rots

    def reverse_step(self, rots_t, pred_rots_0, t_now, t_next):
        sigma_now = t_now * self.sigma_max
        sigma_next = t_next * self.sigma_max
        if sigma_now < 1e-8:
            return pred_rots_0
        interp = 1.0 - (sigma_next / (sigma_now + 1e-8))
        R_rel = torch.einsum("...ij,...kj->...ik", pred_rots_0, rots_t)
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos_angle = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)
        axis = torch.stack(
            [
                R_rel[..., 2, 1] - R_rel[..., 1, 2],
                R_rel[..., 0, 2] - R_rel[..., 2, 0],
                R_rel[..., 1, 0] - R_rel[..., 0, 1],
            ],
            dim=-1,
        )
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        scaled_rot = axis_angle_to_rotation_matrix(axis * (angle * interp).unsqueeze(-1))
        return torch.einsum("...ij,...jk->...ik", scaled_rot, rots_t)


class R3Diffuser:
    """R(3) diffusion with cosine DDPM schedule."""

    def __init__(self, sigma_max: float = R3_SIGMA_MAX):
        self.sigma_max = sigma_max

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        s = 0.008
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def forward_marginal(self, trans_0, t):
        ab = self.alpha_bar(t)
        while ab.dim() < trans_0.dim():
            ab = ab.unsqueeze(-1)
        noise = torch.randn_like(trans_0)
        return torch.sqrt(ab) * trans_0 + torch.sqrt(1 - ab) * noise, noise

    def reverse_step(self, trans_t, pred_trans_0, t_now, t_next):
        t_now_t = torch.tensor(t_now, device=trans_t.device)
        t_next_t = torch.tensor(t_next, device=trans_t.device)
        ab_now, ab_next = self.alpha_bar(t_now_t), self.alpha_bar(t_next_t)
        noise_est = (trans_t - torch.sqrt(ab_now) * pred_trans_0) / (
            torch.sqrt(1 - ab_now) + 1e-8
        )
        trans_mean = torch.sqrt(ab_next) * pred_trans_0 + torch.sqrt(1 - ab_next) * noise_est
        if t_next > 0:
            beta = 1 - ab_next / ab_now
            return trans_mean + torch.sqrt(beta.clamp(min=1e-8)) * torch.randn_like(trans_mean)
        return trans_mean


class SE3Diffusion:
    """Product SE(3) = SO(3) x R(3) diffusion on RigidTransforms."""

    def __init__(
        self,
        trans_sigma_max: float = R3_SIGMA_MAX,
        rot_sigma_max: float = SO3_SIGMA_MAX,
    ):
        self.so3 = SO3Diffuser(sigma_max=rot_sigma_max)
        self.r3 = R3Diffuser(sigma_max=trans_sigma_max)

    def forward_marginal(self, frames: RigidTransform, t):
        noisy_rots, noise_rots = self.so3.forward_marginal(frames.rots, t)
        noisy_trans, noise_trans = self.r3.forward_marginal(frames.trans, t)
        return RigidTransform(noisy_rots, noisy_trans), {"trans": noise_trans, "rot": noise_rots}

    def reverse_step(self, frames_t, pred_0, t_now, t_next):
        rots = self.so3.reverse_step(frames_t.rots, pred_0.rots, t_now, t_next)
        trans = self.r3.reverse_step(frames_t.trans, pred_0.trans, t_now, t_next)
        return RigidTransform(rots, trans)


# ---------------------------------------------------------------------------
# Diffusion Module (denoiser for SE(3) frame diffusion)
# Inputs: noisy frames, timestep, single [B,L,C_S], pair [B,L,L,C_Z]
# ---------------------------------------------------------------------------


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal embedding of timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> output: [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
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
        """x: [B, L, dim], cond: [B, dim]"""
        scale_shift = self.proj(cond)
        while scale_shift.dim() < x.dim():
            scale_shift = scale_shift.unsqueeze(-2)
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

    def forward(
        self, x: torch.Tensor, pair: torch.Tensor, time_cond: torch.Tensor
    ) -> torch.Tensor:
        """x: [B,L,c_atom], pair: [B,L,L,c_z], time_cond: [B,c_atom]"""
        B, L = x.shape[:2]
        h = self.adaln1(x, time_cond)
        q = self.to_q(h).view(B, L, self.n_heads, self.head_dim)
        k = self.to_k(h).view(B, L, self.n_heads, self.head_dim)
        v = self.to_v(h).view(B, L, self.n_heads, self.head_dim)
        attn = torch.einsum("bihd,bjhd->bhij", q, k) / (self.head_dim**0.5)
        attn = attn + self.pair_bias(pair).permute(0, 3, 1, 2)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(B, L, self.c_atom)
        x = x + self.to_out(out)
        x = x + self.ffn(self.adaln2(x, time_cond))
        return x


class DiffusionModule(nn.Module):
    """Denoising network for SE(3) frame diffusion.

    Input: noisy frames (RigidTransform) + timestep + single/pair conditioning.
    Output: predicted clean frames (RigidTransform).
    """

    def __init__(
        self,
        c_s: int = C_S,
        c_z: int = C_Z,
        c_atom: int = C_ATOM,
        n_blocks: int = NUM_DENOISE_BLOCKS,
        n_heads: int = DENOISE_HEADS,
    ):
        super().__init__()
        self.frame_proj = nn.Linear(12, c_atom)  # rots[9] + trans[3]
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
        self.output_proj = nn.Linear(c_atom, 6)  # 3 axis-angle + 3 translation
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_frames: RigidTransform,
        t: torch.Tensor,
        single: torch.Tensor,
        pair: torch.Tensor,
    ) -> RigidTransform:
        """Predict denoised frames.

        Args:
            noisy_frames: RigidTransform with rots [B,L,3,3] and trans [B,L,3]
            t: [B] timestep in [0, 1]
            single: [B, L, C_S] single representation from Pairformer
            pair: [B, L, L, C_Z] pair representation from Pairformer
        Returns:
            Predicted clean frames (RigidTransform)
        """
        B, L = noisy_frames.trans.shape[:2]

        # Flatten frame: cat(rots.reshape(B,L,9), trans) -> [B,L,12]
        frame_feat = torch.cat(
            [noisy_frames.rots.reshape(B, L, 9), noisy_frames.trans], dim=-1
        )
        h = self.frame_proj(frame_feat) + self.single_cond_proj(single)

        # Time conditioning
        time_emb = self.time_embed(t)
        time_cond = self.time_mlp(time_emb)

        for block in self.blocks:
            h = block(h, pair, time_cond)

        # Predict frame correction: 3 axis-angle rotation + 3 translation
        correction = self.output_proj(h)  # [B, L, 6]
        rot_update = axis_angle_to_rotation_matrix(correction[..., :3])
        trans_update = correction[..., 3:]

        # Compose correction onto noisy frame
        update = RigidTransform(rot_update, trans_update)
        return noisy_frames.compose(update)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


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
# Losses (all handle batched inputs)
# ---------------------------------------------------------------------------


def fape_loss(
    pred_frames: RigidTransform,
    true_frames: RigidTransform,
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    mask: torch.Tensor | None = None,
    clamp: float = FAPE_CLAMP,
) -> torch.Tensor:
    """Frame Aligned Point Error.

    For each frame f, transform all CA positions into frame-local coords,
    then compute L2 distance between predicted and true local positions.

    Args:
        pred_frames, true_frames: RigidTransform [B, L, ...]
        pred_ca, true_ca: [B, L, 3]
        mask: [B, L] bool
        clamp: max error per pair
    Returns:
        scalar loss
    """
    # pred_delta: [B, F, 1, 3] - [B, 1, A, 3] = [B, F, A, 3]
    pred_delta = pred_ca.unsqueeze(1) - pred_frames.trans.unsqueeze(2)
    pred_local = torch.einsum(
        "bfij,bfaj->bfai", pred_frames.rots.transpose(-1, -2), pred_delta
    )

    true_delta = true_ca.unsqueeze(1) - true_frames.trans.unsqueeze(2)
    true_local = torch.einsum(
        "bfij,bfaj->bfai", true_frames.rots.transpose(-1, -2), true_delta
    )

    error = ((pred_local - true_local) ** 2).sum(-1).add(1e-8).sqrt()  # [B, F, A]
    if clamp > 0:
        error = error.clamp(max=clamp)

    if mask is not None:
        # pair_mask: [B, F, A] — both frame and atom must be valid
        pair_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).float()
        return (error * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    return error.mean()


def frame_loss(
    pred_frames: RigidTransform,
    true_frames: RigidTransform,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct frame comparison: translation MSE + rotation geodesic distance.

    More stable than FAPE for training — guaranteed non-zero gradients.
    Returns (trans_loss, rot_loss) as separate scalars.
    """
    mask_f = mask.float() if mask is not None else torch.ones_like(pred_frames.trans[..., 0])

    # Translation: MSE
    trans_diff = ((pred_frames.trans - true_frames.trans) ** 2).sum(-1)  # [B, L]
    trans_loss = (trans_diff * mask_f).sum() / mask_f.sum().clamp(min=1)

    # Rotation: geodesic distance on SO(3)
    R_diff = torch.einsum("...ij,...kj->...ik", pred_frames.rots, true_frames.rots)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    rot_loss = (torch.acos(cos_angle) ** 2 * mask_f).sum() / mask_f.sum().clamp(min=1)

    return trans_loss, rot_loss


def plddt_loss(
    pred_logits: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_bins: int = NUM_PLDDT_BINS,
) -> torch.Tensor:
    """pred_logits [B,L,bins], pred/true_coords [B,L,3], mask [B,L]."""
    diff = (pred_coords - true_coords).norm(dim=-1)  # [B,L]
    lddt = torch.clamp(1.0 - diff / 15.0, 0.0, 1.0)
    true_bins = (lddt * (num_bins - 1)).long().clamp(0, num_bins - 1)
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]), true_bins.reshape(-1), reduction="none"
    ).reshape_as(true_bins)
    if mask is not None:
        return (loss * mask.float()).sum() / mask.sum().clamp(min=1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Input Embedding (batched)
# ---------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    """Embed sequence -> (single [B, L, C_S], pair [B, L, L, C_Z]). No MSA axis."""

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
        """sequence: [B, L], residue_index: [B, L]"""
        B, L = sequence.shape
        one_hot = torch.zeros(B, L, 21, device=sequence.device)
        one_hot.scatter_(2, sequence.clamp(0, 20).unsqueeze(2), 1.0)
        single = self.single_proj(one_hot)  # [B, L, c_s]
        left, right = self.left_single(one_hot), self.right_single(one_hot)
        pair = left[:, :, None, :] + right[:, None, :, :]  # [B, L, L, c_z]
        d = torch.clamp(residue_index[:, :, None] - residue_index[:, None, :] + 32, 0, 64).long()
        return single, pair + self.relpos(d)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AlphaFold2(nn.Module):
    """AlphaFold2 protein structure prediction model.

    Single-sequence mode. Predicts backbone frames from sequence using
    Pairformer trunk + SE(3) frame diffusion (product SO(3) x R(3)).
    Default: 4 pairformer blocks, 4 denoise blocks = ~30M params.
    """

    def __init__(
        self,
        c_s: int = C_S,
        c_z: int = C_Z,
        c_atom: int = C_ATOM,
        num_pairformer_blocks: int = NUM_PAIRFORMER_BLOCKS,
        pairformer_heads: int = PAIRFORMER_HEADS,
        num_denoise_blocks: int = NUM_DENOISE_BLOCKS,
        denoise_heads: int = DENOISE_HEADS,
        num_plddt_bins: int = NUM_PLDDT_BINS,
        fape_weight: float = FAPE_WEIGHT,
        trans_weight: float = TRANS_WEIGHT,
        rot_weight: float = ROT_WEIGHT,
        plddt_weight: float = PLDDT_WEIGHT,
    ):
        super().__init__()
        self.fape_weight = fape_weight
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.plddt_weight = plddt_weight
        self.num_plddt_bins = num_plddt_bins

        self.input_embedding = InputEmbedding(c_s, c_z)
        self.pairformer_blocks = nn.ModuleList(
            [PairformerBlock(c_s, c_z, pairformer_heads) for _ in range(num_pairformer_blocks)]
        )
        self.diffusion = SE3Diffusion()
        self.diffusion_module = DiffusionModule(c_s, c_z, c_atom, num_denoise_blocks, denoise_heads)
        self.plddt_head = PLDDTHead(c_s, num_plddt_bins)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-zero output_proj so model starts with near-identity frame corrections
        nn.init.zeros_(self.diffusion_module.output_proj.weight)
        nn.init.zeros_(self.diffusion_module.output_proj.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass — fully batched.

        Args:
            batch: dict with coords_N/CA/C [B,L,3], sequence [B,L], mask [B,L].
        Returns:
            dict with loss (scalar), fape_loss (scalar).
        """
        B, L = batch["sequence"].shape
        device = batch["sequence"].device
        seq = batch["sequence"]
        mask = batch["mask"]
        true_N = batch["coords_N"]
        true_CA = batch["coords_CA"]
        true_C = batch["coords_C"]

        # Pairformer trunk
        residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        single, pair = self.input_embedding(seq, residue_index)
        for block in self.pairformer_blocks:
            single, pair = block(single, pair)

        # Compute true frames from backbone atoms
        true_frames = compute_local_frame(true_N, true_CA, true_C)

        # Sample timestep and apply SE(3) noise
        t = torch.rand(B, device=device)
        t_expanded = t[:, None].expand(-1, L)
        noisy_frames, _ = self.diffusion.forward_marginal(true_frames, t_expanded)

        # Denoise
        pred_frames = self.diffusion_module(noisy_frames, t, single, pair)
        _, pred_CA, _ = backbone_from_frames(pred_frames)

        # Losses: direct frame loss (stable gradients) + FAPE (structural quality)
        loss_trans, loss_rot = frame_loss(pred_frames, true_frames, mask)
        loss_fape = fape_loss(pred_frames, true_frames, pred_CA, true_CA, mask)

        plddt_logits = self.plddt_head(single)
        loss_plddt = plddt_loss(plddt_logits, pred_CA.detach(), true_CA, mask, self.num_plddt_bins)

        total = (
            self.trans_weight * loss_trans
            + self.rot_weight * loss_rot
            + self.fape_weight * loss_fape
            + self.plddt_weight * loss_plddt
        )

        return {
            "loss": total,
            "fape_loss": loss_fape,
            "trans_loss": loss_trans,
            "rot_loss": loss_rot,
        }

    @torch.no_grad()
    def predict(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict structure from sequence via reverse SE(3) diffusion.

        Args:
            sequence: [L] unbatched sequence tensor.
        Returns:
            dict with frames, coords_N/CA/C [L,3], plddt [L].
        """
        self.eval()
        L = sequence.shape[0]
        device = sequence.device

        # Add batch dim
        seq = sequence.unsqueeze(0)  # [1, L]
        residue_index = torch.arange(L, device=device).unsqueeze(0)
        single, pair = self.input_embedding(seq, residue_index)
        for block in self.pairformer_blocks:
            single, pair = block(single, pair)

        # Initialize from noise (t=1)
        frames = RigidTransform.identity((1, L), device=device)
        t_init = torch.ones(1, L, device=device)
        frames, _ = self.diffusion.forward_marginal(frames, t_init)

        # Reverse diffusion loop
        timesteps = torch.linspace(1, 0, NUM_SAMPLE_STEPS + 1, device=device)
        for i in range(NUM_SAMPLE_STEPS):
            t_now = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            t_batch = torch.tensor([t_now], device=device)
            pred = self.diffusion_module(frames, t_batch, single, pair)
            if t_next > 0:
                frames = self.diffusion.reverse_step(frames, pred, t_now, t_next)
            else:
                frames = pred

        # Reconstruct backbone and remove batch dim
        coords_N, coords_CA, coords_C = backbone_from_frames(frames)
        coords_N = coords_N.squeeze(0)
        coords_CA = coords_CA.squeeze(0)
        coords_C = coords_C.squeeze(0)

        # pLDDT
        plddt_logits = self.plddt_head(single).squeeze(0)
        plddt = torch.softmax(plddt_logits, dim=-1)
        bins = torch.linspace(0, 1, self.num_plddt_bins, device=device)
        plddt_score = (plddt * bins).sum(dim=-1)

        return {
            "frames": RigidTransform(frames.rots.squeeze(0), frames.trans.squeeze(0)),
            "coords_N": coords_N,
            "coords_CA": coords_CA,
            "coords_C": coords_C,
            "plddt": plddt_score,
        }
