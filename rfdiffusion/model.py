"""RFDiffusion — de novo backbone generation via SE(3) diffusion.

Self-contained implementation with SO(3)+R(3) noise schedules, denoising
network with IPA, triangular updates, and reverse diffusion sampling.

Total parameters: ~35M with default config.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_DIM = 256
PAIR_DIM = 64
NUM_BLOCKS = 8
N_HEADS = 8
N_QK_POINTS = 4
N_V_POINTS = 8
NUM_TIMESTEPS = 100
TRANS_SIGMA_MAX = 10.0
ROT_SIGMA_MAX = 1.5
SELF_CONDITIONING = True
TRANS_LOSS_WEIGHT = 1.0
ROT_LOSS_WEIGHT = 1.0

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
    angle = axis_angle.norm(dim=-1, keepdim=True).unsqueeze(-1)
    axis = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)
    K = skew_symmetric(axis)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    return eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y], -1
            ),
            torch.stack(
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x], -1
            ),
            torch.stack(
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y], -1
            ),
        ],
        dim=-2,
    )


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    s = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) * 2
    w = s / 4
    x = (R[..., 2, 1] - R[..., 1, 2]) / (s + 1e-8)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (s + 1e-8)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (s + 1e-8)
    return F.normalize(torch.stack([w, x, y, z], dim=-1), dim=-1)


class RigidTransform:
    def __init__(self, rots: torch.Tensor, trans: torch.Tensor):
        self.rots = rots
        self.trans = trans

    @classmethod
    def identity(cls, batch_shape: tuple, device="cpu", dtype=torch.float32):
        rots = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).clone()
        trans = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
        return cls(rots, trans)

    def compose(self, other):
        new_rots = self.rots @ other.rots
        new_trans = (self.rots @ other.trans.unsqueeze(-1)).squeeze(-1) + self.trans
        return RigidTransform(new_rots, new_trans)

    def invert(self):
        inv_rots = self.rots.transpose(-1, -2)
        inv_trans = -(inv_rots @ self.trans.unsqueeze(-1)).squeeze(-1)
        return RigidTransform(inv_rots, inv_trans)

    def to_tensor_7(self) -> torch.Tensor:
        return torch.cat([rotation_matrix_to_quaternion(self.rots), self.trans], dim=-1)

    @classmethod
    def from_tensor_7(cls, tensor: torch.Tensor):
        return cls(quaternion_to_rotation_matrix(tensor[..., :4]), tensor[..., 4:])

    def detach(self):
        return RigidTransform(self.rots.detach(), self.trans.detach())


def compute_local_frame(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor) -> RigidTransform:
    x = C - CA
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    v = N - CA
    z = torch.cross(x, v, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=-1)
    return RigidTransform(torch.stack([x, y, z], dim=-1), CA)


# ---------------------------------------------------------------------------
# Diffusion: SO(3) + R(3)
# ---------------------------------------------------------------------------


def sample_igso3(shape: tuple, sigma, device=torch.device("cpu")) -> torch.Tensor:
    """Sample from Isotropic Gaussian on SO(3)."""
    if isinstance(sigma, (int, float)):
        omega = torch.abs(torch.randn(*shape, device=device) * sigma)
    else:
        omega = torch.abs(torch.randn(*shape, device=device) * sigma)
    axis = torch.randn(*shape, 3, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    return axis_angle_to_rotation_matrix(axis * omega.unsqueeze(-1))


class SO3Diffuser:
    def __init__(self, sigma_max: float = ROT_SIGMA_MAX):
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
    def __init__(self, sigma_max: float = TRANS_SIGMA_MAX):
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
        noise_est = (trans_t - torch.sqrt(ab_now) * pred_trans_0) / (torch.sqrt(1 - ab_now) + 1e-8)
        trans_mean = torch.sqrt(ab_next) * pred_trans_0 + torch.sqrt(1 - ab_next) * noise_est
        if t_next > 0:
            beta = 1 - ab_next / ab_now
            return trans_mean + torch.sqrt(beta.clamp(min=1e-8)) * torch.randn_like(trans_mean)
        return trans_mean


class SE3Diffusion:
    def __init__(
        self,
        trans_sigma_max: float = TRANS_SIGMA_MAX,
        rot_sigma_max: float = ROT_SIGMA_MAX,
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
# Network modules (IPA, TriangularUpdate inlined from AF2-like architecture)
# ---------------------------------------------------------------------------


class Transition(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion)
        self.linear2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        return x + self.linear2(F.relu(self.linear1(self.ln(x))))


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

    def forward(self, pair):
        z = self.ln(pair)
        left = self.left_proj(z) * torch.sigmoid(self.left_gate(z))
        right = self.right_proj(z) * torch.sigmoid(self.right_gate(z))
        if self.mode == "outgoing":
            out = torch.einsum("ikc,jkc->ijc", left, right)
        else:
            out = torch.einsum("kic,kjc->ijc", left, right)
        out = self.output_proj(self.final_norm(out))
        return pair + torch.sigmoid(self.output_gate(pair)) * out


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_heads: int = 8,
        n_qk_points: int = 4,
        n_v_points: int = 8,
    ):
        super().__init__()
        self.c_s, self.n_heads = c_s, n_heads
        self.head_dim = c_s // n_heads
        self.n_qk_points, self.n_v_points = n_qk_points, n_v_points
        self.ln = nn.LayerNorm(c_s)
        self.to_q = nn.Linear(c_s, c_s, bias=False)
        self.to_k = nn.Linear(c_s, c_s, bias=False)
        self.to_v = nn.Linear(c_s, c_s, bias=False)
        self.to_q_pts = nn.Linear(c_s, n_heads * n_qk_points * 3)
        self.to_k_pts = nn.Linear(c_s, n_heads * n_qk_points * 3)
        self.to_v_pts = nn.Linear(c_s, n_heads * n_v_points * 3)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.head_weights = nn.Parameter(torch.zeros(n_heads))
        out_dim = c_s + n_heads * n_v_points * 3 + n_heads * c_z
        self.to_out = nn.Linear(out_dim, c_s)

    def forward(self, single, pair, rigids):
        L = single.shape[0]
        s = self.ln(single)
        q = self.to_q(s).view(L, self.n_heads, self.head_dim)
        k = self.to_k(s).view(L, self.n_heads, self.head_dim)
        v = self.to_v(s).view(L, self.n_heads, self.head_dim)
        q_pts = self.to_q_pts(s).view(L, self.n_heads, self.n_qk_points, 3)
        k_pts = self.to_k_pts(s).view(L, self.n_heads, self.n_qk_points, 3)
        v_pts = self.to_v_pts(s).view(L, self.n_heads, self.n_v_points, 3)

        def apply_frames(pts):
            HP = pts.shape[1] * pts.shape[2]
            flat = pts.reshape(L, HP, 3)
            rotated = torch.einsum("lij,lnj->lni", rigids.rots, flat)
            return (rotated + rigids.trans[:, None, :]).view_as(pts)

        q_g, k_g, v_g = apply_frames(q_pts), apply_frames(k_pts), apply_frames(v_pts)
        attn = torch.einsum("ihd,jhd->hij", q, k) / (self.head_dim**0.5)
        pt_diff = q_g[:, None] - k_g[None, :]
        pt_dist_sq = (pt_diff**2).sum(-1).sum(-1)
        w_c = F.softplus(self.head_weights)
        attn = attn - 0.5 * w_c[:, None, None] * pt_dist_sq.permute(2, 0, 1)
        attn = attn + self.pair_bias(pair).permute(2, 0, 1)
        attn = torch.softmax(attn, dim=-1)
        out_scalar = torch.einsum("hij,jhd->ihd", attn, v).reshape(L, self.c_s)
        out_pts_g = torch.einsum("hij,jhpc->ihpc", attn, v_g)
        inv_rots = rigids.rots.transpose(-1, -2)
        HP_v = self.n_heads * self.n_v_points
        flat_pts = out_pts_g.reshape(L, HP_v, 3) - rigids.trans[:, None, :]
        out_pts_l = torch.einsum("lij,lnj->lni", inv_rots, flat_pts).reshape(L, HP_v * 3)
        n_pair = self.n_heads * pair.shape[-1]
        out_pair = torch.einsum("hij,ijc->ihc", attn, pair).reshape(L, n_pair)
        return self.to_out(torch.cat([out_scalar, out_pts_l, out_pair], -1))


# ---------------------------------------------------------------------------
# Denoising Network
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        t_flat = t.reshape(-1)
        args = t_flat[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb.view(*t.shape, self.dim)


class DenoisingBlock(nn.Module):
    def __init__(self, node_dim: int, pair_dim: int, n_heads=8, n_qk_points=4, n_v_points=8):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.ipa_norm = nn.LayerNorm(node_dim)
        self.ipa = InvariantPointAttention(node_dim, pair_dim, n_heads, n_qk_points, n_v_points)
        self.transition = Transition(node_dim)
        self.tri_out = TriangularMultiplicativeUpdate(pair_dim, pair_dim, "outgoing")
        self.tri_in = TriangularMultiplicativeUpdate(pair_dim, pair_dim, "incoming")
        self.pair_transition = Transition(pair_dim)
        self.frame_norm = nn.LayerNorm(node_dim)
        self.frame_update = nn.Linear(node_dim, 6)
        nn.init.zeros_(self.frame_update.weight)
        nn.init.zeros_(self.frame_update.bias)

    def forward(self, node_feat, pair_feat, frames, t_embed):
        node_feat = node_feat + self.time_proj(t_embed).unsqueeze(0)
        node_feat = node_feat + self.ipa(self.ipa_norm(node_feat), pair_feat, frames)
        node_feat = self.transition(node_feat)
        pair_feat = self.tri_out(pair_feat)
        pair_feat = self.tri_in(pair_feat)
        pair_feat = self.pair_transition(pair_feat)
        update = self.frame_update(self.frame_norm(node_feat))
        rot_mat = axis_angle_to_rotation_matrix(update[:, :3] * 0.1)
        frames = frames.compose(RigidTransform(rot_mat, update[:, 3:]))
        return node_feat, pair_feat, frames


class DenoisingNetwork(nn.Module):
    """Takes noisy frames + timestep, produces predicted clean frames."""

    def __init__(
        self,
        node_dim=NODE_DIM,
        pair_dim=PAIR_DIM,
        num_blocks=NUM_BLOCKS,
        n_heads=N_HEADS,
        n_qk_points=N_QK_POINTS,
        n_v_points=N_V_POINTS,
        self_conditioning=SELF_CONDITIONING,
    ):
        super().__init__()
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.self_conditioning = self_conditioning
        self.time_embed = SinusoidalTimestepEmbedding(node_dim)
        node_input = 7 + (7 if self_conditioning else 0)
        self.node_proj = nn.Linear(node_input, node_dim)
        self.pair_proj = nn.Linear(65, pair_dim)
        self.blocks = nn.ModuleList(
            [
                DenoisingBlock(node_dim, pair_dim, n_heads, n_qk_points, n_v_points)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(node_dim)
        self.rot_head = nn.Linear(node_dim, 3)
        self.trans_head = nn.Linear(node_dim, 3)
        nn.init.zeros_(self.rot_head.weight)
        nn.init.zeros_(self.rot_head.bias)
        nn.init.zeros_(self.trans_head.weight)
        nn.init.zeros_(self.trans_head.bias)

    def forward(self, noisy_frames, t, prev_pred=None):
        L = noisy_frames.trans.shape[0]
        device = noisy_frames.trans.device
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_embed = self.time_embed(t).squeeze(0)
        frame_7d = noisy_frames.to_tensor_7()
        if self.self_conditioning and prev_pred is not None:
            node_input = torch.cat([frame_7d, prev_pred.to_tensor_7().detach()], dim=-1)
        elif self.self_conditioning:
            node_input = torch.cat([frame_7d, torch.zeros_like(frame_7d)], dim=-1)
        else:
            node_input = frame_7d
        node_feat = self.node_proj(node_input)
        idx = torch.arange(L, device=device)
        d = torch.clamp(idx[:, None] - idx[None, :] + 32, 0, 64).long()
        pair_feat = self.pair_proj(F.one_hot(d, 65).float())
        frames = RigidTransform(noisy_frames.rots.clone(), noisy_frames.trans.clone())
        for block in self.blocks:
            node_feat, pair_feat, frames = block(node_feat, pair_feat, frames, t_embed)
        h = self.output_norm(node_feat)
        pred_rots = axis_angle_to_rotation_matrix(self.rot_head(h))
        correction = RigidTransform(pred_rots, self.trans_head(h))
        return noisy_frames.compose(correction)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RFDiffusion(nn.Module):
    """RFDiffusion: SE(3) diffusion for de novo backbone generation.

    Default: 8 blocks, 256 node dim = ~35M params.
    """

    def __init__(
        self,
        node_dim=NODE_DIM,
        pair_dim=PAIR_DIM,
        num_blocks=NUM_BLOCKS,
        n_heads=N_HEADS,
        n_qk_points=N_QK_POINTS,
        n_v_points=N_V_POINTS,
        num_timesteps=NUM_TIMESTEPS,
        trans_sigma_max=TRANS_SIGMA_MAX,
        rot_sigma_max=ROT_SIGMA_MAX,
        self_conditioning=SELF_CONDITIONING,
        trans_loss_weight=TRANS_LOSS_WEIGHT,
        rot_loss_weight=ROT_LOSS_WEIGHT,
    ):
        super().__init__()
        self.trans_loss_weight = trans_loss_weight
        self.rot_loss_weight = rot_loss_weight
        self.network = DenoisingNetwork(
            node_dim,
            pair_dim,
            num_blocks,
            n_heads,
            n_qk_points,
            n_v_points,
            self_conditioning,
        )
        self.diffusion = SE3Diffusion(trans_sigma_max, rot_sigma_max)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        B = batch["coords_CA"].shape[0]
        device = batch["coords_CA"].device
        all_losses, all_trans, all_rot = [], [], []

        for b in range(B):
            mask = batch["mask"][b]
            N, CA, C = batch["coords_N"][b], batch["coords_CA"][b], batch["coords_C"][b]
            true_frames = compute_local_frame(N, CA, C)

            t = torch.rand(1, device=device).squeeze()
            noisy_frames, _ = self.diffusion.forward_marginal(true_frames, t.expand(CA.shape[0]))
            pred_frames = self.network(noisy_frames, t)

            trans_loss, rot_loss = self._compute_loss(pred_frames, true_frames, mask)
            total = self.trans_loss_weight * trans_loss + self.rot_loss_weight * rot_loss
            all_losses.append(total)
            all_trans.append(trans_loss)
            all_rot.append(rot_loss)

        return {
            "loss": torch.stack(all_losses).mean(),
            "trans_loss": torch.stack(all_trans).mean(),
            "rot_loss": torch.stack(all_rot).mean(),
        }

    def _compute_loss(self, pred, true, mask):
        mask_f = mask.float()
        trans_diff = (pred.trans - true.trans) ** 2
        trans_loss = (trans_diff.sum(-1) * mask_f).sum() / mask_f.sum().clamp(min=1)
        R_diff = torch.einsum("...ij,...kj->...ik", pred.rots, true.rots)
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        cos_angle = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
        rot_loss = ((torch.acos(cos_angle) ** 2) * mask_f).sum() / mask_f.sum().clamp(min=1)
        return trans_loss, rot_loss


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample(
    network: DenoisingNetwork,
    diffusion: SE3Diffusion,
    num_residues: int,
    num_steps: int = 100,
    device=torch.device("cpu"),
    self_conditioning: bool = True,
) -> list[RigidTransform]:
    """Generate a protein backbone via reverse diffusion."""
    network.eval()
    L = num_residues
    frames = RigidTransform.identity((L,), device=device)
    frames, _ = diffusion.forward_marginal(frames, torch.ones(L, device=device))

    prev_pred = None
    timesteps = torch.linspace(1, 0, num_steps + 1)
    trajectory = [frames]

    for i in range(num_steps):
        t_now, t_next = timesteps[i].item(), timesteps[i + 1].item()
        pred = network(frames, torch.tensor(t_now, device=device), prev_pred)
        if self_conditioning and torch.rand(1).item() < 0.5:
            prev_pred = pred.detach()
        if t_next > 0:
            frames = diffusion.reverse_step(frames, pred, t_now, t_next)
        else:
            frames = pred
        trajectory.append(RigidTransform(frames.rots.detach(), frames.trans.detach()))

    return trajectory
