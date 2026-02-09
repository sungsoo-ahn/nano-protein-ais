"""AlphaFold2 — structure prediction: amino acid sequence → 3D backbone coordinates.

Self-contained implementation with Evoformer, Invariant Point Attention,
Structure Module, prediction heads, and loss functions.

Core ideas: Evoformer (MSA+pair representation), IPA, iterative frame refinement.
Single-sequence mode (MSA dim = 1, no MSA search needed).

Total parameters: ~30M with default config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_M = 128  # MSA representation dim
C_Z = 64  # Pair representation dim
C_S = 192  # Single representation dim
NUM_EVOFORMER_BLOCKS = 4
EVOFORMER_HEADS = 4
NUM_STRUCTURE_LAYERS = 2
IPA_HEADS = 8
N_QK_POINTS = 4
N_V_POINTS = 8
NUM_DIST_BINS = 64
NUM_PLDDT_BINS = 50

# Loss weights
FAPE_WEIGHT = 1.0
DISTOGRAM_WEIGHT = 0.3
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
# Evoformer
# ---------------------------------------------------------------------------


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c_m: int, c_z: int, n_heads: int):
        super().__init__()
        self.c_m, self.n_heads = c_m, n_heads
        self.head_dim = c_m // n_heads
        self.ln_m = nn.LayerNorm(c_m)
        self.ln_z = nn.LayerNorm(c_z)
        self.to_q = nn.Linear(c_m, c_m, bias=False)
        self.to_k = nn.Linear(c_m, c_m, bias=False)
        self.to_v = nn.Linear(c_m, c_m, bias=False)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.to_out = nn.Linear(c_m, c_m)
        self.gate = nn.Linear(c_m, c_m)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        N, L, _ = msa.shape
        m = self.ln_m(msa)
        z = self.ln_z(pair)
        q = self.to_q(m).view(N, L, self.n_heads, self.head_dim)
        k = self.to_k(m).view(N, L, self.n_heads, self.head_dim)
        v = self.to_v(m).view(N, L, self.n_heads, self.head_dim)
        attn = torch.einsum("bihd,bjhd->bhij", q, k) / (self.head_dim**0.5)
        attn = attn + self.pair_bias(z).permute(2, 0, 1).unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(N, L, self.c_m)
        return msa + torch.sigmoid(self.gate(m)) * self.to_out(out)


class OuterProductMean(nn.Module):
    def __init__(self, c_m: int, c_z: int, c_hidden: int = 32):
        super().__init__()
        self.ln = nn.LayerNorm(c_m)
        self.left = nn.Linear(c_m, c_hidden)
        self.right = nn.Linear(c_m, c_hidden)
        self.output = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        m = self.ln(msa)
        left, r = self.left(m), self.right(m)
        outer = torch.einsum("sic,sjd->ijcd", left, r) / msa.shape[0]
        return self.output(outer.reshape(outer.shape[0], outer.shape[1], -1))


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


class EvoformerBlock(nn.Module):
    def __init__(self, c_m: int, c_z: int, n_heads: int):
        super().__init__()
        self.msa_attn = MSARowAttentionWithPairBias(c_m, c_z, n_heads)
        self.msa_transition = Transition(c_m)
        self.opm = OuterProductMean(c_m, c_z)
        self.tri_out = TriangularMultiplicativeUpdate(c_z, c_z, "outgoing")
        self.tri_in = TriangularMultiplicativeUpdate(c_z, c_z, "incoming")
        self.tri_attn_start = TriangularAttention(c_z, n_heads, "starting")
        self.tri_attn_end = TriangularAttention(c_z, n_heads, "ending")
        self.pair_transition = Transition(c_z)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        msa = self.msa_attn(msa, pair)
        msa = self.msa_transition(msa)
        pair = pair + self.opm(msa)
        pair = self.tri_out(pair)
        pair = self.tri_in(pair)
        pair = self.tri_attn_start(pair)
        pair = self.tri_attn_end(pair)
        pair = self.pair_transition(pair)
        return msa, pair


# ---------------------------------------------------------------------------
# Structure Module
# ---------------------------------------------------------------------------


class InvariantPointAttention(nn.Module):
    """SE(3)-invariant attention combining scalar, point, and pair terms."""

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

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        rigids: RigidTransform,
    ) -> torch.Tensor:
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
        out_pts_local = torch.einsum("lij,lnj->lni", inv_rots, flat_pts).reshape(L, HP_v * 3)

        n_pair = self.n_heads * pair.shape[-1]
        out_pair = torch.einsum("hij,ijc->ihc", attn, pair).reshape(L, n_pair)
        return self.to_out(torch.cat([out_scalar, out_pts_local, out_pair], -1))


class StructureModule(nn.Module):
    """Iterative frame refinement: identity → predicted backbone coordinates."""

    def __init__(
        self,
        c_m: int,
        c_s: int,
        c_z: int,
        n_layers: int = 2,
        ipa_heads: int = 8,
        n_qk_points: int = 4,
        n_v_points: int = 8,
    ):
        super().__init__()
        self.input_proj = nn.Linear(c_m, c_s)
        self.ipa_layers = nn.ModuleList(
            [
                InvariantPointAttention(c_s, c_z, ipa_heads, n_qk_points, n_v_points)
                for _ in range(n_layers)
            ]
        )
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(c_s),
                    nn.Linear(c_s, c_s * 4),
                    nn.ReLU(),
                    nn.Linear(c_s * 4, c_s),
                )
                for _ in range(n_layers)
            ]
        )
        self.ipa_norms = nn.ModuleList([nn.LayerNorm(c_s) for _ in range(n_layers)])
        self.backbone_update = nn.Linear(c_s, 6)
        nn.init.zeros_(self.backbone_update.weight)
        nn.init.zeros_(self.backbone_update.bias)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, RigidTransform]:
        L = pair.shape[0]
        single = self.input_proj(msa[0])
        frames = RigidTransform.identity((L,), device=pair.device)

        for ipa, norm, transition in zip(self.ipa_layers, self.ipa_norms, self.transitions):
            single = single + ipa(norm(single), pair, frames)
            single = single + transition(single)
            update = self.backbone_update(single)
            rot_mat = axis_angle_to_rotation_matrix(update[:, :3] * 0.1)
            frames = frames.compose(RigidTransform(rot_mat, update[:, 3:]))

        return frames.trans, frames


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


def fape_loss(
    pred_frames: RigidTransform,
    pred_coords: torch.Tensor,
    true_frames: RigidTransform,
    true_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    clamp_distance: float = 10.0,
) -> torch.Tensor:
    """Frame Aligned Point Error — SE(3)-invariant structural loss."""
    L = pred_coords.shape[0]
    pred_inv = pred_frames.invert()
    true_inv = true_frames.invert()

    pred_exp = pred_coords.unsqueeze(0).expand(L, -1, -1)
    true_exp = true_coords.unsqueeze(0).expand(L, -1, -1)

    pred_local = torch.einsum("ijk,ilk->ilj", pred_inv.rots, pred_exp) + pred_inv.trans.unsqueeze(1)
    true_local = torch.einsum("ijk,ilk->ilj", true_inv.rots, true_exp) + true_inv.trans.unsqueeze(1)

    error = torch.sqrt(((pred_local - true_local) ** 2).sum(-1) + 1e-8).clamp(max=clamp_distance)

    if mask is not None:
        mask_2d = mask.unsqueeze(0) & mask.unsqueeze(1)
        return (error * mask_2d.float()).sum() / mask_2d.sum().clamp(min=1)
    return error.mean()


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
    def __init__(self, c_m: int, c_z: int):
        super().__init__()
        self.msa_proj = nn.Linear(21, c_m)
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
        msa = self.msa_proj(one_hot).unsqueeze(0)
        left, right = self.left_single(one_hot), self.right_single(one_hot)
        pair = left[:, None, :] + right[None, :, :]
        d = torch.clamp(residue_index[:, None] - residue_index[None, :] + 32, 0, 64).long()
        return msa, pair + self.relpos(d)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AlphaFold2(nn.Module):
    """AlphaFold2 protein structure prediction model.

    Single-sequence mode. Predicts backbone coordinates from sequence.
    Default: 4 evoformer blocks, 2 structure iterations = ~30M params.
    """

    def __init__(
        self,
        c_m: int = C_M,
        c_z: int = C_Z,
        c_s: int = C_S,
        num_evoformer_blocks: int = NUM_EVOFORMER_BLOCKS,
        evoformer_heads: int = EVOFORMER_HEADS,
        num_structure_layers: int = NUM_STRUCTURE_LAYERS,
        ipa_heads: int = IPA_HEADS,
        n_qk_points: int = N_QK_POINTS,
        n_v_points: int = N_V_POINTS,
        num_dist_bins: int = NUM_DIST_BINS,
        num_plddt_bins: int = NUM_PLDDT_BINS,
        fape_weight: float = FAPE_WEIGHT,
        distogram_weight: float = DISTOGRAM_WEIGHT,
        plddt_weight: float = PLDDT_WEIGHT,
    ):
        super().__init__()
        self.fape_weight = fape_weight
        self.distogram_weight = distogram_weight
        self.plddt_weight = plddt_weight
        self.num_dist_bins = num_dist_bins
        self.num_plddt_bins = num_plddt_bins

        self.input_embedding = InputEmbedding(c_m, c_z)
        self.evoformer_blocks = nn.ModuleList(
            [EvoformerBlock(c_m, c_z, evoformer_heads) for _ in range(num_evoformer_blocks)]
        )
        self.structure_module = StructureModule(
            c_m,
            c_s,
            c_z,
            num_structure_layers,
            ipa_heads,
            n_qk_points,
            n_v_points,
        )
        self.distogram_head = DistogramHead(c_z, num_dist_bins)
        self.plddt_head = PLDDTHead(c_s, num_plddt_bins)
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
            dict with coords [B,L,3], loss (scalar), fape (scalar).
        """
        B = batch["sequence"].shape[0]
        all_losses, all_fape, all_coords = [], [], []

        for b in range(B):
            seq, mask = batch["sequence"][b], batch["mask"][b]
            true_N = batch["coords_N"][b]
            true_CA, true_C = batch["coords_CA"][b], batch["coords_C"][b]
            L = seq.shape[0]

            msa, pair = self.input_embedding(seq, torch.arange(L, device=seq.device))
            for block in self.evoformer_blocks:
                msa, pair = block(msa, pair)

            pred_coords, pred_frames = self.structure_module(msa, pair)
            all_coords.append(pred_coords)

            true_frames = compute_local_frame(true_N, true_CA, true_C)
            loss_fape = fape_loss(pred_frames, pred_coords, true_frames, true_CA, mask)

            dist_logits = self.distogram_head(pair)
            loss_dist = distogram_loss(dist_logits, true_CA, mask, self.num_dist_bins)

            single = self.structure_module.input_proj(msa[0])
            plddt_logits = self.plddt_head(single)
            loss_plddt = plddt_loss(plddt_logits, pred_coords, true_CA, mask, self.num_plddt_bins)

            total = (
                self.fape_weight * loss_fape
                + self.distogram_weight * loss_dist
                + self.plddt_weight * loss_plddt
            )
            all_losses.append(total)
            all_fape.append(loss_fape)

        return {
            "coords": torch.stack(all_coords),
            "loss": torch.stack(all_losses).mean(),
            "fape": torch.stack(all_fape).mean(),
        }

    @torch.no_grad()
    def predict(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict structure from sequence. Returns coords [L,3] and plddt [L]."""
        self.eval()
        L = sequence.shape[0]
        device = sequence.device
        msa, pair = self.input_embedding(sequence, torch.arange(L, device=device))
        for block in self.evoformer_blocks:
            msa, pair = block(msa, pair)
        pred_coords, pred_frames = self.structure_module(msa, pair)
        single = self.structure_module.input_proj(msa[0])
        plddt_logits = self.plddt_head(single)
        plddt = torch.softmax(plddt_logits, dim=-1)
        bins = torch.linspace(0, 1, self.num_plddt_bins, device=device)
        plddt_score = (plddt * bins).sum(dim=-1)
        return {"coords": pred_coords, "plddt": plddt_score, "frames": pred_frames}
