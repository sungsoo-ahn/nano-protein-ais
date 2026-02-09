"""ProteinMPNN — inverse folding: backbone structure → amino acid sequence.

Self-contained implementation with k-NN graph, RBF encoding, MPNN encoder,
and autoregressive decoder. No shared code with other models.

Total parameters: ~3.5M with default config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 192
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
K_NEIGHBORS = 30
NUM_RBF = 16
DROPOUT = 0.1
VOCAB_SIZE = 20  # 20 standard amino acids

# ---------------------------------------------------------------------------
# Geometry (inlined — no shared module)
# ---------------------------------------------------------------------------


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances. coords: [..., N, 3] -> [..., N, N]."""
    diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)
    return diff.norm(dim=-1)


def dihedral_angle(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> torch.Tensor:
    """Dihedral angle between four points. Returns [...] in radians."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1 = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-8)
    b2_norm = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    m1 = torch.cross(n1, b2_norm, dim=-1)
    return torch.atan2((m1 * n2).sum(dim=-1), (n1 * n2).sum(dim=-1))


def compute_local_frame(
    N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local frames. Returns (rots [L,3,3], trans [L,3])."""
    x = C - CA
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    v = N - CA
    z = torch.cross(x, v, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)
    return R, CA


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_knn_graph(
    ca_coords: torch.Tensor, k: int, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN graph from CA coords. Returns (edge_index [2,E], edge_dist [E])."""
    L = ca_coords.shape[0]
    dist = pairwise_distances(ca_coords)

    if mask is not None:
        invalid = ~mask.unsqueeze(0) | ~mask.unsqueeze(1)
        dist = dist.masked_fill(invalid, float("inf"))
    dist = dist.fill_diagonal_(float("inf"))

    k_actual = min(k, L - 1)
    if k_actual <= 0:
        empty = torch.zeros(2, 0, dtype=torch.long, device=ca_coords.device)
        return empty, torch.zeros(0, device=ca_coords.device)

    knn_dist, knn_idx = torch.topk(dist, k_actual, dim=1, largest=False)
    src = torch.arange(L, device=ca_coords.device).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_dist = knn_dist.reshape(-1)

    valid = edge_dist < float("inf")
    return torch.stack([src[valid], dst[valid]]), edge_dist[valid]


def rbf_encode(
    distances: torch.Tensor,
    num_rbf: int = NUM_RBF,
    max_dist: float = 20.0,
) -> torch.Tensor:
    """Gaussian RBF encoding. distances: [E] -> [E, num_rbf]."""
    centers = torch.linspace(0, max_dist, num_rbf, device=distances.device)
    gamma = 1.0 / (max_dist / num_rbf) ** 2
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-gamma * diff**2)


def compute_edge_features(
    coords_N: torch.Tensor,
    coords_CA: torch.Tensor,
    coords_C: torch.Tensor,
    coords_O: torch.Tensor,
    edge_index: torch.Tensor,
    num_rbf: int = NUM_RBF,
) -> torch.Tensor:
    """Edge features: RBF distances (16 atom pairs) + seq separation (65) + direction (3)."""
    src, dst = edge_index[0], edge_index[1]
    bb = torch.stack([coords_N, coords_CA, coords_C, coords_O], dim=1)  # [L, 4, 3]

    # RBF-encoded distances for all 4x4 backbone atom pairs
    rbf_feats = []
    for ai in range(4):
        for aj in range(4):
            d = (bb[src, ai] - bb[dst, aj]).norm(dim=-1)
            rbf_feats.append(rbf_encode(d, num_rbf))
    rbf_feats = torch.cat(rbf_feats, dim=-1)  # [E, 16*num_rbf]

    # Sequence separation one-hot
    seq_sep = (dst - src).clamp(-32, 32).float()
    seq_sep_idx = (seq_sep + 32).long()
    seq_sep_feat = F.one_hot(seq_sep_idx, num_classes=65).float()

    # Direction in local frame
    rots, trans = compute_local_frame(coords_N, coords_CA, coords_C)
    R_src = rots[src]
    direction = coords_CA[dst] - trans[src]
    direction_local = torch.einsum("eij,ej->ei", R_src.transpose(-1, -2), direction)
    direction_local = direction_local / (direction_local.norm(dim=-1, keepdim=True) + 1e-8)

    return torch.cat([rbf_feats, seq_sep_feat, direction_local], dim=-1)


def compute_node_features(
    coords_N: torch.Tensor, coords_CA: torch.Tensor, coords_C: torch.Tensor
) -> torch.Tensor:
    """Node features: dihedral sin/cos (6) + flattened rotation matrix (9) = 15."""
    L = coords_CA.shape[0]
    device = coords_CA.device

    phi = torch.zeros(L, device=device)
    psi = torch.zeros(L, device=device)
    omega = torch.zeros(L, device=device)

    if L > 1:
        phi[1:] = dihedral_angle(coords_C[:-1], coords_N[1:], coords_CA[1:], coords_C[1:])
        psi[:-1] = dihedral_angle(coords_N[:-1], coords_CA[:-1], coords_C[:-1], coords_N[1:])
        omega[:-1] = dihedral_angle(coords_CA[:-1], coords_C[:-1], coords_N[1:], coords_CA[1:])

    dihedral_feats = torch.stack(
        [phi.sin(), phi.cos(), psi.sin(), psi.cos(), omega.sin(), omega.cos()], dim=-1
    )
    rots, _ = compute_local_frame(coords_N, coords_CA, coords_C)
    frame_feats = rots.reshape(L, 9)
    return torch.cat([dihedral_feats, frame_feats], dim=-1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        L = h.shape[0]
        messages = self.message_mlp(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        agg = torch.zeros(L, self.hidden_dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, messages)
        return self.norm(h + self.update_mlp(torch.cat([h, agg], dim=-1)))


class StructureEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embedding(node_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def create_decoding_mask(decoding_order: torch.Tensor) -> torch.Tensor:
    """Causal mask for random-order decoding. mask[i,j]=True if j decoded before/at i."""
    L = decoding_order.shape[0]
    rank = torch.zeros(L, dtype=torch.long, device=decoding_order.device)
    for step, pos in enumerate(decoding_order):
        rank[pos] = step
    # mask[i, j] = True if rank[j] <= rank[i] (j decoded before or at i)
    return rank.unsqueeze(0) <= rank.unsqueeze(1)  # [L, L]


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=False,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        L = h.shape[0]
        h_t = h.unsqueeze(1)
        attn_mask = torch.zeros(L, L, device=h.device, dtype=h.dtype)
        attn_mask = attn_mask.masked_fill(~causal_mask, float("-inf"))

        # Fix fully masked rows
        all_masked = torch.isinf(attn_mask).all(dim=1)
        if all_masked.any():
            for i in torch.where(all_masked)[0]:
                attn_mask[i, :] = float("-inf")
                attn_mask[i, i] = 0.0

        h_attn, _ = self.self_attn(h_t, h_t, h_t, attn_mask=attn_mask, need_weights=False)
        h = self.norm1(h + h_attn.squeeze(1))

        enc_t = encoder_output.unsqueeze(1)
        h_cross, _ = self.cross_attn(h.unsqueeze(1), enc_t, enc_t)
        h = self.norm2(h + h_cross.squeeze(1))

        h = self.norm3(h + self.ffn(h))
        return h


class AutoregressiveDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, vocab_size: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        decoding_order: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.token_embedding(tokens)
        causal_mask = create_decoding_mask(decoding_order) & mask.unsqueeze(0)
        for layer in self.layers:
            h = layer(h, encoder_output, causal_mask, mask)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ProteinMPNN(nn.Module):
    """ProteinMPNN: predicts amino acid sequence from backbone structure.

    Default: hidden_dim=192, 3 encoder + 3 decoder layers = ~3.5M params.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        num_decoder_layers: int = NUM_DECODER_LAYERS,
        k_neighbors: int = K_NEIGHBORS,
        num_rbf: int = NUM_RBF,
        dropout: float = DROPOUT,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf
        self.vocab_size = vocab_size

        edge_dim = 16 * num_rbf + 65 + 3
        node_dim = 15

        self.encoder = StructureEncoder(node_dim, edge_dim, hidden_dim, num_encoder_layers, dropout)
        self.decoder = AutoregressiveDecoder(hidden_dim, num_decoder_layers, vocab_size, dropout)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            batch: dict with coords_N/CA/C/O [B,L,3], sequence [B,L], mask [B,L].

        Returns:
            dict with logits [B,L,vocab], loss (scalar), sequence_recovery (scalar).
        """
        B, L = batch["sequence"].shape
        device = batch["coords_CA"].device
        all_logits, all_losses, all_recoveries = [], [], []

        for b in range(B):
            coords_N = batch["coords_N"][b]
            coords_CA = batch["coords_CA"][b]
            coords_C = batch["coords_C"][b]
            coords_O = batch["coords_O"][b]
            seq = batch["sequence"][b]
            mask = batch["mask"][b]

            edge_index, _ = build_knn_graph(coords_CA, k=self.k_neighbors, mask=mask)
            node_feats = compute_node_features(coords_N, coords_CA, coords_C)
            edge_feats = compute_edge_features(
                coords_N,
                coords_CA,
                coords_C,
                coords_O,
                edge_index,
                self.num_rbf,
            )

            encoder_out = self.encoder(node_feats, edge_index, edge_feats)
            decoding_order = torch.randperm(L, device=device)
            logits = self.decoder(seq, encoder_out, decoding_order, mask)

            loss = F.cross_entropy(logits[mask], seq[mask], reduction="mean")
            pred = logits.argmax(dim=-1)
            recovery = (pred[mask] == seq[mask]).float().mean()

            all_logits.append(logits)
            all_losses.append(loss)
            all_recoveries.append(recovery)

        return {
            "logits": torch.stack(all_logits),
            "loss": torch.stack(all_losses).mean(),
            "sequence_recovery": torch.stack(all_recoveries).mean(),
        }

    @torch.no_grad()
    def design(
        self,
        coords_N: torch.Tensor,
        coords_CA: torch.Tensor,
        coords_C: torch.Tensor,
        coords_O: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 0.1,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Design sequences for a given backbone (inference).

        Returns: [num_samples, L] sampled amino acid indices.
        """
        self.eval()
        L = coords_CA.shape[0]
        device = coords_CA.device

        edge_index, _ = build_knn_graph(coords_CA, k=self.k_neighbors, mask=mask)
        node_feats = compute_node_features(coords_N, coords_CA, coords_C)
        edge_feats = compute_edge_features(
            coords_N,
            coords_CA,
            coords_C,
            coords_O,
            edge_index,
            self.num_rbf,
        )
        encoder_out = self.encoder(node_feats, edge_index, edge_feats)

        sequences = []
        for _ in range(num_samples):
            order = torch.randperm(L, device=device)
            seq = torch.randint(0, self.vocab_size, (L,), device=device)
            logits = self.decoder(seq, encoder_out, order, mask)
            probs = F.softmax(logits / temperature, dim=-1)
            for pos in range(L):
                if mask[pos]:
                    seq[pos] = torch.multinomial(probs[pos], 1).item()
            sequences.append(seq)
        return torch.stack(sequences)
