"""ESM2 protein language model — minimal, self-contained implementation.

BERT-style masked language model for proteins using:
- Token embeddings (no learned positional embeddings, uses RoPE instead)
- 6 Transformer layers with SwiGLU FFN
- Pre-LayerNorm architecture
- Rotary positional embeddings (RoPE)

Total parameters: ~8M for default configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Amino acid vocabulary
# ---------------------------------------------------------------------------

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
NUM_AMINO_ACIDS = 20
PAD_IDX = 20
MASK_IDX = 21
CLS_IDX = 22
EOS_IDX = 23
UNK_IDX = 24
VOCAB_SIZE = 25

AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 6
HIDDEN_DIM = 320
NUM_HEADS = 20
FFN_DIM = 1280  # 4x hidden
MAX_SEQ_LEN = 514  # 512 + CLS + EOS
DROPOUT = 0.0
MASK_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {PAD_IDX, CLS_IDX, EOS_IDX, UNK_IDX}


def mask_tokens(
    tokens: torch.Tensor,
    mask_fraction: float = MASK_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking: 15% selected, of which 80% [MASK], 10% random, 10% keep.

    Returns (masked_tokens, labels) where labels=-100 at non-masked positions.
    """
    masked_tokens = tokens.clone()
    labels = torch.full_like(tokens, -100)

    # Positions that can be masked (non-special tokens)
    maskable = torch.ones_like(tokens, dtype=torch.bool)
    for sid in SPECIAL_TOKENS:
        maskable &= tokens != sid

    # Select positions
    selected = maskable & (torch.rand_like(tokens, dtype=torch.float) < mask_fraction)

    # Strategy: 80% MASK, 10% random, 10% keep
    strategy = torch.rand_like(tokens, dtype=torch.float)
    masked_tokens[selected & (strategy < 0.8)] = MASK_IDX
    random_mask = selected & (strategy >= 0.8) & (strategy < 0.9)
    rand_tokens = torch.randint(0, NUM_AMINO_ACIDS, tokens.shape, device=tokens.device)
    masked_tokens[random_mask] = rand_tokens[random_mask]

    labels[selected] = tokens[selected]
    return masked_tokens, labels


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU(x) = W3 * (SiLU(W1*x) * W2*x)"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: encodes position by rotating query/key vectors."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        return self.out_proj(out), attn_weights


class TransformerBlock(nn.Module):
    """Pre-LayerNorm: x = x + MHSA(LN(x)), x = x + SwiGLU(LN(x))"""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(self.ln1(x), attention_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ESM2(nn.Module):
    """ESM2 protein language model.

    BERT-style MLM trained on protein sequences.
    Default: 6 layers, 320 hidden, 20 heads = ~8M params.
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = FFN_DIM,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: [B, L] token IDs.
            attention_mask: [B, L] True for valid positions.

        Returns:
            dict with logits [B, L, vocab_size], embeddings [B, L, hidden_dim],
            attention_weights [num_layers, B, num_heads, L, L].
        """
        x = self.token_embedding(tokens)
        all_attn = []
        for layer in self.layers:
            x, attn_w = layer(x, attention_mask)
            all_attn.append(attn_w)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return {
            "logits": logits,
            "embeddings": x,
            "attention_weights": torch.stack(all_attn),
        }

    @torch.no_grad()
    def extract_embeddings(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract embeddings without gradients (for downstream tasks)."""
        return self.forward(tokens, attention_mask)["embeddings"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_mlm_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy on masked positions + accuracy/perplexity metrics."""
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    mask = labels != -100
    accuracy = ((logits.argmax(dim=-1) == labels) & mask).sum().float() / mask.sum().float()
    perplexity = torch.exp(loss)
    return loss, {"masked_accuracy": accuracy.item(), "perplexity": perplexity.item()}
