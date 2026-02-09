"""Tests for ESM2 protein language model."""

import pytest
import torch
import torch.nn.functional as F

from esm2.model import (
    CLS_IDX,
    EOS_IDX,
    ESM2,
    MASK_IDX,
    NUM_AMINO_ACIDS,
    PAD_IDX,
    UNK_IDX,
    RotaryPositionalEmbedding,
    SwiGLU,
    apply_rotary_pos_emb,
    mask_tokens,
)


@pytest.fixture
def model():
    return ESM2(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, dropout=0.0)


def test_swiglu_shape():
    swiglu = SwiGLU(64, 128)
    x = torch.randn(2, 10, 64)
    assert swiglu(x).shape == (2, 10, 64)


def test_rope_relative_position():
    head_dim = 16
    seq_len = 16
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=seq_len)
    q = torch.randn(1, 4, seq_len, head_dim)
    k = torch.randn(1, 4, seq_len, head_dim)
    cos, sin = rope(seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
    adjacent = [scores[0, 0, i, i + 1].item() for i in range(seq_len - 1)]
    assert torch.tensor(adjacent).std().item() < 6.0


def test_masking_preserves_special_tokens():
    tokens = torch.randint(0, NUM_AMINO_ACIDS, (4, 20))
    tokens[:, 0] = CLS_IDX
    tokens[:, -1] = EOS_IDX
    tokens[:, 5] = PAD_IDX
    tokens[:, 10] = UNK_IDX
    masked_tokens, labels = mask_tokens(tokens)

    assert (masked_tokens[:, 0] == CLS_IDX).all()
    assert (masked_tokens[:, -1] == EOS_IDX).all()
    assert (masked_tokens[:, 5] == PAD_IDX).all()
    assert (masked_tokens[:, 10] == UNK_IDX).all()
    assert (labels[:, 0] == -100).all()
    assert (labels[:, -1] == -100).all()


def test_masking_ratios_approximate():
    tokens = torch.randint(0, NUM_AMINO_ACIDS, (100, 50))
    masked_tokens, labels = mask_tokens(tokens, mask_fraction=0.15)

    actual = (labels != -100).sum().item() / labels.numel()
    assert abs(actual - 0.15) < 0.15 * 0.2

    masked_mask = labels != -100
    mask_ratio = (masked_tokens[masked_mask] == MASK_IDX).sum().item() / masked_mask.sum().item()
    assert abs(mask_ratio - 0.8) < 0.1


def test_model_forward_shape(model):
    tokens = torch.randint(0, 25, (2, 50))
    mask = torch.ones(2, 50, dtype=torch.bool)
    outputs = model(tokens, mask)

    assert outputs["logits"].shape == (2, 50, 25)
    assert outputs["embeddings"].shape == (2, 50, 64)
    assert outputs["attention_weights"].shape == (2, 2, 4, 50, 50)


def test_attention_weights_sum_to_one(model):
    tokens = torch.randint(0, 25, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)
    attn = model(tokens, mask)["attention_weights"]
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


def test_model_param_count():
    small = ESM2(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128)
    assert 10_000 < small.count_parameters() < 1_000_000

    full = ESM2()  # default ~8M
    assert 5_000_000 < full.count_parameters() < 10_000_000


def test_embedding_extraction(model):
    tokens = torch.randint(0, 25, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)
    emb = model.extract_embeddings(tokens, mask)
    assert emb.shape == (2, 20, 64)
    assert not emb.requires_grad


def test_loss_is_finite(model):
    tokens = torch.randint(0, 25, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)
    masked_tokens, labels = mask_tokens(tokens)
    logits = model(masked_tokens, mask)["logits"]
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.view(-1, 25), labels.view(-1), ignore_index=-100)
    assert torch.isfinite(loss)


def test_attention_mask_effect(model):
    tokens = torch.randint(0, 25, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[:, 15:] = False
    attn = model(tokens, mask)["attention_weights"]
    padded_attn = attn[:, :, :, :, 15:].sum()
    assert padded_attn < 0.01 * attn.numel()
