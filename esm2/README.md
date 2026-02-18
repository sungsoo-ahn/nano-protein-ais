# nano-esm2

Minimal implementation of **ESM2** (Evolutionary Scale Modeling) — a BERT-style masked language model for protein sequences. ~8M parameters, 288 lines of model code.

Inspired by [nanochat](https://github.com/karpathy/nanochat): two files, no config objects, no framework abstractions.

## What it does

**Problem:** Learn the "language" of proteins from raw amino acid sequences.

**How:** Mask 15% of amino acids in a protein sequence, then train a Transformer to predict the masked residues from context (just like BERT for natural language).

**Why it matters:** Protein language models capture evolutionary information — which amino acids can substitute for each other, which positions are conserved, how local sequence motifs work. The learned embeddings transfer to downstream tasks like structure prediction, function annotation, and variant effect prediction.

```
Input:   M A [MASK] K T [MASK] V L ...
Output:  - -   E   - -   G   - - ...  (predict masked amino acids)
```

## Architecture

```
Token Embedding (25 vocab → 320 dim, no positional embedding)
       │
       ▼
┌─────────────────────────────────────┐
│  Transformer Block (x6)            │
│  ┌───────────────────────────────┐ │
│  │ LayerNorm                     │ │
│  │ Multi-Head Self-Attention     │ │
│  │   (20 heads, RoPE positions)  │ │
│  │ + Residual                    │ │
│  ├───────────────────────────────┤ │
│  │ LayerNorm                     │ │
│  │ SwiGLU FFN (320 → 1280 → 320)│ │
│  │ + Residual                    │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
       │
       ▼
  LayerNorm → Linear (320 → 25)  →  logits
```

**Key design choices:**
- **RoPE** (Rotary Position Embeddings) instead of learned position embeddings — encodes relative position by rotating Q/K vectors
- **SwiGLU** activation instead of standard FFN — `W3 * (SiLU(W1*x) * W2*x)` for better gradient flow
- **Pre-LayerNorm** — normalize before attention/FFN, not after (more stable training)
- **BERT-style masking** — 80% [MASK], 10% random token, 10% keep original

## Quick start

```bash
# Install dependencies (just PyTorch + numpy)
pip install torch numpy

# Get training data — any protein FASTA file works
# Option A: Download Swiss-Prot (~85MB, 570K sequences)
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
mkdir -p data/sequences
mv uniprot_sprot.fasta data/sequences/

# Option B: Use any FASTA file you have
# python train.py --data_path your_sequences.fasta

# Train
python train.py --data_path data/sequences/uniprot_sprot.fasta
```

## Files

```
esm2/
├── model.py   (288 lines)  # Full ESM2 architecture
│   ├── Vocabulary: 20 AAs + PAD/MASK/CLS/EOS/UNK = 25 tokens
│   ├── SwiGLU FFN
│   ├── Rotary Position Embeddings (RoPE)
│   ├── Multi-Head Self-Attention with RoPE
│   ├── TransformerBlock (Pre-LN)
│   ├── ESM2 model: forward() + extract_embeddings()
│   ├── mask_tokens(): BERT-style masking logic
│   └── compute_mlm_loss(): cross-entropy + accuracy + perplexity
│
├── train.py   (192 lines)  # Self-contained training script
│   ├── FASTADataset: reads any .fasta file
│   ├── Tokenizer: AA sequence → integer tokens with CLS/EOS
│   └── Training loop with cosine LR schedule + warmup
│
└── README.md  (this file)
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layers | 6 | Transformer blocks |
| Hidden dim | 320 | Token representation size |
| Attention heads | 20 | 16 dim per head |
| FFN dim | 1280 | 4x hidden (SwiGLU) |
| Vocab size | 25 | 20 AAs + 5 special tokens |
| Mask fraction | 15% | Standard BERT masking |
| Max seq length | 512 | + CLS + EOS = 514 tokens |
| Batch size | 32 | |
| Learning rate | 4e-4 | AdamW with weight decay 0.01 |
| Parameters | ~8M | |

## Training curve

Expected output when training on ~50K protein sequences:

```
step=50   epoch=1  loss=3.2046 acc=0.061 ppl=24.6 lr=4.00e-05
step=100  epoch=1  loss=3.1498 acc=0.073 ppl=23.3 lr=8.00e-05
step=200  epoch=1  loss=3.0752 acc=0.094 ppl=21.6 lr=1.60e-04
step=500  epoch=2  loss=2.9220 acc=0.128 ppl=18.6 lr=3.85e-04
step=1000 epoch=4  loss=2.7814 acc=0.162 ppl=16.2 lr=3.78e-04
step=2000 epoch=8  loss=2.6390 acc=0.197 ppl=14.0 lr=3.46e-04
step=3000 epoch=12 loss=2.5641 acc=0.218 ppl=13.0 lr=2.96e-04
```

The loss starts around 3.2 (random guessing over 20 amino acids = ln(20) = 3.0, slightly higher due to masking strategy) and decreases as the model learns amino acid co-occurrence patterns.

## Results

On a nano-scale training run (50K sequences, 20 epochs):

| Metric | Start | End |
|--------|-------|-----|
| MLM Loss | 3.20 | 2.56 |
| Masked Accuracy | 6% | 22% |
| Perplexity | 24.6 | 13.0 |

**What this means:**
- **6% accuracy** = random guessing (1/20 amino acids)
- **22% accuracy** = model learned which amino acids are common at each position type (hydrophobic cores, charged surfaces, etc.)
- **Perplexity 13** = on average, the model is "confused" between ~13 amino acids per position (down from ~25 at random)

With more data and longer training, ESM2 reaches >50% accuracy and perplexity <5.

## Key concepts for students

### 1. Protein sequences as language
Proteins are chains of 20 amino acids. Like words in a sentence, amino acids have "grammar" — certain combinations are allowed (form stable structures) and others are not. MLM learns this grammar.

### 2. Masked Language Modeling (MLM)
Same idea as BERT: randomly hide 15% of tokens, train the model to fill in the blanks. The masking strategy (80/10/10) prevents the model from only learning to recognize [MASK] tokens.

### 3. Rotary Position Embeddings (RoPE)
Instead of adding a position vector to each token, RoPE rotates the query and key vectors based on position. This naturally encodes *relative* position: the dot product between positions i and j depends only on (i-j), not on absolute position.

```python
# Core idea: rotate Q/K by position-dependent angle
q_rotated = q * cos(pos * freq) + rotate_half(q) * sin(pos * freq)
```

### 4. SwiGLU
A gated feed-forward network: `output = W3 * (SiLU(W1*x) * W2*x)`. The gating mechanism (multiply by W2*x) allows the network to selectively pass information, improving over standard ReLU FFNs.

### 5. Vocabulary design
20 standard amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y) plus 5 special tokens:
- **PAD** (20): padding for batching variable-length sequences
- **MASK** (21): MLM mask token
- **CLS** (22): classification token (start of sequence)
- **EOS** (23): end of sequence
- **UNK** (24): unknown amino acid

## References

- Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model" (2023). [Science](https://doi.org/10.1126/science.ade2574)
- Rives et al., "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" (2021). [PNAS](https://doi.org/10.1073/pnas.2016239118)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). [arXiv](https://arxiv.org/abs/2104.09864)
- Shazeer, "GLU Variants Improve Transformer" (2020). [arXiv](https://arxiv.org/abs/2002.05202)
