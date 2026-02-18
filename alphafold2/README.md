# nano-alphafold2

Minimal implementation of **AlphaFold2** — structure prediction from amino acid sequence using Pairformer + SE(3) frame diffusion. ~30M parameters, ~850 lines of model code.

Inspired by [nanochat](https://github.com/karpathy/nanochat): two files, no config objects, no framework abstractions.

## What it does

**Problem:** Given a protein's amino acid sequence, predict the 3D structure of its backbone.

**How:** Encode the sequence into single and pair representations using a Pairformer trunk, then use SE(3) diffusion to iteratively denoise random frames into the predicted backbone structure.

**Why it matters:** Protein structure prediction is one of the most important problems in biology. Knowing a protein's 3D shape reveals how it functions, how it interacts with drugs, and how mutations cause disease. AlphaFold2 solved this problem at scale.

```
Input:   "MAKTEVL..."  (amino acid sequence)
Output:   3D backbone coordinates (N, CA, C per residue)
          + pLDDT confidence scores
```

## Architecture

```
Amino acid sequence [B, L]
       │
       ▼
  Input Embedding
  ├─→ Single rep [B, L, 256]     (one-hot → learned projection)
  └─→ Pair rep [B, L, L, 64]     (outer sum + relative position encoding)
       │
       ▼
┌─────────────────────────────────────────┐
│  Pairformer Block (x4)                 │
│  ┌───────────────────────────────────┐ │
│  │ Pair Stack:                       │ │
│  │   Triangular Multiplicative (out) │ │
│  │   Triangular Multiplicative (in)  │ │
│  │   Triangular Attention (start)    │ │
│  │   Triangular Attention (end)      │ │
│  │   Pair Transition                 │ │
│  ├───────────────────────────────────┤ │
│  │ Single Stack:                     │ │
│  │   Attention with Pair Bias        │ │
│  │   Single Transition               │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
       │
       ├─→ pLDDT head → confidence [B, L]
       │
       ▼
┌─────────────────────────────────────────┐
│  SE(3) Frame Diffusion                 │
│                                         │
│  Training:                              │
│    true_frames → add noise(t) → noisy  │
│    denoise(noisy, t, single, pair)     │
│    → predicted frames → FAPE loss      │
│                                         │
│  Inference (reverse diffusion):         │
│    noise(t=1) → denoise → ... → t=0   │
│    100 denoising steps                  │
└─────────────────────────────────────────┘
       │
       ▼
  backbone_from_frames() → N, CA, C coordinates
```

**Key design choices:**
- **Pairformer** (not Evoformer) — drops the MSA axis entirely, operates only on single + pair representations. Simpler and works for single-sequence prediction.
- **Triangular updates** — pair features propagate information along edges of a triangle: if (i,j) and (j,k) are related, then (i,k) should be too. This captures the constraint that protein structures are spatially consistent.
- **SE(3) diffusion** — predicts backbone frames (rotation + translation per residue) using product diffusion: SO(3) noise on rotations + R(3) noise on translations.
- **FAPE loss** — Frame Aligned Point Error: measures structural error in each residue's local coordinate frame, making the loss invariant to global rigid-body motion.

## Quick start

```bash
# Install dependencies
pip install torch numpy

# Train (auto-downloads 8 PDB files from RCSB, ~2MB total)
python train.py

# Predict a structure (after training)
python -c "
import torch
from model import AlphaFold2, AA_TO_IDX
model = AlphaFold2()
model.load_state_dict(torch.load('outputs/alphafold2/final_model.pt')['model_state_dict'])
seq = torch.tensor([AA_TO_IDX.get(aa, 0) for aa in 'TTCCPSIVARSNFNVCRLPGT'])
result = model.predict(seq)
print(f'CA coords: {result[\"coords_CA\"].shape}')
print(f'pLDDT: {result[\"plddt\"].mean():.3f}')
"
```

## Files

```
alphafold2/
├── model.py   (~850 lines)  # Full AlphaFold2 architecture
│   ├── Geometry: axis_angle_to_rotation_matrix (Rodrigues formula)
│   ├── RigidTransform: SE(3) frames with compose/invert/apply
│   ├── compute_local_frame: N/CA/C → rotation + translation
│   ├── backbone_from_frames: SE(3) → ideal N/CA/C coordinates
│   ├── Pairformer:
│   │   ├── TriangularMultiplicativeUpdate (outgoing + incoming)
│   │   ├── TriangularAttention (starting + ending)
│   │   ├── AttentionWithPairBias
│   │   └── PairformerBlock
│   ├── SE(3) Diffusion:
│   │   ├── SO3Diffuser: IGSO3 noise on rotations
│   │   ├── R3Diffuser: cosine DDPM on translations
│   │   └── SE3Diffusion: product SO(3) x R(3)
│   ├── DiffusionModule: denoiser with AdaLN + pair bias
│   ├── Heads: PLDDTHead
│   ├── Losses: fape_loss, frame_loss, plddt_loss
│   └── AlphaFold2: forward() + predict()
│
├── train.py   (~255 lines)  # Self-contained training script
│   ├── PDB parser (inlined)
│   ├── PDB downloader (auto-downloads from RCSB)
│   ├── PDBDataset with padding/batching
│   └── Training loop with cosine LR schedule
│
└── README.md  (this file)
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Single dim (C_S) | 256 | Per-residue representation |
| Pair dim (C_Z) | 64 | Per-pair representation |
| Denoiser dim (C_ATOM) | 256 | Diffusion network hidden size |
| Pairformer blocks | 4 | Trunk depth |
| Pairformer heads | 4 | |
| Denoise blocks | 4 | Denoiser depth |
| Denoise heads | 8 | |
| Diffusion steps | 100 | Reverse diffusion at inference |
| SO(3) sigma_max | 1.5 | Max rotation noise |
| R(3) sigma_max | 10.0 | Max translation noise |
| Batch size | 1 | (pair features are O(L^2)) |
| Learning rate | 3e-3 | AdamW |
| Parameters | ~30M | |

## Training curve

Expected output when training on 8 small proteins:

```
step=50   epoch=6   loss=15.8432 fape=12.421 trans=8.123 rot=2.461
step=100  epoch=12  loss=12.1847 fape=10.224 trans=5.841 rot=1.932
step=200  epoch=25  loss=10.3524 fape=8.872  trans=4.521 rot=1.643
step=500  epoch=62  loss=9.1253  fape=8.103  trans=3.672 rot=1.421
step=900  epoch=112 loss=8.7563  fape=7.799  trans=3.421 rot=1.312
```

The loss has multiple components. The dominant term (FAPE) measures structural accuracy in local frames. Translation and rotation losses provide more direct gradients for training stability.

## Results

### Training set (8 small proteins)

| Metric | Start | End |
|--------|-------|-----|
| Total loss | 22.6 | 8.8 |
| FAPE loss | 14.2 | 7.8 |
| Translation loss | 8.1 | 3.4 |
| Rotation loss | 2.5 | 1.3 |

### Evaluation (structure prediction)

After training on 8 proteins, predicting structures via 100-step reverse diffusion:

| Protein | Length | CA-RMSD | mean pLDDT |
|---------|--------|---------|------------|
| 1CRN (crambin) | 46 | 14.4 A | 0.001 |
| 1UBQ (ubiquitin) | 76 | 35.6 A | 0.001 |
| 2GB1 (protein G) | 56 | 29.2 A | 0.000 |

**Interpretation:** With only 8 training proteins and ~30M parameters, the model cannot generalize to produce accurate structures. The real AlphaFold2 trains on ~200K structures. This nano version demonstrates the *architecture* and training procedure, not competitive accuracy.

## Key concepts for students

### 1. SE(3) — the group of rigid motions
A protein backbone can be described as one rigid frame (rotation + translation) per residue. SE(3) = SO(3) x R(3) is the space of 3D rotations (3x3 orthogonal matrices) combined with translations (3D vectors). Operating in frame space is natural for proteins.

### 2. Pairformer (simplified Evoformer)
The original AlphaFold2 uses Evoformer, which operates on MSA (multiple sequence alignment) + pair representations. This nano version drops the MSA axis, keeping only:
- **Single representation** [L, 256]: per-residue features
- **Pair representation** [L, L, 64]: per-pair features

The pair stack uses triangular operations to enforce geometric consistency.

### 3. Triangular multiplicative updates
The key insight: in a protein, if residues i and j are close, and j and k are close, then the relationship between i and k is constrained. The triangular update computes:
```
pair[i,j] += sum_k (pair[i,k] * pair[j,k])   # outgoing
pair[i,j] += sum_k (pair[k,i] * pair[k,j])   # incoming
```
This propagates spatial constraints through the pair representation.

### 4. SE(3) diffusion
Training: corrupt true frames with noise (rotations via IGSO3, translations via Gaussian), then train a denoiser to predict the clean frames.

Inference: start from pure noise and iteratively denoise over 100 steps:
```
t=1.0 (pure noise) → t=0.99 → ... → t=0.01 → t=0.0 (predicted structure)
```

### 5. IGSO3 — Isotropic Gaussian on SO(3)
To add noise to rotations, we can't just add Gaussian noise (that would break orthogonality). Instead, IGSO3 samples a random rotation axis and a Gaussian-distributed angle, then converts to a rotation matrix via the Rodrigues formula.

### 6. FAPE loss — Frame Aligned Point Error
Standard RMSD depends on global alignment. FAPE measures error in each residue's local frame:
```
For each frame f:
    transform all CA positions into f's coordinate system
    compare predicted vs. true local positions
```
This is invariant to global rigid-body motion and provides richer gradients.

### 7. backbone_from_frames()
Given SE(3) frames, reconstruct N, CA, C atom coordinates using ideal bond geometry:
- CA is at the frame's translation
- C is along the local x-axis at 1.523 A
- N is in the local x-y plane at 1.458 A with ideal bond angle 111 degrees

## References

- Jumper et al., "Highly accurate protein structure prediction with AlphaFold" (2021). [Nature](https://doi.org/10.1038/s41586-021-03819-2)
- Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3" (2024). [Nature](https://doi.org/10.1038/s41586-024-07487-w)
- Yim et al., "SE(3) diffusion model with application to protein backbone generation" (2023). [ICML](https://arxiv.org/abs/2302.02277)
