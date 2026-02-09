# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Minimal, self-contained implementations of four protein AI models, inspired by [nanochat](https://github.com/karpathy/nanochat). Each model folder is a complete, readable implementation that a student can understand by reading a few files. The reference (full) implementation lives in `../ai-protein-impl`.

### The Four Models

- **alphafold2/** — Structure prediction: amino acid sequence → 3D backbone coordinates. Core ideas: Evoformer (MSA+pair representation), Invariant Point Attention, iterative frame refinement.
- **esm2/** — Protein language model: masked language modeling on sequences (BERT-style). Core ideas: RoPE, SwiGLU, pre-LayerNorm transformer.
- **rfdiffusion/** — De novo backbone generation via SE(3) diffusion. Core ideas: SO(3)+R(3) noise schedules, denoising network with IPA, self-conditioning.
- **proteinmpnn/** — Inverse folding: backbone structure → amino acid sequence. Core ideas: k-NN graph on CA atoms, MPNN encoder, autoregressive decoder.

## Development Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Lint & format
ruff check .
ruff format .

# Tests
pytest tests/ -v
pytest tests/test_esm2.py -v          # single test file
pytest tests/ -k "test_forward" -v    # single test by name
```

## Design Philosophy (nanochat-style)

- **Flat, not deep.** Each model folder is self-contained. No shared base classes, no framework abstractions, no config registries. Duplicate a few lines rather than adding an import chain.
- **Each model = `model.py` + `train.py`.** `model.py` has the full architecture. `train.py` is a runnable script. Additional files only when a module (e.g., evoformer, diffusion) is genuinely large enough to warrant separation.
- **No configuration objects.** Use simple constants or dataclass defaults at the top of each file. A student should see the hyperparameters right where they're used.
- **Readable over reusable.** Code is meant to be forked and modified, not imported as a library. Optimize for someone reading top-to-bottom.
- **Small parameter counts.** Models should train on a single GPU in hours (AlphaFold2 ~30M, ESM2 ~8M, RFDiffusion ~35M, ProteinMPNN ~3.5M).

## Key Protein AI Concepts Used Across Models

- **ProteinStructure**: Backbone atoms (N, CA, C, O) per residue, plus sequence and validity mask.
- **Rigid frames / SE(3)**: Rotation (3×3) + translation (3,) per residue. Used in AlphaFold2 (structure module), RFDiffusion (diffusion), ProteinMPNN (features).
- **Amino acid vocabulary**: 20 standard AAs + special tokens (PAD=20, MASK=21, CLS=22, EOS=23, UNK=24), vocab_size=25.
- **Geometry primitives**: `axis_angle_to_rotation_matrix` (Rodrigues), quaternion conversions, `compute_local_frame` from N/CA/C, dihedral angles, pairwise distances.

## Architecture (3,410 lines across 13 files)

```
esm2/
  model.py    (287 lines)  # AA vocab, masking, SwiGLU, RoPE, transformer, ESM2, MLM loss
  train.py    (182 lines)  # FASTA dataset + training loop

proteinmpnn/
  model.py    (448 lines)  # Geometry, features (kNN/RBF/dihedrals), MPNN encoder,
                           # autoregressive decoder with random-order causal mask,
                           # ProteinMPNN with forward() + design()
  train.py    (231 lines)  # PDB parser + dataset + training loop

alphafold2/
  model.py    (636 lines)  # Geometry + RigidTransform, Evoformer (MSA attn, OPM,
                           # triangular updates/attention), IPA, StructureModule,
                           # heads (distogram, pLDDT), losses (FAPE, distogram, pLDDT),
                           # AlphaFold2 with forward() + predict()
  train.py    (227 lines)  # PDB parser + dataset + training loop

rfdiffusion/
  model.py    (557 lines)  # Geometry + quaternions + RigidTransform, SO(3) diffusion
                           # (IGSO3), R(3) diffusion (cosine DDPM), SE(3) diffusion,
                           # IPA + triangular updates, denoising network,
                           # RFDiffusion with forward() + sample()
  train.py    (227 lines)  # PDB parser + dataset + training loop

tests/
  conftest.py         (47 lines)   # Shared fixtures
  test_esm2.py        (123 lines)  # 10 tests
  test_proteinmpnn.py (145 lines)  # 10 tests
  test_alphafold2.py  (148 lines)  # 11 tests
  test_rfdiffusion.py (152 lines)  # 10 tests
```

## Coding Style

- Python 3.10+, PyTorch
- Package manager: `uv` preferred over pip
- Ruff for linting/formatting (line-length 100)
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Never push to remote without explicit permission
