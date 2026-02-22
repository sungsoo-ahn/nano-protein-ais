# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Minimal, self-contained implementations of protein AI and polymer generative models, inspired by [nanochat](https://github.com/karpathy/nanochat). Each model folder is a complete, readable implementation that a student can understand by reading a few files. The reference (full) implementation lives in `../ai-protein-impl`.

### The Five Models

- **alphafold2/** — Structure prediction: amino acid sequence → 3D backbone frames. Core ideas: Pairformer (single+pair representation, no MSA), SE(3) frame diffusion with product SO(3)×R(3) noise schedules, FAPE loss.
- **rfdiffusion/** — De novo backbone generation via SE(3) diffusion. Core ideas: SO(3)+R(3) noise schedules, denoising network with IPA, self-conditioning.
- **proteinmpnn/** — Inverse folding: backbone structure → amino acid sequence. Core ideas: k-NN graph on CA atoms, MPNN encoder, autoregressive decoder.
- **polymer_vae/** — Beta-VAE for 2D bead-spring polymer conformations. Core ideas: MLP encoder/decoder, 2D latent space, beta-KL weighting, Rg evaluation.
- **polymer_diffusion/** — DDPM for 2D bead-spring polymer conformations. Core ideas: linear noise schedule, sinusoidal timestep embedding, MLP denoiser, 200-step reverse sampling.

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
pytest tests/test_proteinmpnn.py -v    # single test file
pytest tests/ -k "test_forward" -v    # single test by name
```

## Design Philosophy (nanochat-style)

- **Flat, not deep.** Each model folder is self-contained. No shared base classes, no framework abstractions, no config registries. Duplicate a few lines rather than adding an import chain.
- **Each model = `model.py` + `train.py`.** `model.py` has the full architecture. `train.py` is a runnable script. Additional files only when a module (e.g., evoformer, diffusion) is genuinely large enough to warrant separation.
- **No configuration objects.** Use simple constants or dataclass defaults at the top of each file. A student should see the hyperparameters right where they're used.
- **Readable over reusable.** Code is meant to be forked and modified, not imported as a library. Optimize for someone reading top-to-bottom.
- **Small parameter counts.** Models should train on a single GPU in hours (AlphaFold2 ~30M, RFDiffusion ~35M, ProteinMPNN ~3.5M, Polymer VAE/Diffusion ~250K).

## Key Protein AI Concepts Used Across Models

- **ProteinStructure**: Backbone atoms (N, CA, C, O) per residue, plus sequence and validity mask.
- **Rigid frames / SE(3)**: Rotation (3×3) + translation (3,) per residue. Used in AlphaFold3 (evaluation), RFDiffusion (diffusion), ProteinMPNN (features).
- **Amino acid vocabulary**: 20 standard AAs + special tokens (PAD=20, MASK=21, CLS=22, EOS=23, UNK=24), vocab_size=25.
- **Geometry primitives**: `axis_angle_to_rotation_matrix` (Rodrigues), quaternion conversions, `compute_local_frame` from N/CA/C, dihedral angles, pairwise distances.

## Architecture

```
proteinmpnn/
  model.py    (448 lines)  # Geometry, features (kNN/RBF/dihedrals), MPNN encoder,
                           # autoregressive decoder with random-order causal mask,
                           # ProteinMPNN with forward() + design()
  train.py    (231 lines)  # PDB parser + dataset + training loop

alphafold2/
  model.py    (~650 lines) # Geometry + RigidTransform, Pairformer (single attn w/
                           # pair bias, triangular updates/attention), SE(3) diffusion
                           # (IGSO3 + cosine DDPM), diffusion transformer, DiffusionModule,
                           # heads (pLDDT), losses (FAPE, pLDDT),
                           # AlphaFold2 with forward() + predict()
  train.py    (227 lines)  # PDB parser + dataset + training loop

rfdiffusion/
  model.py    (557 lines)  # Geometry + quaternions + RigidTransform, SO(3) diffusion
                           # (IGSO3), R(3) diffusion (cosine DDPM), SE(3) diffusion,
                           # IPA + triangular updates, denoising network,
                           # RFDiffusion with forward() + sample()
  train.py    (227 lines)  # PDB parser + dataset + training loop

polymer_vae/
  model.py    (~120 lines) # Encoder, Decoder, vae_loss, PolymerVAE with
                           # forward() + sample(), radius_of_gyration
  train.py    (~130 lines) # Data download + preprocessing + dataset + training loop

polymer_diffusion/
  model.py    (~150 lines) # NoiseSchedule, SinusoidalEmbedding, Denoiser,
                           # PolymerDiffusion with forward() + sample(),
                           # radius_of_gyration
  train.py    (~130 lines) # Data download + preprocessing + dataset + training loop

tests/
  conftest.py                (47 lines)   # Shared fixtures
  test_proteinmpnn.py        (145 lines)  # 10 tests
  test_alphafold2.py         (~160 lines) # 12 tests
  test_rfdiffusion.py        (152 lines)  # 10 tests
  test_polymer_vae.py        (~80 lines)  # 9 tests
  test_polymer_diffusion.py  (~80 lines)  # 9 tests
```

## Coding Style

- Python 3.10+, PyTorch
- Package manager: `uv` preferred over pip
- Ruff for linting/formatting (line-length 100)
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Never push to remote without explicit permission
