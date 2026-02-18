# nano-protein-ais

Minimal, self-contained implementations of four protein AI models, inspired by [nanochat](https://github.com/karpathy/nanochat). Each model is a complete, readable implementation that fits in two files (`model.py` + `train.py`).

## Models

| Folder | Model | Task | Parameters |
|--------|-------|------|------------|
| `proteinmpnn/` | ProteinMPNN | Inverse folding: backbone structure -> amino acid sequence | ~3.5M |
| `alphafold2/` | AlphaFold2 | Structure prediction: amino acid sequence -> 3D backbone | ~30M |
| `rfdiffusion/` | RFDiffusion | De novo backbone generation via SE(3) diffusion | ~35M |
| `esm2/` | ESM2 | Protein language model (masked language modeling on sequences) | ~8M |

Each model folder contains:
- `model.py` -- full architecture in a single file
- `train.py` -- self-contained training script with inlined PDB parser
- `README.md` -- detailed explanation of the architecture and key concepts

## Other folders

| Folder | Purpose |
|--------|---------|
| `data/pdb/` | 9 small PDB files (1BDD, 1CRN, 1ENH, 1L2Y, 1PRB, 1UBQ, 1VII, 2GB1, 2RLK) used for training the structure models |
| `data/sequences/` | UniRef50 subset FASTA file for ESM2 training |
| `scripts/` | Training orchestration -- `run_all.sh` launches all models, `train_and_eval.py` handles individual runs, `overfit_single.py` for single-protein overfitting |
| `tests/` | Pytest suite with unit tests for all four models |

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Train a single model
cd proteinmpnn && python train.py

# Train all models in parallel (requires multiple GPUs)
bash scripts/run_all.sh

# Run tests
pytest tests/ -v
```

## Design philosophy

- **Flat, not deep.** Each model folder is self-contained. No shared base classes or framework abstractions.
- **Two files per model.** `model.py` has the full architecture, `train.py` is a runnable script.
- **No config objects.** Hyperparameters are simple constants at the top of each file.
- **Readable over reusable.** Optimized for someone reading top-to-bottom, not for import as a library.
- **Small scale.** Models train on a single GPU in hours on 9 small proteins.
