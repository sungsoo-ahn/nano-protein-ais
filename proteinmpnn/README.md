# nano-proteinmpnn

Minimal implementation of **ProteinMPNN** — inverse folding: given a protein backbone structure, predict the amino acid sequence that folds into it. ~3.5M parameters, 449 lines of model code.

Inspired by [nanochat](https://github.com/karpathy/nanochat): two files, no config objects, no framework abstractions.

## What it does

**Problem:** Given 3D coordinates of a protein backbone (N, CA, C, O atoms per residue), predict which amino acid sequence would fold into that structure.

**How:** Build a k-nearest-neighbor graph on CA atoms, encode distances and geometry as edge features, pass messages through a graph neural network (MPNN encoder), then decode the sequence autoregressively in random order.

**Why it matters:** This is the *inverse* of structure prediction (AlphaFold). It's used for protein design — given a desired backbone shape, ProteinMPNN finds sequences that fold into it. It's the core tool in the de novo protein design pipeline.

```
Input:  3D backbone coordinates  →  [N, CA, C, O] x L residues
Output: amino acid sequence      →  "MAKTEVL..."
```

## Architecture

```
Backbone Coordinates (N, CA, C, O per residue)
       │
       ├─→ Node features: dihedral angles (φ/ψ/ω) + local frame (15 dim)
       ├─→ k-NN graph on CA atoms (k=30)
       └─→ Edge features: RBF distances + seq separation + direction (324 dim)
              │
              ▼
┌─────────────────────────────────────┐
│  MPNN Encoder (x3 layers)          │
│  ┌───────────────────────────────┐ │
│  │ Message: MLP(h_src, h_dst, e) │ │
│  │ Aggregate: sum over neighbors │ │
│  │ Update: MLP(h, agg) + LN     │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
       │ encoder output (192 dim per residue)
       ▼
┌─────────────────────────────────────┐
│  Autoregressive Decoder (x3 layers)│
│  ┌───────────────────────────────┐ │
│  │ Causal self-attention         │ │
│  │   (random decoding order)     │ │
│  │ Cross-attention to encoder    │ │
│  │ FFN                           │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
       │
       ▼
  Linear → logits (20 amino acids)
```

**Key design choices:**
- **k-NN graph** — each residue connects to its 30 nearest neighbors (by CA distance), capturing spatial proximity
- **RBF distance encoding** — all 4x4 = 16 backbone atom pair distances encoded with 16 Gaussian radial basis functions = 256 edge features
- **Random-order autoregressive** — decoder generates sequence in a random permutation order (not N→C), with a causal mask ensuring only previously decoded positions are visible. This breaks sequential bias and improves diversity.
- **Message passing** — MPNN layers propagate structural information along edges of the k-NN graph

## Quick start

```bash
# Install dependencies
pip install torch numpy

# Train (auto-downloads 8 PDB files from RCSB, ~2MB total)
python train.py

# Or specify a custom PDB directory
python train.py --data_dir /path/to/your/pdbs
```

## Files

```
proteinmpnn/
├── model.py   (449 lines)  # Full ProteinMPNN architecture
│   ├── Geometry: pairwise_distances, dihedral_angle, local_frame
│   ├── Features: build_knn_graph, rbf_encode, edge/node features
│   ├── MPNNLayer: message passing on graph
│   ├── StructureEncoder: stack of MPNN layers
│   ├── AutoregressiveDecoder: causal attention + cross-attention
│   └── ProteinMPNN: forward() + design()
│
├── train.py   (257 lines)  # Self-contained training script
│   ├── PDB parser (inlined, no biopython needed)
│   ├── PDB downloader (auto-downloads from RCSB)
│   ├── PDBDataset with padding/batching
│   └── Training loop with cosine LR schedule
│
└── README.md  (this file)
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dim | 192 | Node/edge representation size |
| Encoder layers | 3 | MPNN message passing layers |
| Decoder layers | 3 | Autoregressive transformer layers |
| k neighbors | 30 | k-NN graph connectivity |
| RBF features | 16 | Gaussian basis functions per distance |
| Vocab size | 20 | Standard amino acids only |
| Batch size | 8 | |
| Learning rate | 1e-3 | AdamW |
| Parameters | ~3.5M | |

## Training curve

Expected output when training on 8 small proteins (overfitting demo):

```
step=10   epoch=5   loss=2.5143 recovery=0.128
step=20   epoch=10  loss=1.8821 recovery=0.312
step=50   epoch=25  loss=0.8432 recovery=0.621
step=100  epoch=50  loss=0.1874 recovery=0.912
step=150  epoch=75  loss=0.0321 recovery=0.985
step=200  epoch=100 loss=0.0018 recovery=1.000
```

Loss drops from ~3.0 (random: -ln(1/20)) to near zero, and sequence recovery goes from ~5% (random) to 100% on the training set. This is intentional overfitting on a tiny dataset — the model memorizes the 8 protein sequences given their structures.

## Results

### Training set (overfitting on 8 proteins)

| Metric | Start | End |
|--------|-------|-----|
| Cross-entropy loss | 2.99 | 0.002 |
| Sequence recovery | 5% | 100% |

### Evaluation (test proteins, not in training)

When evaluating on held-out proteins, recovery drops to ~3-10% — expected for a model trained on only 8 structures. The real ProteinMPNN (trained on ~18K structures) achieves 50-60% recovery.

| Protein | Length | Recovery |
|---------|--------|----------|
| 1CRN (crambin) | 46 | 2-4% |
| 1UBQ (ubiquitin) | 76 | 3-10% |
| 2GB1 (protein G) | 56 | 2-9% |

## Key concepts for students

### 1. Inverse folding
The central dogma: sequence → structure (folding). Inverse folding reverses this: structure → sequence. Given a desired backbone geometry, find sequences that would fold into it.

### 2. k-NN graph on protein structures
Proteins are 3D objects. Instead of treating the sequence linearly, ProteinMPNN builds a spatial graph: each residue connects to its 30 nearest neighbors in 3D space. Residues far apart in sequence but close in space (e.g., across a beta-sheet) become graph neighbors.

### 3. RBF distance encoding
Raw distances (5.2 Angstroms) are hard for neural nets to use. Gaussian RBF encoding maps each distance to a vector of 16 soft "bins":
```
RBF(d) = [exp(-γ(d-c₁)²), exp(-γ(d-c₂)²), ..., exp(-γ(d-c₁₆)²)]
```
where c₁...c₁₆ are centers uniformly spaced from 0 to 20 Angstroms.

### 4. Message passing neural network (MPNN)
Each layer:
1. **Message**: for each edge (i→j), compute message from node features of i, j, and edge features
2. **Aggregate**: sum all incoming messages at each node
3. **Update**: combine node features with aggregated messages

After 3 layers, each node "knows" about its 3-hop neighborhood in the graph.

### 5. Random-order autoregressive decoding
Standard autoregressive decoding (left-to-right) introduces a bias: the first residue is decoded without any sequence context, while the last sees the full sequence. ProteinMPNN decodes in a random permutation order, so each position has roughly equal context on average. This improves sequence diversity.

### 6. The design() method
At inference, `design()` generates sequences by:
1. Encoding the backbone structure (encoder pass)
2. Sampling a random decoding order
3. For each position in order: predict logits → sample from softmax/temperature → fill in
4. Repeat with different orders for diverse designs

## References

- Dauparas et al., "Robust deep learning based protein sequence design using ProteinMPNN" (2022). [Science](https://doi.org/10.1126/science.add2187)
- Ingraham et al., "Generative models for graph-based protein design" (NeurIPS 2019). [Paper](https://papers.nips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)
