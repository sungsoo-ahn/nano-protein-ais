"""Training script for ProteinMPNN inverse folding model.

Self-contained: includes PDB parser + dataset + training loop.

Usage:
    python -m proteinmpnn.train [--data_dir path/to/pdbs]
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from proteinmpnn.model import ProteinMPNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
WARMUP_STEPS = 10
GRADIENT_CLIP = 1.0
MAX_SEQ_LEN = 200
SEED = 42
LOG_EVERY = 10

# ---------------------------------------------------------------------------
# PDB parser (inlined)
# ---------------------------------------------------------------------------

AA_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
UNK_IDX = 24


def parse_pdb(pdb_path: Path) -> dict[str, torch.Tensor] | None:
    """Parse PDB file -> dict with coords_N/CA/C/O [L,3], sequence [L], mask [L]."""
    residues: dict[int, dict[str, list[float]]] = {}
    residue_names: dict[int, str] = {}

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C", "O"):
                continue
            alt_loc = line[16]
            if alt_loc not in (" ", "A"):
                continue
            res_seq = int(line[22:26])
            if res_seq not in residues:
                residues[res_seq] = {}
                residue_names[res_seq] = line[17:20].strip()
            coords = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            residues[res_seq][atom_name] = coords

    coords_N, coords_CA, coords_C, coords_O, seq_idx = [], [], [], [], []
    for rid in sorted(residues):
        atoms = residues[rid]
        if not all(a in atoms for a in ("N", "CA", "C")):
            continue
        coords_N.append(atoms["N"])
        coords_CA.append(atoms["CA"])
        coords_C.append(atoms["C"])
        coords_O.append(atoms.get("O", [0.0, 0.0, 0.0]))
        aa1 = AA_3TO1.get(residue_names[rid], "X")
        seq_idx.append(AA_TO_IDX.get(aa1, UNK_IDX))

    if not coords_CA:
        return None
    L = len(coords_CA)
    return {
        "coords_N": torch.tensor(coords_N, dtype=torch.float32),
        "coords_CA": torch.tensor(coords_CA, dtype=torch.float32),
        "coords_C": torch.tensor(coords_C, dtype=torch.float32),
        "coords_O": torch.tensor(coords_O, dtype=torch.float32),
        "sequence": torch.tensor(seq_idx, dtype=torch.long),
        "mask": torch.ones(L, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def pad_protein(protein: dict, max_len: int) -> dict:
    L = protein["coords_CA"].shape[0]
    if L >= max_len:
        return {k: v[:max_len] if k != "mask" else v[:max_len] for k, v in protein.items()}
    pad = max_len - L

    def pad_c(c):
        return torch.cat([c, torch.zeros(pad, 3)])

    return {
        "coords_N": pad_c(protein["coords_N"]),
        "coords_CA": pad_c(protein["coords_CA"]),
        "coords_C": pad_c(protein["coords_C"]),
        "coords_O": pad_c(protein["coords_O"]),
        "sequence": torch.cat([protein["sequence"], torch.zeros(pad, dtype=torch.long)]),
        "mask": torch.cat([protein["mask"], torch.zeros(pad, dtype=torch.bool)]),
    }


class PDBDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        max_length: int = MAX_SEQ_LEN,
        min_length: int = 20,
        num_repeats: int = 1,
    ):
        self.proteins = []
        self.max_length = max_length
        self.num_repeats = num_repeats
        for pdb_path in sorted(data_dir.glob("*.pdb")):
            try:
                protein = parse_pdb(pdb_path)
                if protein is not None:
                    L = protein["coords_CA"].shape[0]
                    if min_length <= L <= max_length:
                        self.proteins.append(protein)
            except Exception as e:
                logger.warning(f"Failed to parse {pdb_path.name}: {e}")
        logger.info(
            f"Loaded {len(self.proteins)} proteins (x{num_repeats} repeats"
            f" = {len(self)} effective samples)"
        )

    def __len__(self):
        return len(self.proteins) * self.num_repeats

    def __getitem__(self, idx):
        return pad_protein(self.proteins[idx % len(self.proteins)], self.max_length)


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(data_dir: str, output_dir: str = "outputs/proteinmpnn") -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PDBDataset(Path(data_dir))
    if len(dataset) == 0:
        raise ValueError(f"No valid proteins in {data_dir}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = ProteinMPNN().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_steps = len(dataloader) * NUM_EPOCHS

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, num_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                logger.info(
                    f"step={global_step} epoch={epoch + 1} loss={loss.item():.4f} "
                    f"recovery={outputs['sequence_recovery'].item():.3f}"
                )

        if (epoch + 1) % 10 == 0:
            ckpt = {"model_state_dict": model.state_dict()}
            torch.save(ckpt, output_dir / f"ckpt_ep{epoch + 1}.pt")

    torch.save({"model_state_dict": model.state_dict()}, output_dir / "final_model.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/pdb")
    parser.add_argument("--output_dir", default="outputs/proteinmpnn")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
