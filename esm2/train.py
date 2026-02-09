"""Training script for ESM2 masked language model.

Trains ESM2 from scratch on protein sequences in FASTA format.
Self-contained: includes FASTA dataset loader + training loop.

Usage:
    python -m esm2.train [--data_path path/to/seqs.fasta]
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from esm2.model import (
    AA_TO_IDX,
    CLS_IDX,
    EOS_IDX,
    ESM2,
    PAD_IDX,
    UNK_IDX,
    compute_mlm_loss,
    mask_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
WARMUP_STEPS = 500
GRADIENT_CLIP = 1.0
MAX_SEQ_LEN = 512
SEED = 42
LOG_EVERY = 50

# ---------------------------------------------------------------------------
# FASTA dataset
# ---------------------------------------------------------------------------


def tokenize_sequence(sequence: str) -> list[int]:
    """Convert amino acid sequence to token indices with CLS/EOS."""
    return [CLS_IDX] + [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence.upper()] + [EOS_IDX]


class FASTADataset(Dataset):
    """Dataset of protein sequences from a FASTA file."""

    def __init__(self, fasta_path: str | Path, max_length: int = MAX_SEQ_LEN, min_length: int = 20):
        self.max_tokens = max_length + 2  # CLS + seq + EOS
        self.sequences: list[torch.Tensor] = []

        current_seq: list[str] = []
        path = Path(fasta_path)
        if not path.exists():
            logger.warning(f"FASTA file not found: {path}")
            return

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        seq = "".join(current_seq)
                        if min_length <= len(seq) <= max_length:
                            tokens = torch.tensor(tokenize_sequence(seq), dtype=torch.long)
                            self.sequences.append(tokens)
                    current_seq = []
                elif line:
                    current_seq.append(line)
            if current_seq:
                seq = "".join(current_seq)
                if min_length <= len(seq) <= max_length:
                    self.sequences.append(torch.tensor(tokenize_sequence(seq), dtype=torch.long))

        logger.info(f"Loaded {len(self.sequences)} sequences from {path}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        length = len(tokens)
        padded = torch.full((self.max_tokens,), PAD_IDX, dtype=torch.long)
        padded[:length] = tokens
        attention_mask = torch.zeros(self.max_tokens, dtype=torch.bool)
        attention_mask[:length] = True
        return {"tokens": padded, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(data_path: str, output_dir: str = "outputs/esm2") -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FASTADataset(data_path)
    if len(dataset) == 0:
        raise ValueError(f"No sequences found in {data_path}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = ESM2().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_steps = len(dataloader) * NUM_EPOCHS

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, num_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_tokens, labels = mask_tokens(tokens)
            outputs = model(masked_tokens, attention_mask)
            loss, metrics = compute_mlm_loss(outputs["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            if GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                logger.info(
                    f"step={global_step} epoch={epoch} loss={loss.item():.4f} "
                    f"acc={metrics['masked_accuracy']:.3f} ppl={metrics['perplexity']:.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Save checkpoint
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            output_dir / f"checkpoint_epoch_{epoch}.pt",
        )
        logger.info(f"Saved checkpoint epoch {epoch}")

    torch.save({"model_state_dict": model.state_dict()}, output_dir / "final_model.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/sequences/uniref50_subset.fasta")
    parser.add_argument("--output_dir", default="outputs/esm2")
    args = parser.parse_args()
    train(args.data_path, args.output_dir)
