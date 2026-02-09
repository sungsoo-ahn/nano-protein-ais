"""Train a single protein AI model on a specified GPU, then evaluate.

Usage:
    python scripts/train_and_eval.py --model esm2 --gpu 0
    python scripts/train_and_eval.py --model proteinmpnn --gpu 1
    python scripts/train_and_eval.py --model alphafold2 --gpu 2
    python scripts/train_and_eval.py --model rfdiffusion --gpu 3
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

# Must set CUDA_VISIBLE_DEVICES before importing torch
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["esm2", "proteinmpnn", "alphafold2", "rfdiffusion"])
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{args.model.upper()}] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"outputs/{args.model}/train.log"),
    ],
)
logger = logging.getLogger(__name__)

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
DATA_DIR = ROOT / "data"
PDB_DIR = DATA_DIR / "pdb"
FASTA_PATH = DATA_DIR / "sequences" / "uniref50_subset.fasta"

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")


# ===========================================================================
# ESM2
# ===========================================================================
def train_esm2():
    from esm2.model import ESM2, IDX_TO_AA, compute_mlm_loss, mask_tokens
    from esm2.train import FASTADataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS, BATCH_SIZE, LR, GRAD_CLIP, LOG_EVERY = 3, 32, 4e-4, 1.0, 50
    WARMUP_STEPS = 500

    dataset = FASTADataset(FASTA_PATH)
    logger.info(f"Dataset: {len(dataset)} sequences")
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )

    model = ESM2().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    num_steps = len(dataloader) * NUM_EPOCHS

    def lr_lambda(step):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                logger.info(
                    f"step={global_step} epoch={epoch} loss={loss.item():.4f} "
                    f"acc={metrics['masked_accuracy']:.3f} ppl={metrics['perplexity']:.1f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

    ckpt_path = Path(f"outputs/{args.model}/final_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # --- Evaluation ---
    logger.info("=" * 60)
    logger.info("EVALUATION: Masked Language Modeling")
    logger.info("=" * 60)
    model.eval()

    # Take 5 sequences from the dataset
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        tokens = sample["tokens"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        masked_tokens, labels = mask_tokens(tokens)
        with torch.no_grad():
            outputs = model(masked_tokens, attention_mask)
            loss, metrics = compute_mlm_loss(outputs["logits"], labels)

        # Show masked positions and predictions
        mask_pos = (labels[0] != -100).nonzero(as_tuple=True)[0]
        pred_ids = outputs["logits"][0].argmax(dim=-1)
        true_ids = labels[0]

        n_correct = 0
        examples = []
        for pos in mask_pos[:10]:  # show up to 10 positions
            p, t = pred_ids[pos].item(), true_ids[pos].item()
            correct = p == t
            n_correct += int(correct)
            pred_aa = IDX_TO_AA.get(p, "?")
            true_aa = IDX_TO_AA.get(t, "?")
            marker = "OK" if correct else "XX"
            examples.append(f"  pos={pos.item():3d}: pred={pred_aa} true={true_aa} [{marker}]")

        logger.info(
            f"Seq {i}: loss={loss.item():.3f} acc={metrics['masked_accuracy']:.3f} "
            f"ppl={metrics['perplexity']:.1f}"
        )
        for ex in examples:
            logger.info(ex)

    logger.info("ESM2 evaluation complete.")


# ===========================================================================
# ProteinMPNN
# ===========================================================================
def train_proteinmpnn():
    from proteinmpnn.model import ProteinMPNN
    from proteinmpnn.train import PDBDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS, BATCH_SIZE, LR, GRAD_CLIP, LOG_EVERY = 100, 8, 1e-3, 1.0, 10
    WARMUP_STEPS = 100

    dataset = PDBDataset(PDB_DIR)
    logger.info(f"Dataset: {len(dataset)} proteins")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = ProteinMPNN().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                logger.info(
                    f"step={global_step} epoch={epoch+1} loss={loss.item():.4f} "
                    f"recovery={outputs['sequence_recovery'].item():.3f}"
                )

    ckpt_path = Path(f"outputs/{args.model}/final_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # --- Evaluation ---
    logger.info("=" * 60)
    logger.info("EVALUATION: Inverse Folding (Sequence Design)")
    logger.info("=" * 60)
    model.eval()

    from proteinmpnn.train import parse_pdb

    eval_pdbs = ["1CRN.pdb", "1UBQ.pdb", "2GB1.pdb"]
    for pdb_name in eval_pdbs:
        pdb_path = PDB_DIR / pdb_name
        if not pdb_path.exists():
            logger.warning(f"  {pdb_name} not found, skipping")
            continue

        protein = parse_pdb(pdb_path)
        if protein is None:
            continue

        L = protein["coords_CA"].shape[0]
        native_seq = protein["sequence"]
        native_str = "".join(AA_VOCAB[i] if i < 20 else "X" for i in native_seq.tolist())

        # Design 5 sequences at temperature 0.1
        with torch.no_grad():
            designed = model.design(
                coords_N=protein["coords_N"].to(device),
                coords_CA=protein["coords_CA"].to(device),
                coords_C=protein["coords_C"].to(device),
                coords_O=protein["coords_O"].to(device),
                mask=protein["mask"].to(device),
                temperature=0.1,
                num_samples=5,
            )

        logger.info(f"\n  {pdb_name} (L={L}):")
        logger.info(f"  Native:   {native_str[:80]}{'...' if L > 80 else ''}")
        for s in range(designed.shape[0]):
            des_seq = designed[s].cpu()
            des_str = "".join(AA_VOCAB[i] if i < 20 else "X" for i in des_seq.tolist())
            mask = protein["mask"]
            match = ((des_seq == native_seq) & mask).sum().item()
            total = mask.sum().item()
            recovery = match / total * 100
            logger.info(f"  Design {s}: {des_str[:80]}{'...' if L > 80 else ''}  recovery={recovery:.1f}%")

    logger.info("ProteinMPNN evaluation complete.")


# ===========================================================================
# AlphaFold2
# ===========================================================================
def train_alphafold2():
    from alphafold2.model import AlphaFold2
    from alphafold2.train import PDBDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS, BATCH_SIZE, LR, GRAD_CLIP, LOG_EVERY = 100, 1, 1e-3, 1.0, 10
    WARMUP_STEPS = 100

    dataset = PDBDataset(PDB_DIR)
    logger.info(f"Dataset: {len(dataset)} proteins")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = AlphaFold2().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                logger.info(
                    f"step={global_step} epoch={epoch+1} loss={loss.item():.4f} "
                    f"fape={outputs['fape'].item():.4f}"
                )

    ckpt_path = Path(f"outputs/{args.model}/final_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # --- Evaluation ---
    logger.info("=" * 60)
    logger.info("EVALUATION: Structure Prediction")
    logger.info("=" * 60)

    from alphafold2.train import parse_pdb

    eval_pdbs = ["1CRN.pdb", "1UBQ.pdb", "2GB1.pdb"]
    for pdb_name in eval_pdbs:
        pdb_path = PDB_DIR / pdb_name
        if not pdb_path.exists():
            logger.warning(f"  {pdb_name} not found, skipping")
            continue

        protein = parse_pdb(pdb_path)
        if protein is None:
            continue

        seq = protein["sequence"].to(device)
        true_ca = protein["coords_CA"]
        mask = protein["mask"]
        L_real = mask.sum().item()

        with torch.no_grad():
            pred = model.predict(seq)

        pred_ca = pred["coords"].cpu()[:L_real]
        true_ca = true_ca[:L_real]
        plddt = pred["plddt"].cpu()[:L_real]

        # Compute CA RMSD (after centering)
        pred_center = pred_ca - pred_ca.mean(dim=0, keepdim=True)
        true_center = true_ca - true_ca.mean(dim=0, keepdim=True)
        rmsd = ((pred_center - true_center) ** 2).sum(dim=-1).mean().sqrt().item()

        logger.info(
            f"  {pdb_name} (L={L_real}): CA-RMSD={rmsd:.2f}A  "
            f"mean_pLDDT={plddt.mean().item():.3f}"
        )

    logger.info("AlphaFold2 evaluation complete.")


# ===========================================================================
# RFDiffusion
# ===========================================================================
def train_rfdiffusion():
    from rfdiffusion.model import RFDiffusion, sample
    from rfdiffusion.train import PDBDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS, BATCH_SIZE, LR, GRAD_CLIP, LOG_EVERY = 150, 4, 1e-4, 1.0, 10
    WARMUP_STEPS = 500

    dataset = PDBDataset(PDB_DIR)
    logger.info(f"Dataset: {len(dataset)} proteins")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = RFDiffusion().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                logger.info(
                    f"step={global_step} epoch={epoch+1} loss={loss.item():.4f} "
                    f"trans={outputs['trans_loss'].item():.3f} "
                    f"rot={outputs['rot_loss'].item():.3f}"
                )

    ckpt_path = Path(f"outputs/{args.model}/final_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # --- Evaluation ---
    logger.info("=" * 60)
    logger.info("EVALUATION: Backbone Generation via Diffusion")
    logger.info("=" * 60)

    for length in [50, 80, 100]:
        logger.info(f"\n  Generating backbone with {length} residues...")
        with torch.no_grad():
            trajectory = sample(
                network=model.network,
                diffusion=model.diffusion,
                num_residues=length,
                num_steps=100,
                device=device,
            )

        # Final backbone: trajectory[-1].trans gives CA positions
        final_ca = trajectory[-1].trans.cpu()  # [L, 3]

        # CA-CA bond distances (consecutive)
        diffs = final_ca[1:] - final_ca[:-1]
        bond_dists = torch.norm(diffs, dim=-1)

        # Pairwise distances (check for collisions)
        pw = torch.cdist(final_ca.unsqueeze(0), final_ca.unsqueeze(0)).squeeze(0)
        # Exclude self-distances
        pw_nonself = pw[~torch.eye(length, dtype=torch.bool)]

        logger.info(
            f"  L={length}: CA-CA bond mean={bond_dists.mean():.2f}A "
            f"std={bond_dists.std():.2f}A "
            f"min={bond_dists.min():.2f}A max={bond_dists.max():.2f}A"
        )
        logger.info(
            f"  L={length}: min pairwise dist={pw_nonself.min():.2f}A "
            f"(>1A = no collision)"
        )

    logger.info("RFDiffusion evaluation complete.")


# ===========================================================================
# Main dispatch
# ===========================================================================
if __name__ == "__main__":
    # Create output directory
    out_dir = Path(f"outputs/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    logger.info(f"Starting {args.model} on GPU {args.gpu}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    dispatch = {
        "esm2": train_esm2,
        "proteinmpnn": train_proteinmpnn,
        "alphafold2": train_alphafold2,
        "rfdiffusion": train_rfdiffusion,
    }
    dispatch[args.model]()
    logger.info("DONE.")
