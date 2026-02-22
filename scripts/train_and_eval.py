"""Train a single protein AI model on a specified GPU, then evaluate.

Usage:
    python scripts/train_and_eval.py --model proteinmpnn --gpu 0
    python scripts/train_and_eval.py --model alphafold2  --gpu 1
    python scripts/train_and_eval.py --model rfdiffusion --gpu 2
    python scripts/train_and_eval.py --model esm2        --gpu 3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Must set CUDA_VISIBLE_DEVICES before importing torch
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    choices=[
        "proteinmpnn", "alphafold2", "rfdiffusion", "esm2",
        "polymer_vae", "polymer_diffusion",
    ],
)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Create output directory before setting up file handler
Path(f"outputs/{args.model}").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{args.model.upper()}] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"outputs/{args.model}/train.log"),
    ],
)
logger = logging.getLogger(__name__)

# Ensure project root and the target model directory are on path
# (each train.py uses bare `from model import ...` for standalone use)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / args.model))

SEED = 42
DATA_DIR = ROOT / "data"
PDB_DIR = DATA_DIR / "pdb"
FASTA_PATH = DATA_DIR / "sequences" / "uniref50_subset.fasta"
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")


# ===========================================================================
# ProteinMPNN — Overfit config
# ===========================================================================
def train_proteinmpnn():
    from proteinmpnn.model import ProteinMPNN
    from proteinmpnn.train import PDBDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS = 500
    LR = 1e-3
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 10

    dataset = PDBDataset(PDB_DIR)
    BATCH_SIZE = len(dataset)  # full batch (9 PDBs)
    logger.info(f"Dataset: {len(dataset)} samples, batch_size={BATCH_SIZE}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = ProteinMPNN(dropout=0.0).to(device)  # default: h=192, 3+3 layers, ~3.5M params
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0  # constant after warmup

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

        logger.info(
            f"epoch={epoch + 1} step={global_step} loss={loss.item():.4f} "
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
            logger.info(
                f"  Design {s}: {des_str[:80]}{'...' if L > 80 else ''}  recovery={recovery:.1f}%"
            )

    logger.info("ProteinMPNN evaluation complete.")


# ===========================================================================
# Kabsch alignment (SVD-based optimal superposition)
# ===========================================================================
def kabsch_rmsd(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Compute RMSD after optimal rigid-body alignment (Kabsch algorithm).

    Args:
        pred: [L, 3] predicted coordinates
        true: [L, 3] true coordinates
    Returns:
        RMSD in Angstroms (float)
    """
    # Center both
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    true_c = true - true.mean(dim=0, keepdim=True)
    # Covariance matrix
    H = pred_c.T @ true_c  # [3, 3]
    U, S, Vt = torch.linalg.svd(H)
    # Correct for reflection
    d = torch.det(Vt.T @ U.T)
    sign = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=pred.device))
    R = Vt.T @ sign @ U.T
    pred_aligned = pred_c @ R.T
    return ((pred_aligned - true_c) ** 2).sum(dim=-1).mean().sqrt().item()


# ===========================================================================
# AlphaFold2 — Overfit config
# ===========================================================================
def train_alphafold2():
    import math

    import alphafold2.model as af2_model
    from alphafold2.model import AlphaFold2, SE3Diffusion
    from alphafold2.train import PDBDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Reduce SO(3) noise for easier rotation denoising; more sampling steps
    ROT_SIGMA = 0.8
    af2_model.NUM_SAMPLE_STEPS = 200
    logger.info(f"rot_sigma_max={ROT_SIGMA}, NUM_SAMPLE_STEPS={af2_model.NUM_SAMPLE_STEPS}")

    NUM_EPOCHS = 20000
    LR = 3e-3
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 100
    LOG_EVERY = 100

    # max_length=80 fits all 9 proteins (longest is L=76) — reduces O(L^3) tri-attn drastically
    dataset = PDBDataset(PDB_DIR, max_length=80)
    BATCH_SIZE = len(dataset)  # full batch (9 PDBs)
    logger.info(f"Dataset: {len(dataset)} samples, batch_size={BATCH_SIZE}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = AlphaFold2(plddt_weight=0.0).to(device)
    # Replace diffusion with reduced SO(3) noise
    model.diffusion = SE3Diffusion(rot_sigma_max=ROT_SIGMA)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = NUM_EPOCHS  # 1 batch per epoch

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        # Cosine decay to 1% of peak
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Center coordinates per sample (PDB coords can be far from origin)
            center = batch["coords_CA"].mean(dim=1, keepdim=True)
            for k in ("coords_N", "coords_CA", "coords_C", "coords_O"):
                batch[k] = batch[k] - center
            outputs = model(batch)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            global_step += 1

        if (epoch + 1) % LOG_EVERY == 0:
            logger.info(
                f"epoch={epoch + 1} step={global_step} "
                f"loss={loss.item():.4f} "
                f"fape={outputs['fape_loss'].item():.3f} "
                f"trans={outputs['trans_loss'].item():.3f} "
                f"rot={outputs['rot_loss'].item():.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
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
    NUM_SAMPLES = 5  # multiple predictions per protein (diffusion is stochastic)
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

        rmsds = []
        for _ in range(NUM_SAMPLES):
            with torch.no_grad():
                pred = model.predict(seq)
            pred_ca = pred["coords_CA"].cpu()[:L_real]
            rmsds.append(kabsch_rmsd(pred_ca, true_ca[:L_real]))

        best = min(rmsds)
        mean = sum(rmsds) / len(rmsds)
        logger.info(
            f"  {pdb_name} (L={L_real}): best={best:.2f}A  mean={mean:.2f}A  "
            f"(over {NUM_SAMPLES} samples)"
        )

    logger.info("AlphaFold2 evaluation complete.")


# ===========================================================================
# RFDiffusion — Overfit config
# ===========================================================================
def train_rfdiffusion():
    import math

    from rfdiffusion.model import RFDiffusion, sample
    from rfdiffusion.train import PDBDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS = 20000
    BATCH_SIZE = 9  # full batch (9 PDBs), random timesteps provide noise each step
    LR = 5e-4
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 200
    LOG_EVERY = 100

    # max_length=80 fits all 9 proteins (longest is L=76) — reduces padding waste
    dataset = PDBDataset(PDB_DIR, max_length=80)
    logger.info(f"Dataset: {len(dataset)} samples, batch_size={BATCH_SIZE}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = RFDiffusion().to(device)  # default: 8 blocks, 256 node, 64 pair
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = NUM_EPOCHS  # 1 batch per epoch

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        # Cosine decay to 1% of peak
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_trans = 0.0
        epoch_rot = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Center coordinates per sample
            center = batch["coords_CA"].mean(dim=1, keepdim=True)
            for k in ("coords_N", "coords_CA", "coords_C", "coords_O"):
                batch[k] = batch[k] - center
            outputs = model(batch)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_trans += outputs["trans_loss"].item()
            epoch_rot += outputs["rot_loss"].item()
            n_batches += 1

        if (epoch + 1) % LOG_EVERY == 0:
            logger.info(
                f"epoch={epoch + 1} step={global_step} "
                f"loss={epoch_loss / n_batches:.4f} "
                f"trans={epoch_trans / n_batches:.3f} rot={epoch_rot / n_batches:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
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
                num_steps=200,
                device=device,
            )

        final_ca = trajectory[-1].trans.cpu()
        diffs = final_ca[1:] - final_ca[:-1]
        bond_dists = torch.norm(diffs, dim=-1)
        pw = torch.cdist(final_ca.unsqueeze(0), final_ca.unsqueeze(0)).squeeze(0)
        pw_nonself = pw[~torch.eye(length, dtype=torch.bool)]

        logger.info(
            f"  L={length}: CA-CA bond mean={bond_dists.mean():.2f}A "
            f"std={bond_dists.std():.2f}A "
            f"min={bond_dists.min():.2f}A max={bond_dists.max():.2f}A"
        )
        logger.info(f"  L={length}: min pairwise dist={pw_nonself.min():.2f}A (>1A = no collision)")

    logger.info("RFDiffusion evaluation complete.")


# ===========================================================================
# ESM2 — Overfit config
# ===========================================================================
def train_esm2():
    from esm2.model import ESM2, compute_mlm_loss, mask_tokens
    from esm2.train import FASTADataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS = 200
    BATCH_SIZE = 50
    LR = 4e-4
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 10
    MAX_SAMPLES = 500

    dataset = FASTADataset(FASTA_PATH, max_length=256, max_samples=MAX_SAMPLES)
    logger.info(f"Dataset: {len(dataset)} sequences, batch_size={BATCH_SIZE}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ESM2(num_layers=12, hidden_dim=512, num_heads=16, ffn_dim=2048, dropout=0.0).to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
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
            epoch_loss += loss.item()
            epoch_acc += metrics["masked_accuracy"]
            n_batches += 1

        logger.info(
            f"epoch={epoch} step={global_step} "
            f"loss={epoch_loss / n_batches:.4f} "
            f"acc={epoch_acc / n_batches:.3f} "
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

    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    total_correct = 0
    total_masked = 0
    total_loss = 0.0
    n_eval = 0
    with torch.no_grad():
        for batch in eval_loader:
            tokens = batch["tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masked_tokens, labels = mask_tokens(tokens)
            outputs = model(masked_tokens, attention_mask)
            loss, metrics = compute_mlm_loss(outputs["logits"], labels)
            total_loss += loss.item()
            total_correct += metrics["masked_accuracy"] * (labels != -100).sum().item()
            total_masked += (labels != -100).sum().item()
            n_eval += 1

    acc = total_correct / max(1, total_masked)
    logger.info(f"  Eval loss={total_loss / n_eval:.4f}  masked_accuracy={acc:.3f}")
    logger.info("ESM2 evaluation complete.")


# ===========================================================================
# Polymer VAE
# ===========================================================================
def train_polymer_vae():
    from polymer_vae.model import PolymerVAE, radius_of_gyration
    from polymer_vae.train import PolymerDataset, download_polymer_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS = 150
    BATCH_SIZE = 256
    LR = 1e-3

    data_path = download_polymer_data(DATA_DIR / "polymer")
    train_dataset = PolymerDataset(data_path, train=True)
    test_dataset = PolymerDataset(data_path, train=False)
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PolymerVAE().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            test_loss = 0.0
            n_test = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(batch)
                    test_loss += out["loss"].item()
                    n_test += 1
            logger.info(
                f"epoch={epoch} train_loss={epoch_loss / n_batches:.4f} "
                f"test_loss={test_loss / n_test:.4f} "
                f"recon={out['recon_loss'].item():.4f} kl={out['kl_loss'].item():.4f}"
            )

    ckpt_path = Path(f"outputs/{args.model}/final_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # --- Evaluation: Rg distribution ---
    logger.info("=" * 60)
    logger.info("EVALUATION: Rg Distribution Comparison")
    logger.info("=" * 60)
    model.eval()
    samples = model.sample(1000, device=device)
    sample_rg = radius_of_gyration(samples).cpu()
    test_coords = test_dataset.coords.view(-1, 12, 2)
    data_rg = radius_of_gyration(test_coords)
    logger.info(
        f"Rg — data: mean={data_rg.mean():.3f} std={data_rg.std():.3f} | "
        f"samples: mean={sample_rg.mean():.3f} std={sample_rg.std():.3f}"
    )
    logger.info("Polymer VAE evaluation complete.")


# ===========================================================================
# Polymer Diffusion
# ===========================================================================
def train_polymer_diffusion():
    from polymer_diffusion.model import PolymerDiffusion, radius_of_gyration
    from polymer_diffusion.train import PolymerDataset, download_polymer_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    NUM_EPOCHS = 200
    BATCH_SIZE = 256
    LR = 1e-3

    data_path = download_polymer_data(DATA_DIR / "polymer")
    train_dataset = PolymerDataset(data_path, train=True)
    test_dataset = PolymerDataset(data_path, train=False)
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PolymerDiffusion().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            test_loss = 0.0
            n_test = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(batch)
                    test_loss += out["loss"].item()
                    n_test += 1
            logger.info(
                f"epoch={epoch} train_loss={epoch_loss / n_batches:.4f} "
                f"test_loss={test_loss / n_test:.4f}"
            )

    ckpt_path = Path(f"outputs/{args.model}/final_model.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # --- Evaluation: Rg distribution ---
    logger.info("=" * 60)
    logger.info("EVALUATION: Rg Distribution Comparison")
    logger.info("=" * 60)
    model.eval()
    samples = model.sample(1000, device=device)
    sample_rg = radius_of_gyration(samples).cpu()
    test_coords = test_dataset.coords.view(-1, 12, 2)
    data_rg = radius_of_gyration(test_coords)
    logger.info(
        f"Rg — data: mean={data_rg.mean():.3f} std={data_rg.std():.3f} | "
        f"samples: mean={sample_rg.mean():.3f} std={sample_rg.std():.3f}"
    )
    logger.info("Polymer Diffusion evaluation complete.")


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
        "proteinmpnn": train_proteinmpnn,
        "alphafold2": train_alphafold2,
        "rfdiffusion": train_rfdiffusion,
        "esm2": train_esm2,
        "polymer_vae": train_polymer_vae,
        "polymer_diffusion": train_polymer_diffusion,
    }
    dispatch[args.model]()
    logger.info("DONE.")
