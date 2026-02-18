"""Overfit AF2 or RFDiffusion on a single protein (1CRN) for 50K steps.

Usage:
    python scripts/overfit_single.py --model alphafold2  --gpu 1 --steps 50000 &
    python scripts/overfit_single.py --model rfdiffusion --gpu 2 --steps 50000 &
"""

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["alphafold2", "rfdiffusion"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--steps", type=int, default=50000)
parser.add_argument("--pdb", default="1CRN.pdb")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import logging  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

out_dir = Path(f"outputs/{args.model}_single")
out_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{args.model.upper()}] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "train.log")],
)
logger = logging.getLogger(__name__)

SEED = 42
LR = args.lr
GRAD_CLIP = 1.0
LOG_EVERY = 100
EVAL_EVERY = 5000


def load_protein(pdb_path: Path) -> dict:
    if args.model == "alphafold2":
        from alphafold2.train import parse_pdb
    else:
        from rfdiffusion.train import parse_pdb
    protein = parse_pdb(pdb_path)
    if protein is None:
        raise ValueError(f"Failed to parse {pdb_path}")
    # Center coordinates
    center = protein["coords_CA"].mean(dim=0, keepdim=True)
    for k in ("coords_N", "coords_CA", "coords_C", "coords_O"):
        protein[k] = protein[k] - center
    return protein


def make_batch(protein: dict, device: torch.device) -> dict:
    """Single protein → batch of 1 with no padding needed."""
    return {k: v.unsqueeze(0).to(device) for k, v in protein.items()}


def eval_af2(model, protein, device):
    seq = protein["sequence"].to(device)
    true_ca = protein["coords_CA"]
    L = true_ca.shape[0]
    with torch.no_grad():
        pred = model.predict(seq)
    pred_ca = pred["coords_CA"].cpu()[:L]
    pred_ca = pred_ca - pred_ca.mean(dim=0, keepdim=True)
    true_centered = true_ca - true_ca.mean(dim=0, keepdim=True)
    rmsd = ((pred_ca - true_centered) ** 2).sum(dim=-1).mean().sqrt().item()
    plddt = pred["plddt"].cpu()[:L].mean().item()
    logger.info(f"  EVAL: CA-RMSD={rmsd:.2f}A  mean_pLDDT={plddt:.3f}")


def eval_rfd(model, device, num_residues, protein=None):
    from rfdiffusion.model import RigidTransform, compute_local_frame, sample

    # 1) Generate unconditional backbone
    with torch.no_grad():
        trajectory = sample(
            network=model.network,
            diffusion=model.diffusion,
            num_residues=num_residues,
            num_steps=100,
            device=device,
        )
    final_ca = trajectory[-1].trans.cpu()
    diffs = final_ca[1:] - final_ca[:-1]
    bond_dists = torch.norm(diffs, dim=-1)
    parts = [
        f"  EVAL sample: CA-CA bond mean={bond_dists.mean():.2f}A "
        f"std={bond_dists.std():.2f}A  (ideal=3.8A)"
    ]
    logger.info(parts[0])

    # 2) Reconstruction: denoise from low noise to check if denoiser learned the structure
    if protein is not None:
        N = protein["coords_N"].unsqueeze(0).to(device)
        CA = protein["coords_CA"].unsqueeze(0).to(device)
        C = protein["coords_C"].unsqueeze(0).to(device)
        true_frames = compute_local_frame(N, CA, C)
        # Add small noise (t=0.05) and denoise
        t_low = torch.full((1, num_residues), 0.05, device=device)
        noisy, _ = model.diffusion.forward_marginal(true_frames, t_low)
        with torch.no_grad():
            pred = model.network(noisy, torch.tensor([0.05], device=device))
        pred_ca = pred.trans.squeeze(0).cpu()
        true_ca = protein["coords_CA"]
        pred_ca = pred_ca - pred_ca.mean(dim=0, keepdim=True)
        true_ca_c = true_ca - true_ca.mean(dim=0, keepdim=True)
        rmsd = ((pred_ca - true_ca_c) ** 2).sum(dim=-1).mean().sqrt().item()
        logger.info(f"  EVAL denoise: CA-RMSD={rmsd:.2f}A (from t=0.05)")


def train_alphafold2(protein, device):
    from alphafold2.model import AlphaFold2

    model = AlphaFold2(plddt_weight=0.0).to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    batch = make_batch(protein, device)

    for step in range(1, args.steps + 1):
        model.train()
        outputs = model(batch)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            logger.info(
                f"step={step} loss={loss.item():.4f} "
                f"fape={outputs['fape_loss'].item():.3f} "
                f"trans={outputs['trans_loss'].item():.3f} "
                f"rot={outputs['rot_loss'].item():.3f}"
            )
        if step % EVAL_EVERY == 0:
            eval_af2(model, protein, device)

    torch.save({"model_state_dict": model.state_dict()}, out_dir / "final_model.pt")
    logger.info(f"Saved checkpoint to {out_dir / 'final_model.pt'}")
    logger.info("Final evaluation:")
    eval_af2(model, protein, device)


def train_rfdiffusion(protein, device):
    from rfdiffusion.model import RFDiffusion

    model = RFDiffusion().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    batch = make_batch(protein, device)
    L = protein["coords_CA"].shape[0]

    for step in range(1, args.steps + 1):
        model.train()
        outputs = model(batch)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            logger.info(
                f"step={step} loss={loss.item():.4f} "
                f"trans={outputs['trans_loss'].item():.3f} rot={outputs['rot_loss'].item():.3f}"
            )
        if step % EVAL_EVERY == 0:
            eval_rfd(model, device, L, protein)

    torch.save({"model_state_dict": model.state_dict()}, out_dir / "final_model.pt")
    logger.info(f"Saved checkpoint to {out_dir / 'final_model.pt'}")
    logger.info("Final evaluation:")
    eval_rfd(model, device, L, protein)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting {args.model} single-protein overfit on GPU {args.gpu}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    pdb_path = ROOT / "data" / "pdb" / args.pdb
    protein = load_protein(pdb_path)
    L = protein["coords_CA"].shape[0]
    logger.info(f"Protein: {args.pdb} (L={L}), {args.steps} steps, LR={LR}, resume={args.resume}")

    if args.model == "alphafold2":
        train_alphafold2(protein, device)
    else:
        train_rfdiffusion(protein, device)
    logger.info("DONE.")
