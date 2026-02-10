"""Visualize AlphaFold3 predictions and RFDiffusion samples with PyMOL.

Loads trained checkpoints, runs inference, writes CA-only PDB files,
generates a PyMOL render script, and executes it.

Usage:
    python scripts/visualize_results.py --gpu 0
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must set CUDA_VISIBLE_DEVICES before importing torch
parser = argparse.ArgumentParser(description="Visualize AF3 and RFDiffusion results")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--no-render", action="store_true", help="Generate PDBs and script only")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch  # noqa: E402

VIS_DIR = ROOT / "outputs" / "visualizations"
PDB_DIR = ROOT / "data" / "pdb"
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_3TO1 = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4, "GLY": 5, "HIS": 6,
    "ILE": 7, "LYS": 8, "LEU": 9, "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13,
    "ARG": 14, "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
}
IDX_TO_3 = {v: k for k, v in AA_3TO1.items()}


def write_ca_pdb(path: Path, ca_coords: torch.Tensor, sequence: torch.Tensor | None = None):
    """Write CA-only PDB file. ca_coords: [L, 3], sequence: [L] optional."""
    L = ca_coords.shape[0]
    with open(path, "w") as f:
        for i in range(L):
            if sequence is not None and sequence[i].item() < 20:
                resname = IDX_TO_3.get(sequence[i].item(), "ALA")
            else:
                resname = "ALA"
            x, y, z = ca_coords[i].tolist()
            f.write(
                f"ATOM  {i + 1:5d}  CA  {resname:3s} A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


# ---------------------------------------------------------------------------
# Step 1 & 2: Generate structures and write PDBs
# ---------------------------------------------------------------------------

def generate_af3_structures():
    """Load AF3 checkpoint, predict structures for eval proteins."""
    from alphafold3.model import AlphaFold3
    from alphafold3.train import parse_pdb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Try single-protein overfit checkpoint first, then fall back to multi-protein
    ckpt_path = ROOT / "outputs" / "alphafold3_single" / "final_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "outputs" / "alphafold3" / "final_model.pt"
    if not ckpt_path.exists():
        print(f"AF3 checkpoint not found")
        return []
    print(f"  Loading AF3 from {ckpt_path}")

    # Must match training config: sigma_data=7.0, zero aux weights
    model = AlphaFold3(distogram_weight=0.0, plddt_weight=0.0, sigma_data=7.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eval_pdbs = ["1CRN", "1UBQ", "2GB1"]
    results = []
    for name in eval_pdbs:
        pdb_path = PDB_DIR / f"{name}.pdb"
        if not pdb_path.exists():
            print(f"  {name}.pdb not found, skipping")
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

        # Write predicted CA PDB
        pred_path = VIS_DIR / f"af3_pred_{name}.pdb"
        write_ca_pdb(pred_path, pred_ca, protein["sequence"][:L_real])

        # Copy native PDB for overlay
        native_dst = VIS_DIR / f"af3_native_{name}.pdb"
        shutil.copy2(pdb_path, native_dst)

        # Compute RMSD (centered) for labeling
        pred_center = pred_ca - pred_ca.mean(dim=0, keepdim=True)
        true_center = true_ca - true_ca.mean(dim=0, keepdim=True)
        rmsd = ((pred_center - true_center) ** 2).sum(dim=-1).mean().sqrt().item()

        print(f"  AF3 {name}: L={L_real}, CA-RMSD={rmsd:.2f}A")
        results.append((name, rmsd))

    return results


def generate_rfd_structures():
    """Load RFDiffusion checkpoint, sample backbones and denoise native."""
    from rfdiffusion.model import RFDiffusion, compute_local_frame, sample
    from rfdiffusion.train import parse_pdb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Try single-protein overfit checkpoint first, then fall back to multi-protein
    ckpt_path = ROOT / "outputs" / "rfdiffusion_single" / "final_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "outputs" / "rfdiffusion" / "final_model.pt"
    if not ckpt_path.exists():
        print(f"RFDiffusion checkpoint not found")
        return []
    print(f"  Loading RFDiffusion from {ckpt_path}")

    model = RFDiffusion().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Also generate denoised native structure for 1CRN
    crn_path = PDB_DIR / "1CRN.pdb"
    if crn_path.exists():
        protein = parse_pdb(crn_path)
        if protein is not None:
            N = protein["coords_N"].unsqueeze(0).to(device)
            CA = protein["coords_CA"].unsqueeze(0).to(device)
            C = protein["coords_C"].unsqueeze(0).to(device)
            # Center
            center = CA.mean(dim=1, keepdim=True)
            N, CA, C = N - center, CA - center, C - center
            true_frames = compute_local_frame(N, CA, C)
            L = CA.shape[1]
            t_low = torch.full((1, L), 0.05, device=device)
            noisy, _ = model.diffusion.forward_marginal(true_frames, t_low)
            with torch.no_grad():
                pred = model.network(noisy, torch.tensor([0.05], device=device))
            pred_ca = pred.trans.squeeze(0).cpu()
            true_ca = CA.squeeze(0).cpu()

            write_ca_pdb(VIS_DIR / "rfd_denoise_1CRN.pdb", pred_ca, protein["sequence"])
            # Write centered native for comparison
            write_ca_pdb(VIS_DIR / "rfd_native_1CRN.pdb", true_ca, protein["sequence"])

            pred_c = pred_ca - pred_ca.mean(dim=0, keepdim=True)
            true_c = true_ca - true_ca.mean(dim=0, keepdim=True)
            rmsd = ((pred_c - true_c) ** 2).sum(dim=-1).mean().sqrt().item()
            print(f"  RFDiffusion denoise 1CRN: CA-RMSD={rmsd:.2f}A")

    lengths = [30, 50, 70]
    results = []
    for L in lengths:
        print(f"  RFDiffusion: sampling L={L}...")
        with torch.no_grad():
            trajectory = sample(
                network=model.network,
                diffusion=model.diffusion,
                num_residues=L,
                num_steps=100,
                device=device,
            )

        ca_coords = trajectory[-1].trans.cpu()  # [L, 3]

        # Bond length stats
        diffs = ca_coords[1:] - ca_coords[:-1]
        bond_dists = torch.norm(diffs, dim=-1)
        print(f"  RFDiffusion L={L}: CA-CA bond mean={bond_dists.mean():.2f}A")

        pdb_path = VIS_DIR / f"rfd_sample_L{L}.pdb"
        write_ca_pdb(pdb_path, ca_coords)
        results.append(L)

    return results


# ---------------------------------------------------------------------------
# Step 3: Generate PyMOL render script
# ---------------------------------------------------------------------------

def write_render_script(af3_results: list, rfd_lengths: list):
    """Generate PyMOL render script for all structures."""
    script_path = VIS_DIR / "render.py"
    vis_abs = str(VIS_DIR.resolve())

    lines = [
        "import pymol",
        "from pymol import cmd",
        "",
        "pymol.finish_launching(['pymol', '-cq'])",
        "",
        "# Global settings",
        "cmd.set('antialias', 2)",
        "cmd.set('specular', 0.3)",
        "cmd.set('ray_shadow', 1)",
        "cmd.set('ray_opaque_background', 0)",
        "cmd.set('depth_cue', 1)",
        "cmd.set('fog_start', 0.4)",
        "cmd.set('sphere_scale', 0.6)",
        "cmd.set('stick_radius', 0.15)",
        "",
    ]

    # AF3 renders: predicted vs native overlay
    for name, rmsd in af3_results:
        pred_pdb = os.path.join(vis_abs, f"af3_pred_{name}.pdb")
        native_pdb = os.path.join(vis_abs, f"af3_native_{name}.pdb")
        png_path = os.path.join(vis_abs, f"af3_{name}.png")
        lines += [
            f"# --- AF3: {name} (RMSD={rmsd:.1f}A) ---",
            f"cmd.load({pred_pdb!r}, 'pred')",
            f"cmd.load({native_pdb!r}, 'native')",
            "",
            "# Align predicted to native",
            "try:",
            "    rms = cmd.align('pred', 'native and name CA')[0]",
            "    print(f'Alignment RMSD: {rms:.2f}A')",
            "except:",
            "    print('Alignment failed, showing unaligned')",
            "",
            "# Style native: cartoon",
            "cmd.hide('everything', 'native')",
            "cmd.show('cartoon', 'native')",
            "cmd.color('cyan', 'native')",
            "cmd.set('cartoon_transparency', 0.3, 'native')",
            "",
            "# Style predicted: CA spheres",
            "cmd.hide('everything', 'pred')",
            "cmd.show('spheres', 'pred')",
            "cmd.color('green', 'pred')",
            "",
            "cmd.zoom('native', buffer=5)",
            "cmd.ray(1200, 900)",
            f"cmd.png({png_path!r}, dpi=300)",
            f"print('Saved {png_path}')",
            "cmd.delete('all')",
            "",
        ]

    # RFDiffusion denoise overlay (if generated)
    denoise_pdb = os.path.join(vis_abs, "rfd_denoise_1CRN.pdb")
    native_ca_pdb = os.path.join(vis_abs, "rfd_native_1CRN.pdb")
    if os.path.exists(denoise_pdb):
        png_path = os.path.join(vis_abs, "rfd_denoise_1CRN.png")
        lines += [
            "# --- RFDiffusion Denoise: 1CRN ---",
            f"cmd.load({denoise_pdb!r}, 'denoised')",
            f"cmd.load({native_ca_pdb!r}, 'native_ca')",
            "",
            "# Align",
            "try:",
            "    rms = cmd.align('denoised', 'native_ca')[0]",
            "    print(f'Denoise alignment RMSD: {rms:.2f}A')",
            "except:",
            "    print('Denoise alignment failed')",
            "",
            "# Style native: spheres cyan",
            "cmd.hide('everything', 'native_ca')",
            "cmd.show('spheres', 'native_ca')",
            "cmd.color('cyan', 'native_ca')",
            "cmd.set('sphere_transparency', 0.4, 'native_ca')",
            "",
            "# Style denoised: spheres orange",
            "cmd.hide('everything', 'denoised')",
            "cmd.show('spheres', 'denoised')",
            "cmd.color('orange', 'denoised')",
            "",
            "cmd.zoom('native_ca', buffer=5)",
            "cmd.ray(1200, 900)",
            f"cmd.png({png_path!r}, dpi=300)",
            f"print('Saved {png_path}')",
            "cmd.delete('all')",
            "",
        ]

    # RFDiffusion renders: de novo backbones
    for L in rfd_lengths:
        pdb_path = os.path.join(vis_abs, f"rfd_sample_L{L}.pdb")
        png_path = os.path.join(vis_abs, f"rfd_L{L}.png")
        lines += [
            f"# --- RFDiffusion: L={L} ---",
            f"cmd.load({pdb_path!r}, 'bbone')",
            "",
            "# Create bonds between consecutive CAs",
            f"for i in range(1, {L}):",
            "    cmd.bond("
            "f'backbone and resi {i} and name CA', "
            "f'backbone and resi {i+1} and name CA')",
            "",
            "# Style: spheres + sticks with rainbow coloring",
            "cmd.hide('everything', 'bbone')",
            "cmd.show('spheres', 'bbone')",
            "cmd.show('sticks', 'bbone')",
            "cmd.spectrum('resi', 'rainbow', 'bbone')",
            "",
            "cmd.zoom('bbone', buffer=5)",
            "cmd.ray(1200, 900)",
            f"cmd.png({png_path!r}, dpi=300)",
            f"print('Saved {png_path}')",
            "cmd.delete('all')",
            "",
        ]

    lines.append("cmd.quit()")

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return script_path


# ---------------------------------------------------------------------------
# Step 4: Execute PyMOL
# ---------------------------------------------------------------------------

def run_pymol(script_path: Path):
    """Run render script with PyMOL."""
    pymol_python = os.path.expanduser("~/.conda/envs/pymol/bin/python")
    if not os.path.isfile(pymol_python):
        pymol_bin = shutil.which("pymol")
        if pymol_bin:
            pymol_python = os.path.join(os.path.dirname(pymol_bin), "python")
            if not os.path.isfile(pymol_python):
                pymol_python = None
        else:
            pymol_python = None

    if pymol_python is None:
        print("PyMOL not found. Install with:")
        print("  conda create -n pymol -c conda-forge pymol-open-source python=3.12")
        print(f"Or run manually: python {script_path}")
        return

    print(f"Running PyMOL: {pymol_python} {script_path}")
    result = subprocess.run(
        [pymol_python, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        print("Rendering complete!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"PyMOL error (exit {result.returncode}):")
        if result.stdout:
            print(result.stdout[:500])
        if result.stderr:
            print(result.stderr[:500])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1-2: Generate structures and write PDBs")
    print("=" * 60)

    print("\nAlphaFold3 predictions:")
    af3_results = generate_af3_structures()

    print("\nRFDiffusion samples:")
    rfd_lengths = generate_rfd_structures()

    if not af3_results and not rfd_lengths:
        print("No structures generated. Check checkpoints exist.")
        return

    print("\n" + "=" * 60)
    print("Step 3: Generate PyMOL render script")
    print("=" * 60)
    script_path = write_render_script(af3_results, rfd_lengths)
    print(f"Render script: {script_path}")

    if args.no_render:
        print("Skipping rendering (--no-render)")
        return

    print("\n" + "=" * 60)
    print("Step 4: Render with PyMOL")
    print("=" * 60)
    run_pymol(script_path)

    # List outputs
    print("\nOutput files:")
    for f in sorted(VIS_DIR.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
