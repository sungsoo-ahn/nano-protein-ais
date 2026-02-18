"""Visualize SE(3) frames from RFDiffusion as full backbone proteins with Cbeta.

Given per-residue SE(3) frames (rotation + translation), reconstruct the full
backbone (N, CA, C, O) plus virtual Cbeta using ideal bond geometry, then write
proper PDB files and render with PyMOL.

Usage:
    python scripts/visualize_frames.py --gpu 0
    python scripts/visualize_frames.py --gpu 0 --no-render  # PDB only
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Visualize SE(3) frames as backbone proteins")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--no-render", action="store_true", help="Generate PDBs only, skip PyMOL")
parser.add_argument(
    "--lengths", type=int, nargs="+", default=[46, 70], help="Backbone lengths to sample"
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch  # noqa: E402

VIS_DIR = ROOT / "outputs" / "frame_visualizations"
PDB_DIR = ROOT / "data" / "pdb"

# ---------------------------------------------------------------------------
# Ideal backbone geometry (bond lengths in Angstroms, angles in radians)
# ---------------------------------------------------------------------------

# Distances
D_N_CA = 1.458
D_CA_C = 1.523
D_C_O = 1.231
D_CA_CB = 1.521

# Angles
A_N_CA_C = math.radians(111.0)
A_CA_C_O = math.radians(120.5)
A_N_CA_CB = math.radians(110.5)

# Dihedral placement: CB is roughly tetrahedral from N-CA-C
# We place it using the frame's local coordinate system


def backbone_from_frames(frames_rots, frames_trans):
    """Reconstruct N, CA, C, O, CB from SE(3) frames using ideal geometry.

    Args:
        frames_rots: [L, 3, 3] rotation matrices (columns = local x, y, z axes)
        frames_trans: [L, 3] translations (= CA positions)

    Returns:
        dict of atom coords, each [L, 3]

    Frame convention (from compute_local_frame):
        x = C - CA direction (normalized)
        y = z cross x
        z = x cross (N - CA) direction (normalized)
    So:
        C is along +x from CA
        N is roughly in the x-y plane (negative x, positive y)
        O is roughly in the x-z plane from C
        CB is roughly in the -x, -y, +z direction from CA (tetrahedral)
    """
    L = frames_trans.shape[0]
    R = frames_rots  # [L, 3, 3]
    t = frames_trans  # [L, 3]

    def apply_frame(local_vec):
        """Apply frame rotation + translation to local vector."""
        return torch.einsum("lij,lj->li", R, local_vec) + t

    # CA = frame origin
    coords_CA = t.clone()

    # C along local x-axis at ideal distance
    c_local = torch.zeros(L, 3, device=t.device)
    c_local[:, 0] = D_CA_C
    coords_C = apply_frame(c_local)

    # N in local x-y plane: negative x, positive y at ideal angle
    n_local = torch.zeros(L, 3, device=t.device)
    n_local[:, 0] = -D_N_CA * math.cos(math.pi - A_N_CA_C)
    n_local[:, 1] = D_N_CA * math.sin(math.pi - A_N_CA_C)
    coords_N = apply_frame(n_local)

    # O placed from C: roughly in x-z plane, ~120.5 degree angle from CA-C
    # In local frame of CA: O is beyond C, tilted into z
    o_local = torch.zeros(L, 3, device=t.device)
    o_local[:, 0] = D_CA_C + D_C_O * math.cos(math.pi - A_CA_C_O)
    o_local[:, 2] = D_C_O * math.sin(math.pi - A_CA_C_O)
    coords_O = apply_frame(o_local)

    # CB: tetrahedral placement opposite to C and N
    # Use the cross product of (N-CA) and (C-CA) directions to get CB direction
    # In local frame: CB is roughly along -x, -y, +z (bisector opposite to N and C)
    # Standard CB placement: angle N-CA-CB ~ 110.5 degrees
    # CB is placed such that it's roughly tetrahedral
    cb_local = torch.zeros(L, 3, device=t.device)
    # Tetrahedral: CB is at ~110.5 from both N and C
    # Simple approach: place in -y, +z quadrant
    cb_local[:, 0] = -D_CA_CB * math.cos(A_N_CA_CB) * 0.33  # slight -x
    cb_local[:, 1] = -D_CA_CB * math.sin(A_N_CA_CB) * 0.71  # -y (opposite side from N)
    cb_local[:, 2] = D_CA_CB * 0.62  # +z (out of NC plane)
    coords_CB = apply_frame(cb_local)

    return {
        "N": coords_N,
        "CA": coords_CA,
        "C": coords_C,
        "O": coords_O,
        "CB": coords_CB,
    }


def write_backbone_pdb(path, coords, chain="A", resname="ALA"):
    """Write full backbone PDB (N, CA, C, O, CB per residue).

    Args:
        path: output file path
        coords: dict with N, CA, C, O, CB each [L, 3]
        chain: chain ID
        resname: 3-letter residue name (ALA for virtual, or actual)
    """
    L = coords["CA"].shape[0]
    atom_order = ["N", "CA", "C", "O", "CB"]
    # For glycine, skip CB (but we use ALA for virtual proteins)

    # Handle per-residue resnames
    if isinstance(resname, list):
        resnames = resname
    else:
        resnames = [resname] * L

    with open(path, "w") as f:
        atom_num = 1
        for i in range(L):
            for atom_name in atom_order:
                if atom_name == "CB" and resnames[i] == "GLY":
                    continue  # Glycine has no CB
                x, y, z = coords[atom_name][i].tolist()
                # PDB ATOM format
                f.write(
                    f"ATOM  {atom_num:5d}  {atom_name:<3s} {resnames[i]:>3s} {chain}"
                    f"{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {atom_name[0]:>1s}\n"
                )
                atom_num += 1
        # Write CONECT for backbone connectivity
        f.write("TER\n")
        f.write("END\n")


def bond_stats(ca_coords):
    """Compute CA-CA consecutive bond distance statistics."""
    diffs = ca_coords[1:] - ca_coords[:-1]
    dists = torch.norm(diffs, dim=-1)
    return {
        "mean": dists.mean().item(),
        "std": dists.std().item(),
        "min": dists.min().item(),
        "max": dists.max().item(),
    }


# ---------------------------------------------------------------------------
# Generate structures
# ---------------------------------------------------------------------------


def sample_rfdiffusion(lengths):
    """Sample backbones from RFDiffusion and reconstruct full atoms from frames."""
    from rfdiffusion.model import RFDiffusion, sample

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best checkpoint
    ckpt_path = ROOT / "outputs" / "rfdiffusion_single" / "final_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "outputs" / "rfdiffusion" / "final_model.pt"
    if not ckpt_path.exists():
        print("No RFDiffusion checkpoint found")
        return []

    print(f"  Loading RFDiffusion from {ckpt_path}")
    model = RFDiffusion().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    for L in lengths:
        print(f"\n  Sampling L={L} backbone...")
        with torch.no_grad():
            trajectory = sample(
                network=model.network,
                diffusion=model.diffusion,
                num_residues=L,
                num_steps=100,
                device=device,
            )

        # Final frames from trajectory
        final_frames = trajectory[-1]  # RigidTransform [L, ...]
        rots = final_frames.rots.cpu()  # [L, 3, 3]
        trans = final_frames.trans.cpu()  # [L, 3]

        # Reconstruct full backbone from frames
        coords = backbone_from_frames(rots, trans)
        stats = bond_stats(coords["CA"])
        print(
            f"  L={L}: CA-CA bond mean={stats['mean']:.2f}A "
            f"std={stats['std']:.2f}A (ideal=3.8A)"
        )

        # Write full backbone PDB
        pdb_path = VIS_DIR / f"rfd_backbone_L{L}.pdb"
        write_backbone_pdb(pdb_path, coords)
        print(f"  Wrote {pdb_path}")

        # Also write CA-only PDB for comparison
        ca_path = VIS_DIR / f"rfd_ca_only_L{L}.pdb"
        with open(ca_path, "w") as f:
            for i in range(L):
                x, y, z = coords["CA"][i].tolist()
                f.write(
                    f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")

        results.append({"L": L, "stats": stats, "coords": coords})

    return results


def denoise_native():
    """Denoise 1CRN from low noise to show frame reconstruction quality."""
    from rfdiffusion.model import RFDiffusion, compute_local_frame
    from rfdiffusion.train import parse_pdb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "outputs" / "rfdiffusion_single" / "final_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "outputs" / "rfdiffusion" / "final_model.pt"
    if not ckpt_path.exists():
        return None

    crn_path = PDB_DIR / "1CRN.pdb"
    if not crn_path.exists():
        return None

    model = RFDiffusion().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    protein = parse_pdb(crn_path)
    if protein is None:
        return None

    N = protein["coords_N"].unsqueeze(0).to(device)
    CA = protein["coords_CA"].unsqueeze(0).to(device)
    C = protein["coords_C"].unsqueeze(0).to(device)
    center = CA.mean(dim=1, keepdim=True)
    N, CA, C = N - center, CA - center, C - center

    true_frames = compute_local_frame(N, CA, C)
    L = CA.shape[1]

    # Denoise from low noise (t=0.05)
    t_low = torch.full((1, L), 0.05, device=device)
    noisy, _ = model.diffusion.forward_marginal(true_frames, t_low)
    with torch.no_grad():
        pred = model.network(noisy, torch.tensor([0.05], device=device))

    # Reconstruct backbone from predicted frames
    pred_rots = pred.rots.squeeze(0).cpu()
    pred_trans = pred.trans.squeeze(0).cpu()
    pred_coords = backbone_from_frames(pred_rots, pred_trans)

    # Also reconstruct from true frames for comparison
    true_rots = true_frames.rots.squeeze(0).cpu()
    true_trans = true_frames.trans.squeeze(0).cpu()
    true_coords = backbone_from_frames(true_rots, true_trans)

    # RMSD
    pred_ca = pred_coords["CA"]
    true_ca = true_coords["CA"]
    pred_c = pred_ca - pred_ca.mean(dim=0, keepdim=True)
    true_c = true_ca - true_ca.mean(dim=0, keepdim=True)
    rmsd = ((pred_c - true_c) ** 2).sum(dim=-1).mean().sqrt().item()
    print(f"\n  1CRN denoise (t=0.05): CA-RMSD={rmsd:.2f}A")

    # Write denoised backbone
    write_backbone_pdb(VIS_DIR / "rfd_denoise_1CRN_backbone.pdb", pred_coords)
    # Write true backbone from frames
    write_backbone_pdb(VIS_DIR / "rfd_true_1CRN_backbone.pdb", true_coords)
    print(f"  Wrote denoised and true backbone PDBs")

    return {"rmsd": rmsd, "L": L}


# ---------------------------------------------------------------------------
# PyMOL render script
# ---------------------------------------------------------------------------


def write_render_script(sample_results, denoise_result):
    """Generate PyMOL script to render backbone structures with CB."""
    script_path = VIS_DIR / "render_frames.py"
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
        "cmd.set('cartoon_fancy_helices', 1)",
        "cmd.set('cartoon_smooth_loops', 1)",
        "cmd.set('sphere_scale', 0.3)",
        "cmd.set('stick_radius', 0.12)",
        "",
    ]

    # Render each sampled backbone
    for res in sample_results:
        L = res["L"]
        stats = res["stats"]
        pdb_path = os.path.join(vis_abs, f"rfd_backbone_L{L}.pdb")
        png_path = os.path.join(vis_abs, f"rfd_backbone_L{L}.png")
        lines += [
            f"# --- RFDiffusion backbone L={L} (CA-CA bond={stats['mean']:.2f}A) ---",
            f"cmd.load({pdb_path!r}, 'bbone')",
            "",
            "# Show backbone as sticks, CB as spheres",
            "cmd.hide('everything', 'bbone')",
            "cmd.show('sticks', 'bbone and (name N+CA+C+O)')",
            "cmd.show('spheres', 'bbone and name CB')",
            "cmd.set('sphere_scale', 0.25, 'bbone and name CB')",
            "",
            "# Color: backbone by rainbow, CB in orange",
            "cmd.spectrum('resi', 'rainbow', 'bbone and (name N+CA+C+O)')",
            "cmd.color('orange', 'bbone and name CB')",
            "",
            "# Assign secondary structure from geometry, then show cartoon",
            "cmd.dss('bbone')",
            "cmd.show('cartoon', 'bbone')",
            "cmd.set('cartoon_transparency', 0.5, 'bbone')",
            "cmd.spectrum('resi', 'rainbow', 'bbone')",
            "",
            "cmd.zoom('bbone', buffer=5)",
            "cmd.ray(1600, 1200)",
            f"cmd.png({png_path!r}, dpi=300)",
            f"print(f'Saved {png_path}')",
            "cmd.delete('all')",
            "",
        ]

    # Render denoise overlay
    if denoise_result is not None:
        pred_pdb = os.path.join(vis_abs, "rfd_denoise_1CRN_backbone.pdb")
        true_pdb = os.path.join(vis_abs, "rfd_true_1CRN_backbone.pdb")
        png_path = os.path.join(vis_abs, "rfd_denoise_overlay.png")
        rmsd = denoise_result["rmsd"]
        lines += [
            f"# --- 1CRN denoise overlay (RMSD={rmsd:.2f}A) ---",
            f"cmd.load({pred_pdb!r}, 'denoised')",
            f"cmd.load({true_pdb!r}, 'native')",
            "",
            "# Align",
            "try:",
            "    rms = cmd.align('denoised and name CA', 'native and name CA')[0]",
            "    print(f'Alignment RMSD: {rms:.2f}A')",
            "except:",
            "    print('Alignment failed')",
            "",
            "# Native: assign SS, cartoon cyan, transparent",
            "cmd.hide('everything', 'native')",
            "cmd.dss('native')",
            "cmd.show('cartoon', 'native')",
            "cmd.color('cyan', 'native')",
            "cmd.set('cartoon_transparency', 0.3, 'native')",
            "",
            "# Denoised: sticks green, CB orange spheres",
            "cmd.hide('everything', 'denoised')",
            "cmd.show('sticks', 'denoised and (name N+CA+C+O)')",
            "cmd.show('spheres', 'denoised and name CB')",
            "cmd.set('sphere_scale', 0.25, 'denoised and name CB')",
            "cmd.color('green', 'denoised and (name N+CA+C+O)')",
            "cmd.color('orange', 'denoised and name CB')",
            "",
            "# Also show denoised cartoon for SS comparison",
            "cmd.dss('denoised')",
            "cmd.show('cartoon', 'denoised')",
            "cmd.set('cartoon_transparency', 0.5, 'denoised')",
            "cmd.color('green', 'denoised')",
            "",
            "cmd.zoom('native', buffer=5)",
            "cmd.ray(1600, 1200)",
            f"cmd.png({png_path!r}, dpi=300)",
            f"print(f'Saved {png_path}')",
            "cmd.delete('all')",
            "",
        ]

    lines.append("cmd.quit()")

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return script_path


def run_pymol(script_path):
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
        print("\nPyMOL not found. Install with:")
        print("  conda create -n pymol -c conda-forge pymol-open-source python=3.12")
        print(f"Or run manually: python {script_path}")
        return

    print(f"\nRunning PyMOL: {pymol_python} {script_path}")
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
    print("SE(3) Frame Visualization: RFDiffusion")
    print("=" * 60)

    print("\n--- Sampling de novo backbones ---")
    sample_results = sample_rfdiffusion(args.lengths)

    print("\n--- Denoising native 1CRN ---")
    denoise_result = denoise_native()

    if not sample_results and denoise_result is None:
        print("No structures generated. Check RFDiffusion checkpoint exists.")
        return

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for res in sample_results:
        s = res["stats"]
        print(
            f"  L={res['L']}: CA-CA bond {s['mean']:.2f} +/- {s['std']:.2f}A "
            f"[{s['min']:.2f}, {s['max']:.2f}]"
        )
    if denoise_result:
        print(f"  1CRN denoise: CA-RMSD={denoise_result['rmsd']:.2f}A")

    print(f"\nPDB files written to {VIS_DIR}/")

    # Generate PyMOL script
    script_path = write_render_script(sample_results, denoise_result)
    print(f"Render script: {script_path}")

    if args.no_render:
        print("Skipping rendering (--no-render)")
        return

    run_pymol(script_path)

    # List outputs
    print("\nOutput files:")
    for f in sorted(VIS_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
