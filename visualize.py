"""Render PDB protein structures with PyMOL (cartoon + SS coloring).

Usage:
    python visualize.py --pdb_dir data/pdb                      # render all PDBs
    python visualize.py --pdb_dir data/pdb --output_dir out/     # custom output
    python visualize.py --pdb_dir data/pdb --no-render           # generate script only
"""

import argparse
import os
import shutil
import subprocess

# ---------------------------------------------------------------------------
# PyMOL render script generation (uses Python API for better rendering)
# ---------------------------------------------------------------------------


def write_render_script(pdb_files: list[str], output_dir: str) -> str:
    """Generate a Python script that uses PyMOL's API to render all PDB files."""
    script_path = os.path.join(output_dir, "render.py")

    pairs = []
    for pdb_file in pdb_files:
        name = os.path.splitext(os.path.basename(pdb_file))[0]
        png_file = os.path.join(os.path.abspath(output_dir), f"{name}.png")
        pairs.append((os.path.abspath(pdb_file), png_file, name))

    with open(script_path, "w") as f:
        f.write("import pymol\n")
        f.write("from pymol import cmd\n\n")
        f.write("pymol.finish_launching(['pymol', '-cq'])\n\n")
        # Global rendering settings
        f.write("cmd.set('cartoon_fancy_helices', 1)\n")
        f.write("cmd.set('cartoon_smooth_loops', 1)\n")
        f.write("cmd.set('cartoon_tube_radius', 0.2)\n")
        f.write("cmd.set('cartoon_loop_radius', 0.2)\n")
        f.write("cmd.set('antialias', 2)\n")
        f.write("cmd.set('specular', 0.3)\n")
        f.write("cmd.set('spec_reflect', 0.5)\n")
        f.write("cmd.set('depth_cue', 1)\n")
        f.write("cmd.set('fog_start', 0.4)\n")
        f.write("cmd.set('ray_shadow', 1)\n")
        f.write("cmd.set('ray_opaque_background', 0)\n\n")
        for pdb_path, png_path, name in pairs:
            f.write(f"cmd.load({pdb_path!r}, {name!r})\n")
            f.write("cmd.hide('everything')\n")
            f.write("cmd.show('cartoon')\n")
            f.write("cmd.color('green', 'ss l')\n")
            f.write("cmd.color('marine', 'ss h')\n")
            f.write("cmd.color('orange', 'ss s')\n")
            f.write(f"cmd.zoom({name!r}, buffer=5)\n")
            f.write("cmd.ray(1200, 900)\n")
            f.write(f"cmd.png({png_path!r}, dpi=300)\n")
            f.write(f"cmd.delete({name!r})\n\n")
        f.write("cmd.quit()\n")

    return script_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Render PDB structures with PyMOL")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory with PDB files")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/pdb", help="Output directory for PNGs"
    )
    parser.add_argument("--no-render", action="store_true", help="Only generate render script")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = sorted(
        os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith(".pdb")
    )
    if not pdb_files:
        print(f"No PDB files found in {args.pdb_dir}")
        return
    print(f"Found {len(pdb_files)} PDB files in {args.pdb_dir}")
    for f in pdb_files:
        print(f"  {f}")

    render_script = write_render_script(pdb_files, args.output_dir)
    print(f"Render script: {render_script}")

    if args.no_render:
        print("Skipping rendering (--no-render)")
        return

    # Find conda pymol python: check for pymol conda env, then PATH
    pymol_python = None
    conda_python = os.path.expanduser("~/.conda/envs/pymol/bin/python")
    if os.path.isfile(conda_python):
        pymol_python = conda_python
    else:
        pymol_bin = shutil.which("pymol")
        if pymol_bin:
            pymol_python = os.path.join(os.path.dirname(pymol_bin), "python")
            if not os.path.isfile(pymol_python):
                pymol_python = None
    if pymol_python is None:
        print("PyMOL not found. Install with:")
        print("  conda create -n pymol -c conda-forge pymol-open-source python=3.12")
        print(f"Or run manually: python {render_script}")
        return

    print(f"Using PyMOL python: {pymol_python}")
    result = subprocess.run(
        [pymol_python, render_script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode == 0:
        print("Rendering complete!")
        for pdb_file in pdb_files:
            name = os.path.splitext(os.path.basename(pdb_file))[0]
            png = os.path.join(args.output_dir, f"{name}.png")
            if os.path.exists(png):
                print(f"  {png}")
    else:
        print(f"PyMOL error (exit {result.returncode}):")
        if result.stdout:
            print(result.stdout[:500])
        if result.stderr:
            print(result.stderr[:500])


if __name__ == "__main__":
    main()
