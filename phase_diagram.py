import os
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import glob
import sys
import time
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from LLG_solver import compare_phases


class HiddenPrints:
    """Suppress verbose output from LLG_solver during the sweep."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._devnull.close()


def _parse_zone_bound(raw_value):
    """Parse one numeric lattice-subzone boundary."""
    value = str(raw_value).strip()
    lowered = value.lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity", "max"}:
        return np.inf
    if lowered in {"-inf", "-infinity", "min"}:
        return -np.inf
    return float(value)



def parse_l_subzones(spec):
    """
    Parse optional lattice-size override zones.

    Format:
        "A,1.0,inf,256;H,2.0,2.5,192"

    Each entry is axis,min,max,L where axis can be:
        A or X  -> anisotropy / x-axis
        H or Y  -> magnetic field / y-axis
    """
    if spec is None:
        return []

    spec = str(spec).strip()
    if not spec:
        return []

    zones = []
    for zone_index, entry in enumerate(spec.split(";"), start=1):
        entry = entry.strip()
        if not entry:
            continue

        parts = [part.strip() for part in entry.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "Each lattice subzone must have exactly 4 comma-separated values: axis,min,max,L"
            )

        axis_raw, min_raw, max_raw, lattice_raw = parts
        axis = axis_raw.upper()
        if axis == "X":
            axis = "A"
        elif axis == "Y":
            axis = "H"

        if axis not in {"A", "H"}:
            raise ValueError(
                f"Unsupported lattice subzone axis '{axis_raw}' in entry {zone_index}. "
                "Use A/X for anisotropy or H/Y for magnetic field."
            )

        zone_min = _parse_zone_bound(min_raw)
        zone_max = _parse_zone_bound(max_raw)
        if zone_max < zone_min:
            raise ValueError(
                f"Lattice subzone entry {zone_index} has max < min: '{entry}'"
            )

        lattice_size = int(lattice_raw)
        if lattice_size < 2:
            raise ValueError(
                f"Lattice subzone entry {zone_index} must use L >= 2, got {lattice_size}."
            )

        zones.append(
            {
                "axis": axis,
                "min": float(zone_min),
                "max": float(zone_max),
                "L": lattice_size,
            }
        )

    return zones



def choose_lattice_size(h_scaled, a_scaled, default_L, l_subzones=None):
    """Return the lattice size for one sweep point, honoring optional override zones."""
    if not l_subzones:
        return int(default_L)

    for zone in l_subzones:
        coordinate = a_scaled if zone["axis"] == "A" else h_scaled
        if zone["min"] <= coordinate <= zone["max"]:
            return int(zone["L"])

    return int(default_L)


def get_output_root(output_root=None):
    """Resolve output root for local runs or cluster runs."""
    if output_root:
        return Path(output_root).expanduser()
    env_output_root = os.environ.get("SKYRMION_OUTPUT_ROOT")
    if env_output_root:
        return Path(env_output_root).expanduser()
    return Path(__file__).resolve().parent / "output"


def ensure_parent_dir(filepath):
    """Create parent directory for a file path if needed."""
    Path(filepath).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def get_phase_data_dir(output_root):
    return output_root / "LLG" / "Phase Diagram Data"


def get_graphs_dir(output_root):
    return output_root / "LLG" / "Graphs"


def generate_phase_diagram(n_H=26, n_A=33, L=32, output_root=None, show_plot=False, l_subzones=None):
    """
    Generates a phase diagram by scanning H and A.
    Saves the data and automatically calls the plotter.
    """
    output_root = get_output_root(output_root)

    H_vals = np.linspace(0, 2.5, n_H)
    A_vals = np.linspace(-1.5, 1.7, n_A)

    phase_grid = np.zeros((n_A, n_H))
    lattice_size_grid = np.full((n_A, n_H), int(L), dtype=int)
    phase_map = {"SkX": 0, "SC": 1, "SP": 2, "FM": 3}

    print("=== Starting Phase Diagram Generation ===")
    print(f"Total points: {n_H * n_A} (H resolution: {n_H}, A resolution: {n_A})")
    print(f"Using default lattice size L={L} for the sweep.")
    if l_subzones:
        print(f"Lattice override zones: {l_subzones}")
    print(f"Output root: {output_root}")
    print("-----------------------------------------")

    start_time = time.time()

    for i, a in enumerate(A_vals):
        for j, h in enumerate(H_vals):
            lattice_size = choose_lattice_size(
                h_scaled=float(h),
                a_scaled=float(a),
                default_L=L,
                l_subzones=l_subzones,
            )
            lattice_size_grid[i, j] = lattice_size

            sys.stdout.write(
                f"\rComputing Point {i * n_H + j + 1}/{n_H * n_A} | H = {h:.2f}, A = {a:.2f}, L = {lattice_size} ... "
            )
            sys.stdout.flush()

            try:
                with HiddenPrints():
                    winner, _ = compare_phases(
                        H_scaled=h,
                        A_scaled=a,
                        L=lattice_size,
                        plot_ansatz=False,
                        live_plot=False,
                        save_outputs=False,
                        output_root=output_root,
                    )

                phase_grid[i, j] = phase_map.get(winner, -1)

            except Exception as e:
                print(f"\nFailed to converge at H={h}, A={a}, L={lattice_size}: {e}")
                phase_grid[i, j] = -1

    elapsed = time.time() - start_time
    print(f"\n\nSweep finished in {elapsed:.2f} seconds!")

    total_pts = n_H * n_A
    out_path = get_phase_data_dir(output_root) / f"phase_diagram_L{L}_{total_pts}.npz"
    ensure_parent_dir(out_path)
    np.savez(
        out_path,
        grid=phase_grid,
        H_vals=H_vals,
        A_vals=A_vals,
        lattice_size_grid=lattice_size_grid,
        default_lattice_size=np.array(int(L)),
        l_subzones_json=np.array(json.dumps(l_subzones or [], sort_keys=True)),
    )

    print(f"Data bundled and saved to '{out_path}'. Generating plot...")
    plot_phase_diagram(
        phase_grid,
        H_vals,
        A_vals,
        out_name=f"phase_diagram_L{L}_{total_pts}.png",
        output_root=output_root,
        show_plot=show_plot,
    )


def plot_phase_diagram(phase_grid, H_vals, A_vals, out_name="phase_diagram.png", output_root=None, show_plot=False):
    """
    Renders the integer grid as a clean, colored phase diagram.
    """
    output_root = get_output_root(output_root)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = ListedColormap(["#4C72B0", "#55A868", "#DD8452", "#EAEAF2"])
    cmap.set_under("#808080")

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    dH = H_vals[1] - H_vals[0] if len(H_vals) > 1 else 1.0
    dA = A_vals[1] - A_vals[0] if len(A_vals) > 1 else 1.0

    H_edges = np.append(H_vals - dH / 2, H_vals[-1] + dH / 2)
    A_edges = np.append(A_vals - dA / 2, A_vals[-1] + dA / 2)

    A_mesh, H_mesh = np.meshgrid(A_edges, H_edges)

    c = ax.pcolormesh(A_mesh, H_mesh, phase_grid.T, cmap=cmap, norm=norm, edgecolors="none")

    cbar = plt.colorbar(c, ax=ax, ticks=[0, 1, 2, 3], pad=0.03)
    cbar.ax.set_yticklabels(
        [
            "Skyrmion Lattice (SkX)",
            "Square Cell (SC)",
            "Spiral Phase (SP)",
            "Ferromagnetic (FM)",
        ]
    )
    cbar.set_label("Ground State Phase", rotation=270, labelpad=25, fontsize=13)

    ax.set_xlabel("Scaled Anisotropy ($A_s$)", fontsize=14, labelpad=10)
    ax.set_ylabel("Scaled Magnetic Field ($H$)", fontsize=14, labelpad=10)
    ax.set_title("Topological Magnetic Phase Diagram", fontsize=16, pad=20)
    ax.grid(color="white", alpha=0.5, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    save_path = get_graphs_dir(output_root) / out_name
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=300)
    print(f"Saved high-res plot to '{save_path}'")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Skyrmion Phase Diagram")
    parser.add_argument("--nH", type=int, default=26, help="Number of points along the H axis")
    parser.add_argument("--nA", type=int, default=33, help="Number of points along the A axis")
    parser.add_argument("--L", type=int, default=32, help="Lattice size for relaxation")
    parser.add_argument(
        "--l-subzones",
        type=str,
        default=None,
        help=(
            "Optional lattice-size overrides in the format "
            "'A,1.0,inf,256;H,2.0,2.5,192'. "
            "Use A/X for anisotropy (x-axis) and H/Y for magnetic field (y-axis). "
            "The same setting can also be passed via the L_SUBZONES environment variable."
        ),
    )
    parser.add_argument("--recompute", action="store_true", help="Force recomputation even if data exists")
    parser.add_argument("--load-file", type=str, default=None, help="Specific .npz phase diagram file to load")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for outputs, e.g. $SCRATCH/skyrmion_runs")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving")

    args = parser.parse_args()
    l_subzones_spec = args.l_subzones
    if l_subzones_spec is None:
        l_subzones_spec = os.environ.get("L_SUBZONES", "")
    l_subzones = parse_l_subzones(l_subzones_spec)

    output_root = get_output_root(args.output_root)

    if args.load_file:
        sel_file = Path(args.load_file).expanduser()
        if not sel_file.is_absolute():
            sel_file = get_phase_data_dir(output_root) / sel_file
        if not sel_file.exists():
            raise FileNotFoundError(f"Cannot find phase diagram data file: {sel_file}")

        print(f"Loading existing data from {sel_file.name}...")
        data = np.load(sel_file)
        grid = data["grid"]
        H = data["H_vals"]
        A = data["A_vals"]
        if "l_subzones_json" in data:
            loaded_subzones = json.loads(str(data["l_subzones_json"]))
            if loaded_subzones:
                print(f"Stored lattice override zones: {loaded_subzones}")
        out_png = sel_file.name.replace(".npz", ".png")
        plot_phase_diagram(grid, H, A, out_name=out_png, output_root=output_root, show_plot=args.show)

    elif args.recompute:
        generate_phase_diagram(
            n_H=args.nH,
            n_A=args.nA,
            L=args.L,
            output_root=output_root,
            show_plot=args.show,
            l_subzones=l_subzones,
        )

    else:
        existing_files = sorted(glob.glob(str(get_phase_data_dir(output_root) / "*.npz")))
        if existing_files:
            sel_file = Path(existing_files[-1])
            print(
                f"Found existing phase diagram data. Loading most recent file '{sel_file.name}'. "
                "Use --recompute to generate a new diagram or --load-file to pick a specific file."
            )
            data = np.load(sel_file)
            grid = data["grid"]
            H = data["H_vals"]
            A = data["A_vals"]
            out_png = sel_file.name.replace(".npz", ".png")
            plot_phase_diagram(grid, H, A, out_name=out_png, output_root=output_root, show_plot=args.show)
        else:
            generate_phase_diagram(
                n_H=args.nH,
                n_A=args.nA,
                L=args.L,
                output_root=output_root,
                show_plot=args.show,
                l_subzones=l_subzones,
            )