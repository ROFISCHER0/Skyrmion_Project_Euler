import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from LLG_solver import compare_phases, ensure_parent_dir, get_output_root, warm_up_numba_kernels

PHASE_MAP = {"SkX": 0, "SC": 1, "SP": 2, "FM": 3}
PHASE_NAMES = {value: key for key, value in PHASE_MAP.items()}
ENERGY_PHASES = ("SkX", "SC", "SP", "FM")
DEFAULT_H_RANGE = (0.0, 2.5)
DEFAULT_A_RANGE = (-1.5, 1.7)
INVALID_PHASE_ID = -1


class HiddenPrints:
    """Suppress verbose solver output inside worker processes."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._devnull.close()


def slugify_float(value):
    """Convert floats into filename-safe ASCII snippets."""
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def build_run_name(n_H, n_A, L, H_min, H_max, A_min, A_max):
    """Create a deterministic run identifier suitable for local and cluster runs."""
    return (
        f"phase_diagram_L{L}_H{n_H}_A{n_A}"
        f"_Hr{slugify_float(H_min)}_{slugify_float(H_max)}"
        f"_Ar{slugify_float(A_min)}_{slugify_float(A_max)}"
    )


def get_output_paths(output_root, run_name, chunk_index=None, chunk_count=None):
    """Resolve all output paths for a full run or for one chunk."""
    output_root = get_output_root(output_root)
    data_dir = output_root / "LLG" / "PhaseDiagramData"
    plot_dir = output_root / "LLG" / "Graphs"
    chunk_dir = output_root / "LLG" / "PhaseDiagramChunks"

    if chunk_index is None:
        stem = run_name
        base_dir = data_dir
    else:
        stem = f"{run_name}_chunk{chunk_index:04d}of{chunk_count:04d}"
        base_dir = chunk_dir

    return {
        "npz": base_dir / f"{stem}.npz",
        "csv": base_dir / f"{stem}.csv",
        "json": base_dir / f"{stem}.json",
        "png": plot_dir / f"{run_name}.png",
    }


def create_axes(n_H, n_A, H_min, H_max, A_min, A_max):
    """Build the H and A sweep axes."""
    if n_H < 1 or n_A < 1:
        raise ValueError("n_H and n_A must both be positive integers")
    H_vals = np.linspace(H_min, H_max, n_H)
    A_vals = np.linspace(A_min, A_max, n_A)
    return H_vals, A_vals


def build_point_tasks(H_vals, A_vals, L, max_dt, cfl_factor):
    """Generate independent sweep points that can be distributed across workers or job-array chunks."""
    tasks = []
    n_H = len(H_vals)
    for a_index, a_scaled in enumerate(A_vals):
        for h_index, h_scaled in enumerate(H_vals):
            tasks.append(
                (
                    a_index * n_H + h_index,
                    a_index,
                    h_index,
                    float(h_scaled),
                    float(a_scaled),
                    int(L),
                    float(max_dt),
                    float(cfl_factor),
                )
            )
    return tasks


def select_chunk(tasks, chunk_index=None, chunk_count=1):
    """Split the full point list into contiguous chunks for Slurm array jobs."""
    if chunk_index is None:
        return tasks
    if chunk_count < 1:
        raise ValueError("chunk_count must be at least 1")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError(f"chunk_index must be between 0 and {chunk_count - 1}")

    total = len(tasks)
    start = (total * chunk_index) // chunk_count
    end = (total * (chunk_index + 1)) // chunk_count
    return tasks[start:end]


def empty_result_bundle(n_A, n_H):
    """Allocate dense arrays used by full or chunked phase-diagram outputs."""
    bundle = {
        "grid": np.full((n_A, n_H), INVALID_PHASE_ID, dtype=np.int16),
        "mask": np.zeros((n_A, n_H), dtype=bool),
        "topological_charge_grid": np.full((n_A, n_H), np.nan, dtype=np.float64),
        "ax_grid": np.full((n_A, n_H), np.nan, dtype=np.float64),
        "ay_grid": np.full((n_A, n_H), np.nan, dtype=np.float64),
        "records": [],
    }
    for phase_name in ENERGY_PHASES:
        bundle[f"energy_{phase_name.lower()}_grid"] = np.full((n_A, n_H), np.nan, dtype=np.float64)
    return bundle


def get_pool_context():
    """Prefer fork on POSIX because it avoids extra import overhead for each worker."""
    if os.name == "posix":
        for method in ("fork", "forkserver", "spawn"):
            try:
                return mp.get_context(method)
            except ValueError:
                continue
    return mp.get_context("spawn")


def sanitize_error(exc):
    """Keep error messages compact and single-line for CSV output."""
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def compute_point(task):
    """Worker entry point for one (H, A) phase-diagram point."""
    (
        point_index,
        a_index,
        h_index,
        h_scaled,
        a_scaled,
        L,
        max_dt,
        cfl_factor,
    ) = task

    start_time = time.time()
    record = {
        "point_index": int(point_index),
        "a_index": int(a_index),
        "h_index": int(h_index),
        "H_scaled": float(h_scaled),
        "A_scaled": float(a_scaled),
        "phase_id": INVALID_PHASE_ID,
        "phase_name": "ERROR",
        "status": "error",
        "error": "",
        "elapsed_seconds": np.nan,
        "winner_topological_charge": np.nan,
        "ax": np.nan,
        "ay": np.nan,
    }
    for phase_name in ENERGY_PHASES:
        record[f"energy_{phase_name}"] = np.nan

    try:
        with HiddenPrints():
            winner, results, details = compare_phases(
                H_scaled=h_scaled,
                A_scaled=a_scaled,
                L=L,
                plot_ansatz=False,
                live_plot=False,
                plot_groundstate=False,
                save_outputs=False,
                output_root=None,
                max_dt=max_dt,
                cfl_factor=cfl_factor,
                return_details=True,
            )

        record["phase_name"] = winner
        record["phase_id"] = PHASE_MAP.get(winner, INVALID_PHASE_ID)
        record["status"] = "ok"
        record["winner_topological_charge"] = float(details["topological_charge"])
        record["ax"] = float(details["ax"])
        record["ay"] = float(details["ay"])

        candidate_energies = details.get("candidate_energies", results)
        for phase_name in ENERGY_PHASES:
            if phase_name in candidate_energies:
                record[f"energy_{phase_name}"] = float(candidate_energies[phase_name])

    except Exception as exc:
        record["error"] = sanitize_error(exc)

    record["elapsed_seconds"] = time.time() - start_time
    return record


def integrate_record(bundle, record):
    """Insert one computed point into dense result arrays."""
    a_index = record["a_index"]
    h_index = record["h_index"]
    bundle["grid"][a_index, h_index] = record["phase_id"]
    bundle["mask"][a_index, h_index] = True
    bundle["topological_charge_grid"][a_index, h_index] = record["winner_topological_charge"]
    bundle["ax_grid"][a_index, h_index] = record["ax"]
    bundle["ay_grid"][a_index, h_index] = record["ay"]

    for phase_name in ENERGY_PHASES:
        bundle[f"energy_{phase_name.lower()}_grid"][a_index, h_index] = record[f"energy_{phase_name}"]

    bundle["records"].append(record)


def print_progress(completed, total, start_time):
    """Emit a lightweight progress message suitable for batch logs."""
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else float("inf")
    if np.isfinite(remaining):
        print(
            f"Completed {completed}/{total} points "
            f"({100.0 * completed / total:.1f}%) | elapsed {elapsed:.1f}s | eta {remaining:.1f}s"
        )
    else:
        print(f"Completed {completed}/{total} points ({100.0 * completed / total:.1f}%) | elapsed {elapsed:.1f}s")


def sort_records(records):
    """Keep CSV rows ordered on the original grid traversal."""
    return sorted(records, key=lambda item: (item["a_index"], item["h_index"]))


def write_records_csv(records, csv_path):
    """Write per-point outputs that are easy to inspect and merge on the cluster."""
    fieldnames = [
        "point_index",
        "a_index",
        "h_index",
        "H_scaled",
        "A_scaled",
        "phase_id",
        "phase_name",
        "status",
        "error",
        "elapsed_seconds",
        "winner_topological_charge",
        "ax",
        "ay",
    ] + [f"energy_{phase_name}" for phase_name in ENERGY_PHASES]

    ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sort_records(records):
            writer.writerow(record)


def write_metadata_json(metadata, json_path):
    """Save a human-readable metadata summary beside the binary arrays."""
    ensure_parent_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def save_result_bundle(bundle, H_vals, A_vals, metadata, npz_path, csv_path, json_path):
    """Persist a full or chunked result bundle."""
    ensure_parent_dir(npz_path)
    np.savez(
        npz_path,
        grid=bundle["grid"],
        mask=bundle["mask"],
        H_vals=H_vals,
        A_vals=A_vals,
        topological_charge_grid=bundle["topological_charge_grid"],
        ax_grid=bundle["ax_grid"],
        ay_grid=bundle["ay_grid"],
        energy_skx_grid=bundle["energy_skx_grid"],
        energy_sc_grid=bundle["energy_sc_grid"],
        energy_sp_grid=bundle["energy_sp_grid"],
        energy_fm_grid=bundle["energy_fm_grid"],
        run_name=np.array(metadata["run_name"]),
        lattice_size=np.array(metadata["lattice_size"]),
        n_H=np.array(metadata["n_H"]),
        n_A=np.array(metadata["n_A"]),
        H_min=np.array(metadata["H_min"]),
        H_max=np.array(metadata["H_max"]),
        A_min=np.array(metadata["A_min"]),
        A_max=np.array(metadata["A_max"]),
        total_points=np.array(metadata["total_points"]),
        completed_points=np.array(metadata["completed_points"]),
        workers=np.array(metadata["workers"]),
        chunk_index=np.array(metadata["chunk_index"]),
        chunk_count=np.array(metadata["chunk_count"]),
        elapsed_seconds=np.array(metadata["elapsed_seconds"]),
        complete=np.array(metadata["complete"]),
        phase_names=np.array(list(PHASE_MAP.keys())),
        phase_ids=np.array(list(PHASE_MAP.values())),
    )
    write_records_csv(bundle["records"], csv_path)
    write_metadata_json(metadata, json_path)


def plot_phase_diagram(phase_grid, H_vals, A_vals, out_path, title="Topological Magnetic Phase Diagram", show_plot=False):
    """Render an integer phase grid into a cluster-safe PNG."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = ListedColormap(["#4C72B0", "#55A868", "#DD8452", "#EAEAF2"])
    cmap.set_under("#808080")

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    dH = H_vals[1] - H_vals[0] if len(H_vals) > 1 else 1.0
    dA = A_vals[1] - A_vals[0] if len(A_vals) > 1 else 1.0

    H_edges = np.append(H_vals - dH / 2.0, H_vals[-1] + dH / 2.0)
    A_edges = np.append(A_vals - dA / 2.0, A_vals[-1] + dA / 2.0)
    A_mesh, H_mesh = np.meshgrid(A_edges, H_edges)

    mesh = ax.pcolormesh(A_mesh, H_mesh, phase_grid.T, cmap=cmap, norm=norm, edgecolors="none")
    colorbar = plt.colorbar(mesh, ax=ax, ticks=[0, 1, 2, 3], pad=0.03)
    colorbar.ax.set_yticklabels(
        ["Skyrmion Lattice (SkX)", "Square Cell (SC)", "Spiral Phase (SP)", "Ferromagnetic (FM)"]
    )
    colorbar.set_label("Ground State Phase", rotation=270, labelpad=25, fontsize=13)

    ax.set_xlabel("Scaled Anisotropy ($A_s$)", fontsize=14, labelpad=10)
    ax.set_ylabel("Scaled Magnetic Field ($H$)", fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(color="white", alpha=0.5, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    ensure_parent_dir(out_path)
    plt.savefig(out_path, dpi=300)
    print(f"Saved phase-diagram plot to '{out_path}'")
    if show_plot:
        plt.show()
    plt.close(fig)


def build_metadata(run_name, L, H_vals, A_vals, workers, elapsed_seconds, completed_points, chunk_index=None, chunk_count=1):
    """Standardize metadata across fresh runs and merged runs."""
    total_points = int(len(H_vals) * len(A_vals))
    return {
        "run_name": run_name,
        "lattice_size": int(L),
        "n_H": int(len(H_vals)),
        "n_A": int(len(A_vals)),
        "H_min": float(H_vals[0]),
        "H_max": float(H_vals[-1]),
        "A_min": float(A_vals[0]),
        "A_max": float(A_vals[-1]),
        "total_points": total_points,
        "completed_points": int(completed_points),
        "workers": int(workers),
        "chunk_index": -1 if chunk_index is None else int(chunk_index),
        "chunk_count": int(chunk_count),
        "physics_model": "dkasper25_original_ansatzes",
        "elapsed_seconds": float(elapsed_seconds),
        "complete": bool(completed_points == total_points),
    }


def generate_phase_diagram(
    n_H=26,
    n_A=33,
    L=32,
    H_min=DEFAULT_H_RANGE[0],
    H_max=DEFAULT_H_RANGE[1],
    A_min=DEFAULT_A_RANGE[0],
    A_max=DEFAULT_A_RANGE[1],
    workers=1,
    output_root=None,
    run_name=None,
    chunk_index=None,
    chunk_count=1,
    max_dt=0.05,
    cfl_factor=0.25,
    save_plot=True,
):
    """
    Generate the phase diagram in a non-interactive, cluster-friendly way.
    Each (H, A) point is independent and can therefore be parallelized across CPU cores.
    """
    H_vals, A_vals = create_axes(n_H, n_A, H_min, H_max, A_min, A_max)
    run_name = run_name or build_run_name(
        n_H=n_H,
        n_A=n_A,
        L=L,
        H_min=H_min,
        H_max=H_max,
        A_min=A_min,
        A_max=A_max,
    )

    all_tasks = build_point_tasks(H_vals, A_vals, L, max_dt, cfl_factor)
    selected_tasks = select_chunk(all_tasks, chunk_index=chunk_index, chunk_count=chunk_count)

    print("=== Starting Phase Diagram Generation ===")
    print(f"Run name: {run_name}")
    print(f"Grid: {n_A} x {n_H}  |  Total points: {len(all_tasks)}")
    print(f"H range: [{H_min}, {H_max}]  |  A range: [{A_min}, {A_max}]")
    print(f"Lattice size: L={L}  |  Workers: {workers}")
    print("Physics model: original dkasper25 Hamiltonian and ansatz definitions")
    if chunk_index is not None:
        print(
            f"Chunk mode enabled: chunk {chunk_index + 1}/{chunk_count} "
            f"covering {len(selected_tasks)} points"
        )
    print("-----------------------------------------")

    bundle = empty_result_bundle(n_A, n_H)
    print("Warming up cached Numba kernels...")
    try:
        warm_up_numba_kernels()
    except Exception as exc:
        print(f"Numba warm-up skipped; continuing without cached precompile: {exc}")
    start_time = time.time()
    total_selected = len(selected_tasks)

    if total_selected == 0:
        raise ValueError("No sweep points were assigned to this chunk")

    progress_every = max(1, total_selected // 20)

    effective_workers = workers
    pool = None

    if workers == 1:
        iterator = map(compute_point, selected_tasks)
    else:
        chunksize = max(1, total_selected // max(workers * 4, 1))
        try:
            pool = get_pool_context().Pool(processes=workers)
            iterator = pool.imap(compute_point, selected_tasks, chunksize=chunksize)
        except (OSError, PermissionError) as exc:
            effective_workers = 1
            print(f"Multiprocessing unavailable in this environment, falling back to 1 worker: {exc}")
            iterator = map(compute_point, selected_tasks)

    try:
        for completed, record in enumerate(iterator, start=1):
            integrate_record(bundle, record)
            if completed % progress_every == 0 or completed == total_selected:
                print_progress(completed, total_selected, start_time)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed_seconds = time.time() - start_time
    completed_points = int(np.count_nonzero(bundle["mask"]))
    metadata = build_metadata(
        run_name=run_name,
        L=L,
        H_vals=H_vals,
        A_vals=A_vals,
        workers=effective_workers,
        elapsed_seconds=elapsed_seconds,
        completed_points=completed_points,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    paths = get_output_paths(output_root, run_name, chunk_index=chunk_index, chunk_count=chunk_count)
    save_result_bundle(bundle, H_vals, A_vals, metadata, paths["npz"], paths["csv"], paths["json"])

    print(f"Sweep finished in {elapsed_seconds:.2f} seconds")
    print(f"Saved data bundle to '{paths['npz']}'")
    print(f"Saved point-wise table to '{paths['csv']}'")
    print(f"Saved metadata summary to '{paths['json']}'")

    if save_plot and chunk_index is None and metadata["complete"]:
        plot_phase_diagram(bundle["grid"], H_vals, A_vals, paths["png"], show_plot=False)

    return paths["npz"], metadata


def load_records_csv(csv_path):
    """Load a previously saved per-point CSV."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["point_index"] = int(row["point_index"])
            row["a_index"] = int(row["a_index"])
            row["h_index"] = int(row["h_index"])
            row["H_scaled"] = float(row["H_scaled"])
            row["A_scaled"] = float(row["A_scaled"])
            row["phase_id"] = int(row["phase_id"])
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            row["winner_topological_charge"] = float(row["winner_topological_charge"])
            row["ax"] = float(row["ax"])
            row["ay"] = float(row["ay"])
            for phase_name in ENERGY_PHASES:
                row[f"energy_{phase_name}"] = float(row[f"energy_{phase_name}"])
            records.append(row)
    return records


def merge_phase_diagram_chunks(run_name, output_root=None, save_plot=True):
    """Merge chunk outputs from a Slurm array sweep into one complete data bundle."""
    output_root = get_output_root(output_root)
    chunk_dir = output_root / "LLG" / "PhaseDiagramChunks"
    chunk_paths = sorted(chunk_dir.glob(f"{run_name}_chunk*.npz"))

    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found for run '{run_name}' in {chunk_dir}")

    first = np.load(chunk_paths[0], allow_pickle=False)
    H_vals = first["H_vals"]
    A_vals = first["A_vals"]
    L = int(first["lattice_size"])
    workers = 0

    bundle = empty_result_bundle(len(A_vals), len(H_vals))
    merged_records = []
    total_elapsed = 0.0
    chunk_count_declared = int(first["chunk_count"])

    for chunk_path in chunk_paths:
        chunk = np.load(chunk_path, allow_pickle=False)
        if not np.array_equal(chunk["H_vals"], H_vals) or not np.array_equal(chunk["A_vals"], A_vals):
            raise ValueError(f"Chunk axes do not match for '{chunk_path}'")
        if int(chunk["lattice_size"]) != L:
            raise ValueError(f"Lattice size mismatch in '{chunk_path}'")

        mask = chunk["mask"].astype(bool)
        bundle["grid"][mask] = chunk["grid"][mask]
        bundle["mask"][mask] = True
        bundle["topological_charge_grid"][mask] = chunk["topological_charge_grid"][mask]
        bundle["ax_grid"][mask] = chunk["ax_grid"][mask]
        bundle["ay_grid"][mask] = chunk["ay_grid"][mask]
        bundle["energy_skx_grid"][mask] = chunk["energy_skx_grid"][mask]
        bundle["energy_sc_grid"][mask] = chunk["energy_sc_grid"][mask]
        bundle["energy_sp_grid"][mask] = chunk["energy_sp_grid"][mask]
        bundle["energy_fm_grid"][mask] = chunk["energy_fm_grid"][mask]

        csv_path = chunk_path.with_suffix(".csv")
        merged_records.extend(load_records_csv(csv_path))
        total_elapsed += float(chunk["elapsed_seconds"])
        workers = max(workers, int(chunk["workers"]))
        chunk_count_declared = max(chunk_count_declared, int(chunk["chunk_count"]))

    bundle["records"] = sort_records(merged_records)
    completed_points = int(np.count_nonzero(bundle["mask"]))
    metadata = build_metadata(
        run_name=run_name,
        L=L,
        H_vals=H_vals,
        A_vals=A_vals,
        workers=workers,
        elapsed_seconds=total_elapsed,
        completed_points=completed_points,
        chunk_index=None,
        chunk_count=chunk_count_declared,
    )

    paths = get_output_paths(output_root, run_name, chunk_index=None, chunk_count=None)
    save_result_bundle(bundle, H_vals, A_vals, metadata, paths["npz"], paths["csv"], paths["json"])

    print(f"Merged {len(chunk_paths)} chunk files into '{paths['npz']}'")
    if save_plot and metadata["complete"]:
        plot_phase_diagram(bundle["grid"], H_vals, A_vals, paths["png"], show_plot=False)

    return paths["npz"], metadata


def plot_saved_phase_diagram(npz_path, out_path=None, show_plot=False):
    """Render a previously saved phase-diagram NPZ without recomputing anything."""
    data = np.load(npz_path, allow_pickle=False)
    run_name = str(data["run_name"]) if "run_name" in data else Path(npz_path).stem
    title = f"Topological Magnetic Phase Diagram ({run_name})"
    target = Path(out_path) if out_path else Path(npz_path).with_suffix(".png")
    plot_phase_diagram(data["grid"], data["H_vals"], data["A_vals"], target, title=title, show_plot=show_plot)


def parse_args():
    """CLI for fresh runs, chunk merges, and re-plotting saved bundles."""
    parser = argparse.ArgumentParser(description="Generate multicore, cluster-friendly skyrmion phase diagrams")
    parser.add_argument("--nH", type=int, default=26, help="Number of points along the H axis")
    parser.add_argument("--nA", type=int, default=33, help="Number of points along the A axis")
    parser.add_argument("--L", type=int, default=32, help="Lattice size for relaxation")
    parser.add_argument("--H-min", type=float, default=DEFAULT_H_RANGE[0], help="Minimum scaled magnetic field")
    parser.add_argument("--H-max", type=float, default=DEFAULT_H_RANGE[1], help="Maximum scaled magnetic field")
    parser.add_argument("--A-min", type=float, default=DEFAULT_A_RANGE[0], help="Minimum scaled anisotropy")
    parser.add_argument("--A-max", type=float, default=DEFAULT_A_RANGE[1], help="Maximum scaled anisotropy")
    parser.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))), help="Number of worker processes")
    parser.add_argument("--run-name", type=str, default=None, help="Custom label used in output filenames")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for outputs, e.g. $SCRATCH/skyrmion_runs")
    parser.add_argument("--chunk-index", type=int, default=None, help="Chunk index for Slurm array jobs (0-based)")
    parser.add_argument("--chunk-count", type=int, default=1, help="Total number of chunks for Slurm array jobs")
    parser.add_argument("--merge-run-name", type=str, default=None, help="Merge previously computed chunk files for this run name")
    parser.add_argument("--plot-file", type=str, default=None, help="Plot an existing NPZ bundle without recomputing")
    parser.add_argument("--plot-out", type=str, default=None, help="Override PNG output path when using --plot-file")
    parser.add_argument("--show-plot", action="store_true", help="Display the phase-diagram window after saving the PNG")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation after computing or merging")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Maximum LLG integration timestep")
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL stability factor for dynamic timestep")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.plot_file:
        plot_saved_phase_diagram(args.plot_file, out_path=args.plot_out, show_plot=args.show_plot)
    elif args.merge_run_name:
        merge_phase_diagram_chunks(args.merge_run_name, output_root=args.output_root, save_plot=not args.no_plot)
    else:
        generate_phase_diagram(
            n_H=args.nH,
            n_A=args.nA,
            L=args.L,
            H_min=args.H_min,
            H_max=args.H_max,
            A_min=args.A_min,
            A_max=args.A_max,
            workers=max(1, args.workers),
            output_root=args.output_root,
            run_name=args.run_name,
            chunk_index=args.chunk_index,
            chunk_count=args.chunk_count,
            max_dt=args.max_dt,
            cfl_factor=args.cfl,
            save_plot=not args.no_plot,
        )
