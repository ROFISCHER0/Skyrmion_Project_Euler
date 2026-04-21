# Skyrmion LLG Solver & Monte Carlo Simulator

This project contains a suite of high-performance Python tools for studying magnetic skyrmions. It includes an optimizer based on the deterministic Landau-Lifshitz-Gilbert (LLG) equation for exact phase stability analysis, and a Monte Carlo simulator using the Metropolis algorithm to model the formation dynamics of magnetic skyrmion lattices.

## Physics Model
The classical spin Hamiltonian used to stabilize the skyrmions on a discrete 2D square lattice includes:
* **Heisenberg Exchange** ($J$): Ferromagnetic coupling between nearest neighbors.
* **Dzyaloshinskii-Moriya Interaction** ($D$): Interfacial antisymmetric exchange that favors chiral spin textures perpendicular to the neighbor direction.
* **Zeeman Energy** ($B$): Coupling to an external out-of-plane magnetic field.
* **Uniaxial Anisotropy** ($K$): Easy-axis out-of-plane anisotropy favoring $z$-oriented spins.

## Features
* **High-Performance LLG Solver**: Numerically integrates the overdamped Landau-Lifshitz-Gilbert (LLG) equation to accurately find exact theoretical magnetic ground states (SkX, SC, SP, FM). Uses a robust Heun (RK2) integrator with fully dynamic spatial scaling, natively compiled with Numba for ~100x speedups.
* **Topological Phase Diagrams**: Systematically sweep across applied magnetic fields and anisotropy boundaries to construct precise quantitative stability phase diagrams of topological magnetic states.
* **Cluster-Friendly Batch Sweeps**: Phase diagrams now support non-interactive multicore execution, Slurm job-array chunking, merge/restart workflows, and structured `.npz` + `.csv` + `.json` outputs.
* **Monte Carlo Simulated Annealing**: Smoothly cools the system from a high-temperature random state to capture the dynamic thermal nucleation of a skyrmion lattice.
* **Live Visualization & Video Export**: Both numerical solvers feature real-time Matplotlib integrations utilizing multidimensional quiver plots, alongside automated MP4 video exports for monitoring structural formation.

## Dependencies
You can install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Running the Project

**1. Calculate Topological Phase Diagram**
Generates and plots a full numerical phase diagram comparing the energy densities of various theoretical skyrmion phase configurations (ansatzes). The sweep is headless by default and is designed to be batch-safe on Euler.
```bash
python phase_diagram.py --nH 50 --nA 50 --L 32 --workers 16 --run-name pd_L32_50x50
```

Outputs are written to:
* `output/LLG/PhaseDiagramData/<run_name>.npz`: dense grids for the winning phase IDs, energy surfaces, effective lattice constants, and approximate topological charge
* `output/LLG/PhaseDiagramData/<run_name>.csv`: one row per `(H, A)` point with phase label, energies, and topological-charge summary
* `output/LLG/PhaseDiagramData/<run_name>.json`: metadata such as resolution, ranges, lattice size, worker count, and ansatz multipliers
* `output/LLG/Graphs/<run_name>.png`: rendered phase-diagram plot

Useful options:
* `--H-min`, `--H-max`, `--A-min`, `--A-max`: control the numerical sweep window
* `--workers`: use multiple CPU cores on one node
* `--chunk-index`, `--chunk-count`: split a large sweep into Slurm-array chunks
* `--merge-run-name <name>`: merge all chunk files for a finished array sweep
* `--skx-multiplier`, `--sc-multiplier`, `--sp-multiplier`: increase the number of structured texture periods in the periodic unit cell

Example chunked workflow for a large Euler run:
```bash
python phase_diagram.py --nH 50 --nA 50 --L 32 --run-name pd_chunks --chunk-index 0 --chunk-count 8 --no-plot
python phase_diagram.py --merge-run-name pd_chunks
```

**2. Deterministic LLG Relaxation**
Test a specific Hamiltonian parameter set by relaxing analytical ansatz formulations directly. Saved ground states now include the approximate total topological charge.
```bash
python LLG_solver.py --H 1.0 --A 0.8 --L 64 --live-plot
```

To increase the number of periodic texture periods in the initial ansatz cell:
```bash
python LLG_solver.py --H 1.0 --A 0.8 --L 64 --skx-multiplier 2
```

**3. Monte Carlo Nucleation**
Simulate thermal melting and nucleation processes, plotting real-time visualizations.
```bash
python MC_metropolis.py
```

**4. Periodic Lattice Visualization**
Load `.npy` spin outputs from any of the numerical solvers to analyze multi-cell periodic states.
```bash
python periodic_plotting.py final_spins.npy --tiles 2 --mode quiver
```

## Euler / ETH HPC

The current ETH HPC documentation says Euler uses **Slurm** and provides Python modules such as:
```bash
module load stack/2024-06 python/3.12.8
```

For cluster runs, it is recommended to write results to scratch:
```bash
export SKYRMION_OUTPUT_ROOT="$SCRATCH/skyrmion_runs"
```

This repository includes ready-to-submit job scripts:
* `scripts/euler_phase_diagram.sbatch`: one multicore node, good for medium sweeps
* `scripts/euler_phase_diagram_array.sbatch`: Slurm array version for large sweeps or limited walltime
* `scripts/euler_merge_phase_diagram.sbatch`: merge chunk files after the array has finished

Examples:
```bash
sbatch scripts/euler_phase_diagram.sbatch
sbatch scripts/euler_phase_diagram_array.sbatch
sbatch scripts/euler_merge_phase_diagram.sbatch
```

If you want to override defaults without editing the scripts:
```bash
RUN_NAME=pd_L64_50x50 NH=50 NA=50 LATTICE_SIZE=64 sbatch scripts/euler_phase_diagram.sbatch
RUN_NAME=pd_L64_50x50_chunks NH=50 NA=50 LATTICE_SIZE=64 sbatch scripts/euler_phase_diagram_array.sbatch
RUN_NAME=pd_L64_50x50_chunks sbatch scripts/euler_merge_phase_diagram.sbatch
```
