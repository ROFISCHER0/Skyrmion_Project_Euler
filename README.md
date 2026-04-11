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
* **Monte Carlo Simulated Annealing**: Smoothly cools the system from a high-temperature random state to capture the dynamic thermal nucleation of a skyrmion lattice.
* **Live Visualization & Video Export**: Both numerical solvers feature real-time Matplotlib integrations utilizing multidimensional quiver plots, alongside automated MP4 video exports for monitoring structural formation.

## Dependencies
You can install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Running the Project

**1. Calculate Topological Phase Diagram**
Generates and plots a full numerical phase diagram comparing the energy densities of various theoretical skyrmion phase configurations (Ansatzes).
```bash
python phase_diagram.py --nH 26 --nA 33 --L 32
```

**2. Deterministic LLG Relaxation**
Test a specific Hamiltonian parameter set by relaxing analytical ansatz formulations directly.
```bash
python LLG_solver.py --H 1.0 --A 0.8 --L 64 --live-plot
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
