# Skyrmion Monte Carlo Simulator

This project contains a highly optimized Python-based Monte Carlo simulator using the Metropolis algorithm to model the formation of magnetic skyrmion lattices.

## Physics Model
The classical spin Hamiltonian used to stabilize the skyrmions on a discrete 2D square lattice includes:
* **Heisenberg Exchange** ($J$): Ferromagnetic coupling between nearest neighbors.
* **Dzyaloshinskii-Moriya Interaction** ($D$): Interfacial antisymmetric exchange that favors chiral spin textures perpendicular to the neighbor direction.
* **Zeeman Energy** ($B$): Coupling to an external out-of-plane magnetic field.
* **Uniaxial Anisotropy** ($K$): Easy-axis out-of-plane anisotropy favoring $z$-oriented spins.

## Features
* **High Performance**: The energy calculations and Monte Carlo steps are fully compiled to machine code using **Numba** (`@nb.njit`), achieving C-like speeds directly from Python.
* **Simulated Annealing**: Gradually cools the system from a high-temperature random state down to a target temperature, smoothly nucleating a skyrmion lattice.
* **Live Visualization & MP4 Export**: Real-time visualization using matplotlib's quiver plots, and automatically captures the structural formation of the lattice to export it as an `mp4` video (via `imageio[ffmpeg]`).

## Dependencies
You can install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## Running the Simulation
Simply run the script:
```bash
python MC_metropolis.py
```
This will open a live matplotlib window showing the annealing process, and will save `skyrmions.mp4` in the project directory when finished.
