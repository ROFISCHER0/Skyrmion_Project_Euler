import numpy as np
import numba as nb
import io
import imageio.v3 as iio
import os
import time
import argparse
import tempfile
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt

# Constants and configuration parameters have been moved inside the run_simulation function.
# This prevents global state lock-in and allows execution in loops for phase diagrams.


def get_output_root(output_root=None):
    """Resolve output root for local runs or cluster runs."""
    if output_root:
        return Path(output_root).expanduser()
    env_output_root = os.environ.get("SKYRMION_OUTPUT_ROOT")
    if env_output_root:
        return Path(env_output_root).expanduser()
    return Path(__file__).resolve().parent / "output"


def resolve_output_path(path_value, output_root, default_relative_path):
    """Resolve relative output paths against the chosen output root."""
    if path_value:
        candidate = Path(path_value).expanduser()
        return candidate if candidate.is_absolute() else output_root / candidate
    return output_root / default_relative_path


def ensure_parent_dir(filepath):
    """Create parent directory for a file path if needed."""
    Path(filepath).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

@nb.njit(cache=True)
def get_energy_diff(spins, ix, iy, S_new, L, J, D, B, A):
    """
    Calculate the energy difference for a proposed spin change at (ix, iy).
    Uses Periodic Boundary Conditions.
    Hamiltonian H = -J * sum(S_i . S_j) - D * sum( (u_ij x z_hat) . (S_i x S_j) ) - B * sum(S_i_z) + A * sum((S_i_z)^2)
    """
    dS = S_new - spins[ix, iy]
    
    # Neighbors (Periodic boundary conditions)
    ix_right = (ix + 1) % L
    ix_left  = (ix - 1 + L) % L
    iy_up    = (iy + 1) % L
    iy_down  = (iy - 1 + L) % L
    
    # Neighbor spins
    S_right = spins[ix_right, iy]
    S_left  = spins[ix_left, iy]
    S_up    = spins[ix, iy_up]
    S_down  = spins[ix, iy_down]
    
    # 1. Exchange energy difference: dE_ex = -J * dS . sum(neighbor spins)
    dE_ex = -J * np.dot(dS, S_right + S_left + S_up + S_down)
    
    # 2. Interfacial DMI energy difference
    # D_ij = D * (z_hat x u_ij). This yields:
    v_right = np.array([-D * S_right[2], 0.0, D * S_right[0]])
    v_left  = np.array([D * S_left[2], 0.0, -D * S_left[0]])
    v_up    = np.array([0.0, -D * S_up[2], D * S_up[1]])
    v_down  = np.array([0.0, D * S_down[2], -D * S_down[1]])
    
    dE_dmi = -np.dot(dS, v_right + v_left + v_up + v_down)

    # 3. Zeeman energy difference: dE_Z = -B * dS_z
    dE_Z = -B * dS[2]
    
    # 4. Anisotropy energy difference: dE_A = A * [(S_new_z)^2 - (S_old_z)^2]
    dE_A = A * (S_new[2]**2 - spins[ix, iy][2]**2)
    
    return dE_ex + dE_dmi + dE_Z + dE_A

@nb.njit(cache=True)
def cone_step(S, max_angle=0.2):
    """Propose a new spin by slightly perturbing the current spin."""
    S_new = S + (np.random.rand(3) - 0.5) * max_angle
    norm = np.linalg.norm(S_new)
    return S_new / norm

@nb.njit(cache=True)
def mc_step(spins, L, J, D, B, A, T):
    """Perform one full Monte Carlo sweep linearly over the lattice."""
    accepted = 0
    # Loop over all sites roughly once per step (N random attempts)
    for _ in range(L * L):
        # Pick random site
        ix = np.random.randint(L)
        iy = np.random.randint(L)
        
        # Propose new spin
        S_new = cone_step(spins[ix, iy], 0.5) # 0.5 controls step size
        
        # Calculate energy difference
        dE = get_energy_diff(spins, ix, iy, S_new, L, J, D, B, A)
        
        # Metropolis criterion
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            spins[ix, iy] = S_new
            accepted += 1
            
    return accepted / (L * L)

def initialize_spins(L, state="ferro"):
    """Initialize the spin lattice."""
    spins = np.zeros((L, L, 3))
    if state == "ferro":
        spins[:, :, 2] = 1.0  # All point in +z
    elif state == "random":
        for i in range(L):
            for j in range(L):
                z = np.random.uniform(-1, 1)
                r = np.sqrt(1 - z**2)
                phi = np.random.uniform(0, 2*np.pi)
                spins[i, j] = [r*np.cos(phi), r*np.sin(phi), z]
    return spins

def plot_spins(spins, step, current_T, display_mode="quiver", cmap_name="bwr"):
    """Plot the spins as arrows in the xy plane."""
    plt.clf()
    L = spins.shape[0]
    
    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(L), np.arange(L))
    
    # Extract components (transpose to match meshgrid)
    U = spins[:, :, 0].T
    V = spins[:, :, 1].T
    Sz = spins[:, :, 2].T
    
    if display_mode == "heatmap":
        # Plot background heatmap of Sz to ensure spins with U=0, V=0 are visible
        im = plt.imshow(Sz, cmap=cmap_name, vmin=-1, vmax=1, origin='lower', extent=[-0.5, L-0.5, -0.5, L-0.5])
        
        # Plot quiver arrows for in-plane components
        q = plt.quiver(X, Y, U, V, pivot='mid', scale=L, width=0.008)
        
        plt.colorbar(im, label='Sz')
    elif display_mode == "quiver":
        # Plot quiver arrows colored by Sz
        q = plt.quiver(X, Y, U, V, Sz, cmap=cmap_name, pivot='mid', scale=L, width=0.008)
        q.set_clim(-1, 1)  # Force constant colorbar scale
        
        plt.colorbar(q, label='Sz')
    plt.title(f'Lattice: {L}x{L}  |  T: {current_T:.3f}  |  Step: {step}')
    
    plt.xlim(-1, L)
    plt.ylim(-1, L)
    plt.gca().set_aspect('equal')
    plt.pause(0.01)

def run_simulation(
    L=15, J=1.0, D=0.5, h_scaled=1.5, a_scaled=-0.5,
    T_start=1.0, T_target=0.01, steps=10000,
    cooling_protocol="continuous", initial_spins=None,
    enable_plotting=False, save_mp4=False,
    video_filename=None, output_filename=None,
    dpi=300, display_mode="quiver", cmap_name="bwr",
    output_root=None
):
    """
    Run the Monte Carlo Metropolis simulation for magnetic skyrmions.
    
    Parameters:
    - cooling_protocol: "continuous", "stepwise", or "constant"
    - initial_spins: Provide a 3D numpy array to start from a specific configuration
    """
    # Calculate physical parameters B and A from the scaled parameters
    B = h_scaled * (D**2 / J)
    A = a_scaled * (D**2 / J)

    # Determine paths
    output_root = get_output_root(output_root)
    output_path = resolve_output_path(
        output_filename,
        output_root,
        Path("MC") / "npy" / f"final_spins_L{L}_A{a_scaled:.2f}_H{h_scaled:.2f}.npy"
    )
    video_path = resolve_output_path(
        video_filename,
        output_root,
        Path("MC") / "videos" / f"skyrmions_L{L}_A{a_scaled:.2f}_H{h_scaled:.2f}.mp4"
    )

    # Initialize spin lattice
    if initial_spins is not None:
        spins = np.copy(initial_spins)
        if spins.shape != (L, L, 3):
            raise ValueError(f"initial_spins shape {spins.shape} does not match L={L}")
    else:
        # Start random to anneal down
        spins = initialize_spins(L, "random")
    
    print("Starting simulation... (Numba JIT compilation may run on first execution)")
    
    # Setup visualization elements
    if enable_plotting:
        if not save_mp4:
            plt.ion()
        fig = plt.figure(figsize=(6, 5))
        plot_every = max(1, steps // 100) # Save ~100 frames total
        frames = []
    else:
        plot_every = max(1, steps // 10)  # Print to console 10 times during sim
    
    # Setup Stepwise Cooling Schedule
    if cooling_protocol == "stepwise":
        # Make the number of sweeps per temperature dependent on the number of spins
        sweeps_per_T = L * L
        total_t_steps = max(1, steps // sweeps_per_T)
        t_schedule = np.linspace(T_start, T_target, total_t_steps)
    else:
        sweeps_per_T = 1
    
    for step in range(steps):
        # Determine current temperature based on the selected protocol
        if cooling_protocol == "continuous":
            # Cool down from T_start down to T_target over the first 60% of steps
            annealing_steps = int(steps * 0.6)
            if step < annealing_steps:
                current_T = T_target + (T_start - T_target) * ((annealing_steps - step) / annealing_steps)
            else:
                current_T = T_target
                
        elif cooling_protocol == "stepwise":
            t_index = min(step // sweeps_per_T, len(t_schedule) - 1)
            current_T = t_schedule[t_index]
            
        elif cooling_protocol == "constant":
            current_T = T_target
            
        else:
            current_T = T_target # Fallback
            
        # Perform 1 Monte Carlo Sweep (L*L random attempts)
        acceptance_rate = mc_step(spins, L, J, D, B, A, current_T)
        
        # Logging and Visualization
        if step % plot_every == 0:
            print(f"Step {step:5d}/{steps} | Acc Rate: {acceptance_rate:.3f} | T: {current_T:.3f}")
            
            if enable_plotting:
                plot_spins(spins, step, current_T, display_mode=display_mode, cmap_name=cmap_name)
                
                if save_mp4:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
                    buf.seek(0)
                    frames.append(iio.imread(buf))
            
    # Teardown & Outputs
    if enable_plotting:
        if not save_mp4:
            plt.ioff()
            plt.show()
        else:
            ensure_parent_dir(video_path)
            print(f"Saving animation to {video_path}...")
            iio.imwrite(video_path, frames, fps=15, codec='libx264')
    else:
        # Avoid popping up un-rendered figures if plotting wasn't enabled
        pass 
    if output_path:
        ensure_parent_dir(output_path)
        print(f"Saving final spin configuration to {output_path}...")
        np.save(output_path, spins)
    print("Run Complete!")
    
    return spins

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monte Carlo Metropolis Skyrmion Simulation")
    parser.add_argument("--L", type=int, default=15, help="Lattice size (L x L)")
    parser.add_argument("--J", type=float, default=1.0, help="Exchange interaction")
    parser.add_argument("--D", type=float, default=0.5, help="DMI strength")
    parser.add_argument("--H", type=float, default=1.5, help="Scaled magnetic field")
    parser.add_argument("--A", type=float, default=-0.5, help="Scaled Anisotropy")
    parser.add_argument("--T-start", type=float, default=1.0, help="Initial temperature")
    parser.add_argument("--T-target", type=float, default=0.01, help="Final/target temperature")
    parser.add_argument("--steps", type=int, default=10000, help="Number of Monte Carlo sweeps")
    parser.add_argument("--protocol", type=str, choices=["continuous", "stepwise", "constant"], default="continuous", help="Cooling protocol")
    parser.add_argument("--plot", action="store_true", help="Enable live plotting (or saving frames if --save-mp4 is used)")
    parser.add_argument("--save-mp4", action="store_true", help="Save the simulation as an MP4 video")
    parser.add_argument("--video-file", type=str, default=None, help="Output MP4 filename (if save_mp4 is enabled)")
    parser.add_argument("--out-npy", type=str, default=None, help="Output .npy spin configuration filename")
    parser.add_argument("--mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for plotting")
    parser.add_argument("--cmap", type=str, default="bwr", help="Colormap for plotting")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for outputs, e.g. $SCRATCH/skyrmion_runs")
    
    args = parser.parse_args()

    time1 = time.time()
    print("--- Starting MC Simulation ---")
    
    final_spins = run_simulation(
        L=args.L, J=args.J, D=args.D, h_scaled=args.H, a_scaled=args.A,
        T_start=args.T_start, T_target=args.T_target, steps=args.steps,
        cooling_protocol=args.protocol,
        enable_plotting=args.plot, save_mp4=args.save_mp4,
        video_filename=args.video_file, output_filename=args.out_npy,
        display_mode=args.mode, cmap_name=args.cmap,
        output_root=args.output_root
    )
    
    time2 = time.time()
    print(f"Total time taken: {time2 - time1:.2f}s")
