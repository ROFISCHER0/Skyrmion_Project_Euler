import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import io
import imageio.v3 as iio
import os

# Model parameters
L = 50     # Lattice size L x L
J = 1.0      # Ferromagnetic exchange
D = 0.5      # Interfacial Dzyaloshinskii-Moriya Interaction (DMI)
B = 0.2      # External magnetic field along z-axis
K = 0.05     # Uniaxial anisotropy constant (easy-axis along z)
T = 0.01      # Target Temperature (in units where kB=1)

@nb.njit
def get_energy_diff(spins, ix, iy, S_new, L, J, D, B, K):
    """
    Calculate the energy difference for a proposed spin change at (ix, iy).
    Uses Periodic Boundary Conditions.
    Hamiltonian H = -J * sum(S_i . S_j) - D * sum( (u_ij x z_hat) . (S_i x S_j) ) - B * sum(S_i_z) - K * sum((S_i_z)^2)
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
    
    # 4. Anisotropy energy difference: dE_K = -K * [(S_new_z)^2 - (S_old_z)^2]
    dE_K = -K * (S_new[2]**2 - spins[ix, iy][2]**2)
    
    return dE_ex + dE_dmi + dE_Z + dE_K

@nb.njit
def cone_step(S, max_angle=0.2):
    """Propose a new spin by slightly perturbing the current spin."""
    S_new = S + (np.random.rand(3) - 0.5) * max_angle
    norm = np.linalg.norm(S_new)
    return S_new / norm

@nb.njit
def mc_step(spins, L, J, D, B, K, T):
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
        dE = get_energy_diff(spins, ix, iy, S_new, L, J, D, B, K)
        
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

def plot_spins(spins, step, current_T):
    """Plot the spins as arrows in the xy plane, colored by Sz."""
    plt.clf()
    L = spins.shape[0]
    
    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(L), np.arange(L))
    
    # Extract components (transpose to match meshgrid)
    U = spins[:, :, 0].T
    V = spins[:, :, 1].T
    Sz = spins[:, :, 2].T
    
    # Plot quiver arrows colored by Sz
    q = plt.quiver(X, Y, U, V, Sz, cmap='bwr', pivot='mid', scale=L * 0.8)
    q.set_clim(-1, 1)  # Force constant colorbar scale
    
    plt.colorbar(q, label='Sz')
    plt.title(f'Lattice: {L}x{L}  |  T: {current_T:.3f}  |  Step: {step}')
    
    plt.xlim(-1, L)
    plt.ylim(-1, L)
    plt.gca().set_aspect('equal')
    plt.pause(0.01)

def run_simulation(save_mp4=True, video_filename="skyrmions.mp4"):
    # Determine the directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, video_filename)

    # Initialize
    spins = initialize_spins(L, "random") # Start random to anneal down
    
    print("Starting simulation... Preparing Numba JIT (may take a moment)")
    
    if not save_mp4:
        plt.ion()
    fig = plt.figure(figsize=(6, 5))
    
    steps = 10000
    plot_every = 100
    frames = []
    
    for step in range(steps):
        # Annealing phase: cool down from T=1.0 down to target T over the first 6000 steps
        current_T = T + (1.0 - T) * max(0.0, (6000 - step)/6000.0) 
        
        acceptance_rate = mc_step(spins, L, J, D, B, K, current_T)
        
        if step % plot_every == 0:
            print(f"Step {step:4d} | Acc Rate: {acceptance_rate:.3f} | T: {current_T:.3f}")
            plot_spins(spins, step, current_T)
            
            if save_mp4:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                frames.append(iio.imread(buf))
            
    if not save_mp4:
        plt.ioff()
        plt.show()
    else:
        print(f"Saving animation to {output_path}...")
        iio.imwrite(output_path, frames, fps=15, codec='libx264')
        print("Done!")

if __name__ == '__main__':
    run_simulation(save_mp4=True)
