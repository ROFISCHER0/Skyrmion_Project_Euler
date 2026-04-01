import numpy as np
import numba as nb
import argparse
import os

# ---------------------------------------------------------
# Part A: Ansatz Generators
# ---------------------------------------------------------

def init_SkX(L, Q=1, gamma=np.pi):
    """
    Skyrmion Lattice (SkX) Ansatz
    A singular skyrmion profile mapped onto the periodic L x L grid.
    n_theta = -pi * r / R
    n_phi = Q * phi + gamma
    """
    spins = np.zeros((L, L, 3))
    R = L / 2.0  # Skyrmion radius (fits within the cell)
    
    for i in range(L):
        for j in range(L):
            # Coordinates relative to center
            dx = i - L/2 + 0.5
            dy = j - L/2 + 0.5
            
            r = np.sqrt(dx**2 + dy**2)
            phi = np.arctan2(dy, dx)
            
            # If outside the skyrmion core radius, it points fully down (FM background is -z)
            # Wait, the paper formula n_theta = -pi*r/R. At r=0, theta=0 (+z). At r=R, theta=-pi (-z)
            if r <= R:
                theta = -np.pi * r / R
            else:
                theta = -np.pi # Fully down (or up depending on convention)
                
            phi_spin = Q * phi + gamma
            
            spins[i, j, 0] = np.sin(theta) * np.cos(phi_spin)
            spins[i, j, 1] = np.sin(theta) * np.sin(phi_spin)
            spins[i, j, 2] = np.cos(theta)
            
    return spins

def init_SP(L):
    """
    Spiral Phase (SP) Ansatz
    Rashba DMI stabilizes a cycloid. Spins rotate in the x-z plane along the x-direction.
    """
    spins = np.zeros((L, L, 3))
    q = 2 * np.pi / L # Positive q to match the Rashba DMI chirality
    
    for i in range(L):
        for j in range(L):
            x = i
            # e_u = z_hat, e_v = x_hat for cycloid
            # n = e_u * cos(q*x) + e_v * sin(q*x)
            spins[i, j, 0] = np.sin(q * x)
            spins[i, j, 1] = 0.0
            spins[i, j, 2] = np.cos(q * x)
            
    return spins

def init_SC(L):
    """
    Square Cell (SC) Vortex-Antivortex Phase Ansatz
    A superposition of two orthogonal cycloids (one along x, one along y).
    """
    spins = np.zeros((L, L, 3))
    q = 2 * np.pi / L
    
    for i in range(L):
        for j in range(L):
            x = i
            y = j
            
            # Spiral 1 (along x): cycloid in xz plane
            n1_x = np.sin(q * x)
            n1_y = 0.0
            n1_z = np.cos(q * x)
            
            # Spiral 2 (along y): cycloid in yz plane
            n2_x = 0.0
            n2_y = np.sin(q * y)
            n2_z = np.cos(q * y)
            
            # Superposition
            n_sum_x = n1_x + n2_x
            n_sum_y = n1_y + n2_y
            n_sum_z = n1_z + n2_z
            
            # Position-dependent normalization constraint |n| = 1
            norm = np.sqrt(n_sum_x**2 + n_sum_y**2 + n_sum_z**2)
            if norm == 0:
                spins[i, j, 2] = 1.0 # Fallback
            else:
                spins[i, j, 0] = n_sum_x / norm
                spins[i, j, 1] = n_sum_y / norm
                spins[i, j, 2] = n_sum_z / norm
                
    return spins

def load_npy_ansatz(filepath, L):
    """Fallback to load MC output if requested."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}")
    spins = np.load(filepath)
    if spins.shape != (L, L, 3):
        raise ValueError(f"Shape mismatch: {spins.shape} vs {(L, L, 3)}")
    return spins

# ---------------------------------------------------------
# Part B & C: Energy, Effective Field, and LLG step
# ---------------------------------------------------------

@nb.njit
def relax_phase_numba(spins, L, H_scaled, A_scaled, max_steps=50000, tol=1e-7, ax_in=1.0, ay_in=1.0, prev_f_in=0.0):
    """
    Perform the full overdamped LLG integration natively in Numba.
    This avoids Python overhead and memory allocations at every step, making it ~100x faster.
    """
    ax = ax_in
    ay = ay_in
    prev_f = prev_f_in
    dt = 0.05 
    
    # Pre-allocate buffers for ping-pong swapping
    spins_current = spins.copy()
    spins_next = np.empty_like(spins)
    
    f_tot = 0.0
    
    for step in range(max_steps):
        # Dynamic timestep bounded by Von Neumann stability analysis (CFL condition)
        dt = min(0.05, 0.25 * min(ax, ay)**2)
        
        # Energy accumulators (unscaled)
        E_ex_x = 0.0
        E_ex_y = 0.0
        E_dmi_x = 0.0
        E_dmi_y = 0.0
        E_z = 0.0
        E_a = 0.0
        
        for i in range(L):
            for j in range(L):
                # Periodic neighbors
                i_next = (i + 1) % L
                i_prev = (i - 1 + L) % L
                j_next = (j + 1) % L
                j_prev = (j - 1 + L) % L
                
                # Unroll vectors heavily to optimize C-compilation
                n0 = spins_current[i, j, 0]
                n1 = spins_current[i, j, 1]
                n2 = spins_current[i, j, 2]
                
                nx_next_0, nx_next_1, nx_next_2 = spins_current[i_next, j, 0], spins_current[i_next, j, 1], spins_current[i_next, j, 2]
                nx_prev_0, nx_prev_1, nx_prev_2 = spins_current[i_prev, j, 0], spins_current[i_prev, j, 1], spins_current[i_prev, j, 2]
                
                ny_next_0, ny_next_1, ny_next_2 = spins_current[i, j_next, 0], spins_current[i, j_next, 1], spins_current[i, j_next, 2]
                ny_prev_0, ny_prev_1, ny_prev_2 = spins_current[i, j_prev, 0], spins_current[i, j_prev, 1], spins_current[i, j_prev, 2]
                
                # --- Energy Accumulation ---
                E_ex_x += 0.5 * ((nx_next_0 - n0)**2 + (nx_next_1 - n1)**2 + (nx_next_2 - n2)**2)
                E_ex_y += 0.5 * ((ny_next_0 - n0)**2 + (ny_next_1 - n1)**2 + (ny_next_2 - n2)**2)
                
                E_dmi_x += (n2 * (nx_next_0 - n0) - n0 * (nx_next_2 - n2))
                E_dmi_y += (n2 * (ny_next_1 - n1) - n1 * (ny_next_2 - n2))
                
                E_z -= H_scaled * n2
                E_a += A_scaled * n2**2
                
                # --- Effective Field Calculation ---
                H0 = (nx_next_0 + nx_prev_0 - 2*n0)/(ax**2) + (ny_next_0 + ny_prev_0 - 2*n0)/(ay**2)
                H1 = (nx_next_1 + nx_prev_1 - 2*n1)/(ax**2) + (ny_next_1 + ny_prev_1 - 2*n1)/(ay**2)
                H2 = (nx_next_2 + nx_prev_2 - 2*n2)/(ax**2) + (ny_next_2 + ny_prev_2 - 2*n2)/(ay**2)
                
                H0 += - (nx_next_2 - nx_prev_2) / (2 * ax)
                H1 += - (ny_next_2 - ny_prev_2) / (2 * ay)
                H2 += (nx_next_0 - nx_prev_0) / (2 * ax) + (ny_next_1 - ny_prev_1) / (2 * ay)
                
                H2 += H_scaled
                H2 -= 2 * A_scaled * n2
                
                # --- LLG Update ---
                dot_val = n0*H0 + n1*H1 + n2*H2
                ndot0 = H0 - dot_val * n0
                ndot1 = H1 - dot_val * n1
                ndot2 = H2 - dot_val * n2
                
                n_new0 = n0 + dt * ndot0
                n_new1 = n1 + dt * ndot1
                n_new2 = n2 + dt * ndot2
                
                # Strict Re-normalization
                norm = np.sqrt(n_new0**2 + n_new1**2 + n_new2**2)
                spins_next[i, j, 0] = n_new0 / norm
                spins_next[i, j, 1] = n_new1 / norm
                spins_next[i, j, 2] = n_new2 / norm

        # Average energies per spin
        L2 = L*L
        E_ex_x /= L2
        E_ex_y /= L2
        E_dmi_x /= L2
        E_dmi_y /= L2
        E_z /= L2
        E_a /= L2
        
        # Total Scaled Energy Density (DMI is lowered, hence minus sign)
        f_tot = (E_ex_x / ax**2) + (E_ex_y / ay**2) - (E_dmi_x / ax) - (E_dmi_y / ay) + E_z + E_a
        
        # Dynamic Scaling Strategy: (ax minimizes E_ex/a^2 - E_dmi/a)
        if abs(E_dmi_x) > 1e-12 and abs(E_ex_x) > 1e-12:
            ax = 2.0 * E_ex_x / E_dmi_x
        if abs(E_dmi_y) > 1e-12 and abs(E_ex_y) > 1e-12:
            ay = 2.0 * E_ex_y / E_dmi_y
            
        # Optional clamping
        if ax <= 0: ax = 1.0
        if ay <= 0: ay = 1.0

        # Check convergence
        if step > 1000 and abs(f_tot - prev_f) < tol:
            # We must break here and transfer the output
            spins_current[:] = spins_next[:]
            break
            
        prev_f = f_tot
        
        # Swap buffers for the next timestep (avoids np.zeros_like allocation)
        temp = spins_current
        spins_current = spins_next
        spins_next = temp
        
    return spins_current, f_tot, ax, ay, step

# ---------------------------------------------------------
# Part D & E: Execution Wrapper
# ---------------------------------------------------------

def relax_phase(spins, L, H_scaled, A_scaled, phase_name, max_steps=50000, tol=1e-7, live_plot=False):
    """
    Relax the given spin configuration using LLG and dynamic scaling.
    This is now a wrapper around the ultra-fast Numba integrator.
    """
    ax, ay = 1.0, 1.0
    prev_f = 0.0
    steps_done = 0
    
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax_plot = plt.subplots(figsize=(5,5))
        im = ax_plot.imshow(spins[:, :, 2], vmin=-1, vmax=1, cmap='bwr', origin='lower')
        ax_plot.set_title(f"Relaxing {phase_name}...")
        plt.show()
    
    chunk = 250 if live_plot else max_steps
    
    while steps_done < max_steps:
        # Numba executes cleanly in small chunks so we can visualize intermediate states
        spins_final, f_tot, ax, ay, steps_taken = relax_phase_numba(
            spins, L, H_scaled, A_scaled, chunk, tol, ax, ay, prev_f
        )
        
        # The true number of steps executed is steps_taken + 1 because the loop is 0-indexed and returns `step`
        steps_done += (steps_taken + 1)
        prev_f = f_tot
        spins = spins_final
        
        if live_plot:
            im.set_data(spins[:, :, 2])
            ax_plot.set_title(f"[{phase_name}] Step {steps_done} | f = {f_tot:.4f}")
            plt.pause(0.01)
            
        if steps_taken + 1 < chunk:
            break
            
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.close(fig)
    
    print(f"[{phase_name}] Relaxed in {steps_done} steps. Energy Density: {f_tot:.5f} (ax={ax:.3f}, ay={ay:.3f})")
    return spins, f_tot

def get_FM_energy(H_scaled, A_scaled):
    """Calculate the exact analytical energy of the FM state."""
    # Aligned FM (nz = 1 or -1)
    # The energy is E = A_s n_z^2 - H n_z
    e_aligned = A_scaled - H_scaled # For nz = +1
    e_anti_aligned = A_scaled + H_scaled # For nz = -1
    
    # Tilted FM (occurs when |H| < 2|A_s| and A_s > 0 usually)
    # nz = H / (2 A_s)
    e_tilted = float('inf')
    if A_scaled != 0:
        nz_tilted = H_scaled / (2 * A_scaled)
        if abs(nz_tilted) <= 1.0:
            e_tilted = - (H_scaled**2) / (4 * A_scaled)
            
    return min(e_aligned, e_anti_aligned, e_tilted)

def compare_phases(H_scaled=0.08, A_scaled=0.5, L=64, npy_file=None, plot_ansatz=False, live_plot=False):
    """
    Main Execution: Tests SkX, SP, and FM to find the true numerical ground state.
    """
    print(f"--- Phase Stability Analysis H={H_scaled}, As={A_scaled} ---")
    results = {}
    
    # 1. Skyrmion Lattice
    print("Initializing SkX...")
    spins_skx = init_SkX(L)
    np.save("ansatz_SkX.npy", spins_skx)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SkX Ansatz...")
            plot_periodic_structure("ansatz_SkX.npy", tiles_x=2, tiles_y=2, display_mode="quiver")
        except: pass
    spins_skx, f_skx = relax_phase(spins_skx, L, H_scaled, A_scaled, "SkX", live_plot=live_plot)
    results["SkX"] = f_skx
    
    # 2. Square Cell 
    print("Initializing SC...")
    spins_sc = init_SC(L)
    np.save("ansatz_SC.npy", spins_sc)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SC Ansatz...")
            plot_periodic_structure("ansatz_SC.npy", tiles_x=2, tiles_y=2, display_mode="quiver")
        except: pass
    spins_sc, f_sc = relax_phase(spins_sc, L, H_scaled, A_scaled, "SC", live_plot=live_plot)
    results["SC"] = f_sc
    
    # 3. Spiral Phase
    print("Initializing SP...")
    spins_sp = init_SP(L)
    np.save("ansatz_SP.npy", spins_sp)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SP Ansatz...")
            plot_periodic_structure("ansatz_SP.npy", tiles_x=2, tiles_y=2, display_mode="quiver")
        except: pass
    spins_sp, f_sp = relax_phase(spins_sp, L, H_scaled, A_scaled, "SP", live_plot=live_plot)
    results["SP"] = f_sp
    
    # 4. Ferromagnetic
    f_fm = get_FM_energy(H_scaled, A_scaled)
    print(f"[FM] Analytical Energy Density: {f_fm:.5f}")
    results["FM"] = f_fm
    
    # 5. Custom NPY (Optional)
    if npy_file and os.path.exists(npy_file):
        print(f"Initializing from custom NPY {npy_file}...")
        spins_cust = load_npy_ansatz(npy_file, L)
        spins_cust, f_cust = relax_phase(spins_cust, L, H_scaled, A_scaled, "Custom (NPY)")
        results["Custom"] = f_cust
    # Determine Winner
    f_fm_val = results["FM"]
    # If a structured phase collapsed into the FM state, it has unraveled. Discard its label.
    for phase_key in ["SkX", "SC", "SP", "Custom"]:
        if phase_key in results and abs(results[phase_key] - f_fm_val) < 1e-5:
            print(f"[{phase_key}] completely unraveled into FM state during relaxation. Discarding its phase label.")
            del results[phase_key]
            
    winner = min(results, key=results.get)
    print(f"\n=> The Ground State Phase is: {winner} (Energy: {results[winner]:.5f})")
    
    # Extract the winning spins array and save it
    best_spins = None
    if winner == "SkX": best_spins = spins_skx
    elif winner == "SC": best_spins = spins_sc
    elif winner == "SP": best_spins = spins_sp
    elif winner == "FM":
        # Synthesize a pure mathematical FM uniform state
        best_spins = np.zeros((L, L, 3))
        if f_fm == A_scaled - H_scaled:
            best_spins[:, :, 2] = 1.0 # Aligned UP
        elif f_fm == A_scaled + H_scaled:
            best_spins[:, :, 2] = -1.0 # Aligned DOWN
        else:
            nz = H_scaled / (2 * A_scaled)
            nx = np.sqrt(abs(1.0 - nz**2))
            best_spins[:, :, 0] = nx # Tilted
            best_spins[:, :, 2] = nz
            
    if best_spins is not None:
        np.save("llg_groundstate.npy", best_spins)
        print("Saved analytical ground state to 'llg_groundstate.npy'")
        # Auto-plot
        try:
            from periodic_plotting import plot_periodic_structure
            print("Launching periodic plot...")
            plot_periodic_structure("llg_groundstate.npy", tiles_x=2, tiles_y=2, display_mode="quiver")
        except Exception as e:
            print(f"Could not load periodic_plotting to display: {e}")
    
    return winner, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic LLG Phase Analyzer")
    parser.add_argument("--H", type=float, default=1.0, help="Scaled magnetic field")
    parser.add_argument("--A", type=float, default=0.8, help="Scaled Anisotropy")
    parser.add_argument("--L", type=int, default=64, help="Grid size L")
    parser.add_argument("--npy", type=str, default=None, help="Optional MC .npy file to use as an ansatz")
    parser.add_argument("--plot-ansatz", action="store_true", help="Plot each ansatz configuration before relaxing")
    parser.add_argument("--live-plot", action="store_true", help="Plot the real-time evolution of the solver")
    
    args = parser.parse_args()
    
    compare_phases(H_scaled=args.H, A_scaled=args.A, L=args.L, npy_file=args.npy, plot_ansatz=args.plot_ansatz, live_plot=args.live_plot)
