import numpy as np
import numba as nb
import argparse
import os
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------------------------------------
# Part A: Ansatz Generators
# ---------------------------------------------------------

def _validate_period_multiplier(name, value):
    """Validate integer texture-density multipliers used in periodic ansatze."""
    if int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def init_SkX(L, Q=1, gamma=0.0, period_multiplier=1):
    """
    Hexagonal 3Q Skyrmion Lattice (SkX) Ansatz
    A proper vector superposition of three cycloids to form an exact hexagonal lattice.
    Mapped onto physical coordinates [-L_x/2, L_x/2] and [-L_y/2, L_y/2].
    Returns spins, ax, ay
    """
    period_multiplier = _validate_period_multiplier("period_multiplier", period_multiplier)
    spins = np.zeros((L, L, 3))
    
    ax = np.pi / L
    ay = (np.pi / np.sqrt(3)) / L
    
    # We use q=4.0 so that exactly two complementary skyrmions precisely fit 
    # the periodic rectangular unit cell (L_x = pi, L_y = pi/sqrt(3)).
    # Dynamic scaling (ax, ay) during LLG will natively expand this to the optimal physical period.
    q_ansatz = 4.0 * period_multiplier
    
    # Base offset to ensure SkX background matches the FM -z state for stability
    # Superposition varies from -1.5 to +3.0. A -1.5 offset makes it vary from -3.0 to +1.5.
    M0 = -1.5 
    
    for i in range(L):
        for j in range(L):
            x = (i - L/2 + 0.5) * ax
            y = (j - L/2 + 0.5) * ay
            
            # The three k-vectors for the hexagonal Brillouin zone
            W1 = q_ansatz * x
            W2 = q_ansatz * (-0.5 * x + (np.sqrt(3)/2) * y)
            W3 = q_ansatz * (-0.5 * x - (np.sqrt(3)/2) * y)
            
            nz = M0 + np.cos(W1) + np.cos(W2) + np.cos(W3)
            
            # Néel skyrmion: in-plane components are parallel to the q-vectors. 
            # Subtracted mathematically so DMI chirality stabilizes negatively (matches our q=-2 fix for SP).
            sign = -1.0 
            nx = sign * (np.sin(W1)*1.0 + np.sin(W2)*(-0.5) + np.sin(W3)*(-0.5))
            ny = sign * (np.sin(W1)*0.0 + np.sin(W2)*(np.sqrt(3)/2) + np.sin(W3)*(-np.sqrt(3)/2))
            
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            if norm == 0:
                spins[i, j, 2] = -1.0
            else:
                spins[i, j, 0] = nx / norm
                spins[i, j, 1] = ny / norm
                spins[i, j, 2] = nz / norm
                
    return spins, ax, ay

def init_SP(L, period_multiplier=1):
    """
    Spiral Phase (SP) Ansatz
    Mapped onto physical coordinates [0, L_x] and [0, L_y].
    Returns spins, ax, ay
    """
    period_multiplier = _validate_period_multiplier("period_multiplier", period_multiplier)
    spins = np.zeros((L, L, 3))
    
    ax = np.pi / L
    ay = np.pi / L
    
    # We use q=2.0 since the cos/sin phase arrangement naturally yields negative DMI
    q = 2.0 * period_multiplier
    
    for i in range(L):
        for j in range(L):
            x = i * ax
            spins[i, j, 0] = np.cos(q * x)
            spins[i, j, 1] = 0.0
            spins[i, j, 2] = np.sin(q * x)
            
    return spins, ax, ay

def init_SC(L, period_multiplier=1):
    """
    Square Cell (SC) Vortex-Antivortex Phase Ansatz
    Mapped onto physical coordinates [0, L_x] and [0, L_y].
    Returns spins, ax, ay
    """
    period_multiplier = _validate_period_multiplier("period_multiplier", period_multiplier)
    spins = np.zeros((L, L, 3))
    
    ax = (np.pi / 2.0) / L
    ay = (np.pi / 2.0) / L
    
    # We use q=4.0 (+4.0) because the user's specific sin/cos permutation natively yields negative DMI
    q = 4.0 * period_multiplier
    
    for i in range(L):
        for j in range(L):
            x = i * ax
            y = j * ay
            
            # Wave 1: Propagates along x
            n1_x = np.cos(q * x)
            n1_y = 0.0
            n1_z = np.sin(q * x)
            
            # Wave 2: Propagates along y
            n2_x = 0.0
            n2_y = np.cos(q * y)
            n2_z = np.sin(q * y)
            
            n_sum_x = n1_x + n2_x
            n_sum_y = n1_y + n2_y
            n_sum_z = n1_z + n2_z
            
            norm = np.sqrt(n_sum_x**2 + n_sum_y**2 + n_sum_z**2)
            
            # Normalization (Crucial Step)
            if norm == 0:
                spins[i, j, 2] = -1.0 # Fallback aligned to negative H field preference
            else:
                spins[i, j, 0] = n_sum_x / norm
                spins[i, j, 1] = n_sum_y / norm
                spins[i, j, 2] = n_sum_z / norm
                
    return spins, ax, ay

def load_ansatz(filepath, L):
    """Load MC or LLG output (npy or npz) as an ansatz."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}")
        
    ax, ay = 1.0, 1.0
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        spins = data['spins']
        if 'ax' in data: ax = float(data['ax'])
        if 'ay' in data: ay = float(data['ay'])
    else:
        spins = np.load(filepath)
        
    if spins.shape != (L, L, 3):
        raise ValueError(f"Shape mismatch: {spins.shape} vs {(L, L, 3)}")
    return spins, ax, ay


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

# ---------------------------------------------------------
# Part B & C: Energy, Effective Field, and LLG step
# ---------------------------------------------------------

@nb.njit(cache=True)
def relax_phase_numba(spins, L, H_scaled, A_scaled, max_steps=50000, tol=1e-7, ax_in=1.0, ay_in=1.0, prev_f_in=0.0, max_dt=0.05, cfl_factor=0.25, global_step_start=0):
    """
    Perform the full overdamped LLG integration natively in Numba using Heun (RK2) method.
    This avoids Python overhead and memory allocations at every step, making it ~100x faster.
    """
    ax = ax_in
    ay = ay_in
    prev_f = prev_f_in
    dt = max_dt 
    
    # Pre-allocate buffers for ping-pong swapping and predictor state
    spins_current = spins.copy()
    spins_next = np.empty_like(spins)
    spins_pred = np.empty_like(spins)
    ndot_current = np.empty_like(spins)
    
    f_tot = 0.0
    
    for step in range(max_steps):
        # Dynamic timestep bounded by Von Neumann stability analysis (CFL condition)
        dt = min(max_dt, cfl_factor * min(ax, ay)**2)
        
        # Energy accumulators (unscaled)
        E_ex_x = 0.0
        E_ex_y = 0.0
        E_dmi_x = 0.0
        E_dmi_y = 0.0
        E_z = 0.0
        E_a = 0.0
        
        # --- PASS 1: Predictor Step ---
        for i in range(L):
            for j in range(L):
                # Periodic neighbors
                i_right = (i + 1) % L
                i_left = (i - 1 + L) % L
                j_up = (j + 1) % L
                j_down = (j - 1 + L) % L
                
                # Unroll vectors heavily to optimize C-compilation
                n_x = spins_current[i, j, 0]
                n_y = spins_current[i, j, 1]
                n_z = spins_current[i, j, 2]
                
                n_right_x, n_right_y, n_right_z = spins_current[i_right, j, 0], spins_current[i_right, j, 1], spins_current[i_right, j, 2]
                n_left_x, n_left_y, n_left_z = spins_current[i_left, j, 0], spins_current[i_left, j, 1], spins_current[i_left, j, 2]
                
                n_up_x, n_up_y, n_up_z = spins_current[i, j_up, 0], spins_current[i, j_up, 1], spins_current[i, j_up, 2]
                n_down_x, n_down_y, n_down_z = spins_current[i, j_down, 0], spins_current[i, j_down, 1], spins_current[i, j_down, 2]
                
                # --- Energy Accumulation ---
                E_ex_x += 0.5 * ((n_right_x - n_x)**2 + (n_right_y - n_y)**2 + (n_right_z - n_z)**2)
                E_ex_y += 0.5 * ((n_up_x - n_x)**2 + (n_up_y - n_y)**2 + (n_up_z - n_z)**2)
                
                E_dmi_x += (n_z * (n_right_x - n_x) - n_x * (n_right_z - n_z))
                E_dmi_y += (n_z * (n_up_y - n_y) - n_y * (n_up_z - n_z))
                
                E_z += H_scaled * n_z
                E_a += A_scaled * n_z**2
                
                # --- Effective Field Calculation ---
                H_x = (n_right_x + n_left_x - 2*n_x)/(ax**2) + (n_up_x + n_down_x - 2*n_x)/(ay**2)
                H_y = (n_right_y + n_left_y - 2*n_y)/(ax**2) + (n_up_y + n_down_y - 2*n_y)/(ay**2)
                H_z = (n_right_z + n_left_z - 2*n_z)/(ax**2) + (n_up_z + n_down_z - 2*n_z)/(ay**2)
                
                # DMI Effective Field (Factor of 2 removed from denominator via analytical Integration By Parts)
                H_x += (n_right_z - n_left_z) / ax
                H_y += (n_up_z - n_down_z) / ay
                H_z += - (n_right_x - n_left_x) / ax - (n_up_y - n_down_y) / ay
                
                H_z -= H_scaled
                H_z -= 2 * A_scaled * n_z
                
                # --- LLG Derivative (Predictor) ---
                dot_val = n_x*H_x + n_y*H_y + n_z*H_z
                ndot_x = H_x - dot_val * n_x
                ndot_y = H_y - dot_val * n_y
                ndot_z = H_z - dot_val * n_z
                
                # Save derivative for corrector
                ndot_current[i, j, 0] = ndot_x
                ndot_current[i, j, 1] = ndot_y
                ndot_current[i, j, 2] = ndot_z
                
                # Predictor Step
                n_pred_x = n_x + dt * ndot_x
                n_pred_y = n_y + dt * ndot_y
                n_pred_z = n_z + dt * ndot_z
                
                # Normalize predictor
                norm_pred = np.sqrt(n_pred_x**2 + n_pred_y**2 + n_pred_z**2)
                spins_pred[i, j, 0] = n_pred_x / norm_pred
                spins_pred[i, j, 1] = n_pred_y / norm_pred
                spins_pred[i, j, 2] = n_pred_z / norm_pred

        # Average energies per spin
        L2 = L*L
        E_ex_x /= L2
        E_ex_y /= L2
        E_dmi_x /= L2
        E_dmi_y /= L2
        E_z /= L2
        E_a /= L2
        
        # Total Scaled Energy Density
        f_tot = (E_ex_x / ax**2) + (E_ex_y / ay**2) + (E_dmi_x / ax) + (E_dmi_y / ay) + E_z + E_a
        
        # --- PASS 2: Corrector Step ---
        for i in range(L):
            for j in range(L):
                # Periodic neighbors
                i_right = (i + 1) % L
                i_left = (i - 1 + L) % L
                j_up = (j + 1) % L
                j_down = (j - 1 + L) % L
                
                # Predictor spin states
                n_x = spins_pred[i, j, 0]
                n_y = spins_pred[i, j, 1]
                n_z = spins_pred[i, j, 2]
                
                n_right_x, n_right_y, n_right_z = spins_pred[i_right, j, 0], spins_pred[i_right, j, 1], spins_pred[i_right, j, 2]
                n_left_x, n_left_y, n_left_z = spins_pred[i_left, j, 0], spins_pred[i_left, j, 1], spins_pred[i_left, j, 2]
                
                n_up_x, n_up_y, n_up_z = spins_pred[i, j_up, 0], spins_pred[i, j_up, 1], spins_pred[i, j_up, 2]
                n_down_x, n_down_y, n_down_z = spins_pred[i, j_down, 0], spins_pred[i, j_down, 1], spins_pred[i, j_down, 2]
                
                # --- Effective Field Calculation (Corrector) ---
                H_x = (n_right_x + n_left_x - 2*n_x)/(ax**2) + (n_up_x + n_down_x - 2*n_x)/(ay**2)
                H_y = (n_right_y + n_left_y - 2*n_y)/(ax**2) + (n_up_y + n_down_y - 2*n_y)/(ay**2)
                H_z = (n_right_z + n_left_z - 2*n_z)/(ax**2) + (n_up_z + n_down_z - 2*n_z)/(ay**2)
                
                H_x += (n_right_z - n_left_z) / ax
                H_y += (n_up_z - n_down_z) / ay
                H_z += - (n_right_x - n_left_x) / ax - (n_up_y - n_down_y) / ay
                
                H_z -= H_scaled
                H_z -= 2 * A_scaled * n_z
                
                # --- LLG Derivative (Corrector) ---
                dot_val = n_x*H_x + n_y*H_y + n_z*H_z
                ndot_x_pred = H_x - dot_val * n_x
                ndot_y_pred = H_y - dot_val * n_y
                ndot_z_pred = H_z - dot_val * n_z
                
                # --- Corrector Step ---
                n_x_orig = spins_current[i, j, 0]
                n_y_orig = spins_current[i, j, 1]
                n_z_orig = spins_current[i, j, 2]
                
                ndot_x_orig = ndot_current[i, j, 0]
                ndot_y_orig = ndot_current[i, j, 1]
                ndot_z_orig = ndot_current[i, j, 2]
                
                ndot_x_avg = 0.5 * (ndot_x_orig + ndot_x_pred)
                ndot_y_avg = 0.5 * (ndot_y_orig + ndot_y_pred)
                ndot_z_avg = 0.5 * (ndot_z_orig + ndot_z_pred)
                
                n_new_x = n_x_orig + dt * ndot_x_avg
                n_new_y = n_y_orig + dt * ndot_y_avg
                n_new_z = n_z_orig + dt * ndot_z_avg
                
                # Strict Re-normalization
                norm = np.sqrt(n_new_x**2 + n_new_y**2 + n_new_z**2)
                spins_next[i, j, 0] = n_new_x / norm
                spins_next[i, j, 1] = n_new_y / norm
                spins_next[i, j, 2] = n_new_z / norm

        # Dynamic Scaling Strategy
        if abs(E_dmi_x) > 1e-12 and abs(E_ex_x) > 1e-12:
            # We strictly enforce the positive root of the energy minimization polynomial
            # ax = 2.0 * Ex / |DMI| guarantees the mathematically required increase in period when DMI is small/Exchange is large.
            ax = 2.0 * E_ex_x / abs(E_dmi_x)
        if abs(E_dmi_y) > 1e-12 and abs(E_ex_y) > 1e-12:
            ay = 2.0 * E_ex_y / abs(E_dmi_y)
            
        # Optional clamping
        if ax <= 0: ax = ax_in
        if ay <= 0: ay = ay_in

        # Check convergence
        if (global_step_start + step) > 1000 and abs(f_tot - prev_f) < tol:
            # We must break here and transfer the output
            spins_current[:] = spins_next[:]
            break
            
        prev_f = f_tot
        
        # Swap buffers for the next timestep (avoids np.zeros_like allocation)
        temp = spins_current
        spins_current = spins_next
        spins_next = temp
        
    return spins_current, f_tot, ax, ay, step


@nb.njit(cache=True)
def compute_topological_charge_numba(spins, ax, ay):
    """
    Approximate the total topological charge using centered finite differences.
    For a periodic texture, |Q| is a good proxy for the number of skyrmions.
    """
    L = spins.shape[0]
    total = 0.0

    for i in range(L):
        i_right = (i + 1) % L
        i_left = (i - 1 + L) % L
        for j in range(L):
            j_up = (j + 1) % L
            j_down = (j - 1 + L) % L

            n_x = spins[i, j, 0]
            n_y = spins[i, j, 1]
            n_z = spins[i, j, 2]

            dndx_x = (spins[i_right, j, 0] - spins[i_left, j, 0]) / (2.0 * ax)
            dndx_y = (spins[i_right, j, 1] - spins[i_left, j, 1]) / (2.0 * ax)
            dndx_z = (spins[i_right, j, 2] - spins[i_left, j, 2]) / (2.0 * ax)

            dndy_x = (spins[i, j_up, 0] - spins[i, j_down, 0]) / (2.0 * ay)
            dndy_y = (spins[i, j_up, 1] - spins[i, j_down, 1]) / (2.0 * ay)
            dndy_z = (spins[i, j_up, 2] - spins[i, j_down, 2]) / (2.0 * ay)

            cross_x = dndx_y * dndy_z - dndx_z * dndy_y
            cross_y = dndx_z * dndy_x - dndx_x * dndy_z
            cross_z = dndx_x * dndy_y - dndx_y * dndy_x

            total += n_x * cross_x + n_y * cross_y + n_z * cross_z

    return total * ax * ay / (4.0 * np.pi)


def compute_topological_charge(spins, ax, ay):
    """Python wrapper around the cached Numba topological charge kernel."""
    return float(compute_topological_charge_numba(spins, ax, ay))

# ---------------------------------------------------------
# Part D & E: Execution Wrapper
# ---------------------------------------------------------

def relax_phase(spins, L, H_scaled, A_scaled, phase_name, ax_in=1.0, ay_in=1.0, max_steps=50000, tol=1e-7, live_plot=False, live_mode="quiver", max_dt=0.05, cfl_factor=0.25, visualize_scaling=False):
    """
    Relax the given spin configuration using LLG and dynamic scaling.
    This is now a wrapper around the ultra-fast Numba integrator.
    """
    ax, ay = ax_in, ay_in
    prev_f = 0.0
    steps_done = 0
    
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax_plot = plt.subplots(figsize=(8,8))
        
        if live_mode == "quiver":
            tiles_x, tiles_y = 1, 1 # Change here number of tiles that are live plotted
            tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
            L_x, L_y = tiled_spins.shape[0], tiled_spins.shape[1]
            X, Y = np.meshgrid(np.arange(L_x), np.arange(L_y))
            
            U = tiled_spins[:, :, 0].T
            V = tiled_spins[:, :, 1].T
            Sz = tiled_spins[:, :, 2].T
            
            q = ax_plot.quiver(X, Y, U, V, Sz, cmap="coolwarm", pivot='mid', scale=max(L_x, L_y)*0.8, width=0.005)
            q.set_clim(-1, 1)
            
            rects = []
            for i in range(tiles_x):
                for j in range(tiles_y):
                    rect = plt.Rectangle((-0.5 + i*L, -0.5 + j*L), L, L, 
                                       fill=False, edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax_plot.add_patch(rect)
                    rects.append((i, j, rect))
                    
            ax_plot.set_xlim(-1, L_x)
            ax_plot.set_ylim(-1, L_y)
            ax_plot.set_aspect('equal')
        else:
            im = ax_plot.imshow(spins[:, :, 2], vmin=-1, vmax=1, cmap='bwr', origin='lower', extent=[0, L, 0, L])
            ax_plot.set_aspect('equal')

        ax_plot.set_title(f"Relaxing {phase_name}...")
        fig.tight_layout()
        plt.show()
    
    chunk = 10 if live_plot else max_steps
    
    while steps_done < max_steps:
        # Numba executes cleanly in small chunks so we can visualize intermediate states
        spins_final, f_tot, ax, ay, steps_taken = relax_phase_numba(
            spins, L, H_scaled, A_scaled, chunk, tol, ax, ay, prev_f, max_dt, cfl_factor, global_step_start=steps_done
        )
        
        # The true number of steps executed is steps_taken + 1 because the loop is 0-indexed and returns `step`
        steps_done += (steps_taken + 1)
        prev_f = f_tot
        spins = spins_final
        
        if live_plot:
            if live_mode == "quiver":
                tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
                U = tiled_spins[:, :, 0].T
                V = tiled_spins[:, :, 1].T
                Sz = tiled_spins[:, :, 2].T
                q.set_UVC(U, V, Sz)
                
                # Update positions for dynamic scaling
                if visualize_scaling:
                    pts = np.column_stack((X.flatten() * ax, Y.flatten() * ay))
                    q.set_offsets(pts)
                    
                    for (i, j, rect) in rects:
                        rect.set_xy(((-0.5 + i*L)*ax, (-0.5 + j*L)*ay))
                        rect.set_width(L * ax)
                        rect.set_height(L * ay)
                        
                    ax_plot.set_xlim(-ax, L_x * ax)
                    ax_plot.set_ylim(-ay, L_y * ay)
            else:
                im.set_data(spins[:, :, 2])
                if visualize_scaling:
                    im.set_extent([0, L*ax, 0, L*ay])
                    ax_plot.set_xlim(0, L*ax)
                    ax_plot.set_ylim(0, L*ay)
                
            ax_plot.set_title(f"[{phase_name}] Step {steps_done} | f={f_tot:.4f} | ax={ax:.2f}, ay={ay:.2f}")
            plt.pause(0.02)
            
        if steps_taken + 1 < chunk:
            break
            
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.close(fig)
    
    print(f"[{phase_name}] Relaxed in {steps_done} steps. Energy Density: {f_tot:.5f} (ax={ax:.3f}, ay={ay:.3f})")
    return spins, f_tot, ax, ay

def get_FM_energy(H_scaled, A_scaled):
    """Calculate the exact analytical energy of the FM state."""
    # Aligned FM (nz = 1 or -1)
    # The energy is E = A_s n_z^2 + H n_z
    e_aligned = A_scaled + H_scaled # For nz = +1
    e_anti_aligned = A_scaled - H_scaled # For nz = -1
    
    # Tilted FM (occurs when |H| < 2|A_s| and A_s > 0 usually)
    # nz = H / (2 A_s)
    e_tilted = float('inf')
    if A_scaled != 0:
        nz_tilted = H_scaled / (2 * A_scaled)
        if abs(nz_tilted) <= 1.0:
            e_tilted = - (H_scaled**2) / (4 * A_scaled)
            
    return min(e_aligned, e_anti_aligned, e_tilted)

def compare_phases(H_scaled=0.08, A_scaled=0.5, L=64, npy_file=None, plot_ansatz=False, live_plot=False, live_mode="quiver", max_dt=0.05, cfl_factor=0.25, visualize_scaling=False, plot_groundstate=False, save_outputs=True, output_root=None, skx_multiplier=1, sc_multiplier=1, sp_multiplier=1, return_details=False):
    """
    Main Execution: Tests SkX, SP, and FM to find the true numerical ground state.
    """
    print(f"--- Phase Stability Analysis H={H_scaled}, As={A_scaled} ---")
    results = {}
    skx_multiplier = _validate_period_multiplier("skx_multiplier", skx_multiplier)
    sc_multiplier = _validate_period_multiplier("sc_multiplier", sc_multiplier)
    sp_multiplier = _validate_period_multiplier("sp_multiplier", sp_multiplier)
    output_root = get_output_root(output_root)
    ansatz_dir = output_root / "LLG" / "Ansatze"
    groundstate_dir = output_root / "LLG" / "Groundstates"
    
    # 1. Skyrmion Lattice
    print("Initializing SkX...")
    spins_skx, ax_skx, ay_skx = init_SkX(L, period_multiplier=skx_multiplier)
    skx_ansatz_path = ansatz_dir / "ansatz_SkX.npz"
    if save_outputs or plot_ansatz:
        ensure_parent_dir(skx_ansatz_path)
        np.savez(skx_ansatz_path, spins=spins_skx, ax=ax_skx, ay=ay_skx, period_multiplier=skx_multiplier)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SkX Ansatz...")
            plot_periodic_structure(str(skx_ansatz_path), tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_skx, ay=ay_skx)
        except: pass
    spins_skx, f_skx, final_ax_skx, final_ay_skx = relax_phase(spins_skx, L, H_scaled, A_scaled, "SkX", ax_in=ax_skx, ay_in=ay_skx, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor, visualize_scaling=visualize_scaling)
    results["SkX"] = f_skx
    
    # 2. Square Cell 
    print("Initializing SC...")
    spins_sc, ax_sc, ay_sc = init_SC(L, period_multiplier=sc_multiplier)
    sc_ansatz_path = ansatz_dir / "ansatz_SC.npz"
    if save_outputs or plot_ansatz:
        ensure_parent_dir(sc_ansatz_path)
        np.savez(sc_ansatz_path, spins=spins_sc, ax=ax_sc, ay=ay_sc, period_multiplier=sc_multiplier)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SC Ansatz...")
            plot_periodic_structure(str(sc_ansatz_path), tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_sc, ay=ay_sc)
        except: pass
    spins_sc, f_sc, final_ax_sc, final_ay_sc = relax_phase(spins_sc, L, H_scaled, A_scaled, "SC", ax_in=ax_sc, ay_in=ay_sc, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor, visualize_scaling=visualize_scaling)
    results["SC"] = f_sc
    
    # 3. Spiral Phase
    print("Initializing SP...")
    spins_sp, ax_sp, ay_sp = init_SP(L, period_multiplier=sp_multiplier)
    sp_ansatz_path = ansatz_dir / "ansatz_SP.npz"
    if save_outputs or plot_ansatz:
        ensure_parent_dir(sp_ansatz_path)
        np.savez(sp_ansatz_path, spins=spins_sp, ax=ax_sp, ay=ay_sp, period_multiplier=sp_multiplier)
    if plot_ansatz:
        try:
            from periodic_plotting import plot_periodic_structure
            print("Displaying SP Ansatz...")
            plot_periodic_structure(str(sp_ansatz_path), tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_sp, ay=ay_sp)
        except: pass
    spins_sp, f_sp, final_ax_sp, final_ay_sp = relax_phase(spins_sp, L, H_scaled, A_scaled, "SP", ax_in=ax_sp, ay_in=ay_sp, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor, visualize_scaling=visualize_scaling)
    results["SP"] = f_sp
    
    # 4. Ferromagnetic
    f_fm = get_FM_energy(H_scaled, A_scaled)
    print(f"[FM] Analytical Energy Density: {f_fm:.5f}")
    results["FM"] = f_fm
    candidate_energies = dict(results)
    
    # 5. Custom Ansatz (Optional)
    if npy_file and os.path.exists(npy_file):
        print(f"Initializing from custom file {npy_file}...")
        spins_cust, ax_cust, ay_cust = load_ansatz(npy_file, L)
        if plot_ansatz:
            try:
                from periodic_plotting import plot_periodic_structure
                print("Displaying Custom Ansatz...")
                plot_periodic_structure(npy_file, tiles_x=2, tiles_y=2, display_mode="quiver", ax=ax_cust, ay=ay_cust)
            except: pass
        spins_cust, f_cust, final_ax_cust, final_ay_cust = relax_phase(spins_cust, L, H_scaled, A_scaled, "Custom", ax_in=ax_cust, ay_in=ay_cust, live_plot=live_plot, live_mode=live_mode, max_dt=max_dt, cfl_factor=cfl_factor)
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
    best_ax = 1.0
    best_ay = 1.0
    if winner == "SkX": 
        best_spins = spins_skx
        best_ax = final_ax_skx
        best_ay = final_ay_skx
    elif winner == "SC": 
        best_spins = spins_sc
        best_ax = final_ax_sc
        best_ay = final_ay_sc
    elif winner == "SP": 
        best_spins = spins_sp
        best_ax = final_ax_sp
        best_ay = final_ay_sp
    elif winner == "Custom":
        best_spins = spins_cust
        best_ax = final_ax_cust
        best_ay = final_ay_cust
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
        topological_charge = compute_topological_charge(best_spins, best_ax, best_ay)
        out_name = groundstate_dir / f"LLG_groundstate_L{L}_A{A_scaled:.2f}_H{H_scaled:.2f}.npz"
        if save_outputs or plot_groundstate:
            ensure_parent_dir(out_name)
            np.savez(
                out_name,
                spins=best_spins,
                ax=best_ax,
                ay=best_ay,
                topological_charge=topological_charge,
                winner=winner,
                H_scaled=H_scaled,
                A_scaled=A_scaled,
                lattice_size=L,
                skx_multiplier=skx_multiplier,
                sc_multiplier=sc_multiplier,
                sp_multiplier=sp_multiplier,
            )
            print(f"Saved numerical ground state to '{out_name}'")
        
        if plot_groundstate:
            # Auto-plot
            try:
                from periodic_plotting import plot_periodic_structure
                print("Launching periodic plot...")
                plot_periodic_structure(str(out_name), tiles_x=2, tiles_y=2, display_mode="quiver", ax=best_ax, ay=best_ay)
            except Exception as e:
                print(f"Could not load periodic_plotting to display: {e}")
        print(f"[{winner}] Approximate topological charge Q = {topological_charge:.4f}")
    else:
        topological_charge = float("nan")

    if return_details:
        details = {
            "winner": winner,
            "H_scaled": float(H_scaled),
            "A_scaled": float(A_scaled),
            "lattice_size": int(L),
            "ax": float(best_ax),
            "ay": float(best_ay),
            "topological_charge": float(topological_charge),
            "skx_multiplier": int(skx_multiplier),
            "sc_multiplier": int(sc_multiplier),
            "sp_multiplier": int(sp_multiplier),
            "energies": {key: float(val) for key, val in results.items()},
            "candidate_energies": {key: float(val) for key, val in candidate_energies.items()},
        }
        return winner, results, details

    return winner, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic LLG Phase Analyzer")
    parser.add_argument("--H", type=float, default=1.0, help="Scaled magnetic field")
    parser.add_argument("--A", type=float, default=0.8, help="Scaled Anisotropy")
    parser.add_argument("--L", type=int, default=64, help="Grid size L")
    parser.add_argument("--npy", type=str, default=None, help="Optional MC .npy or .npz file to use as an ansatz")
    parser.add_argument("--plot-ansatz", action="store_true", help="Plot each ansatz configuration before relaxing")
    parser.add_argument("--live-plot", action="store_true", help="Plot the real-time evolution of the solver")
    parser.add_argument("--live-mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for live plotting")
    parser.add_argument("--vis-scale", action="store_true", help="Turn on dynamic scaling visualization during live plotting")
    parser.add_argument("--plot-groundstate", action="store_true", help="Plot the final ground state configuration at the end")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Maximum integration timestep")
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL stability factor for dynamic timestep")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for outputs, e.g. $SCRATCH/skyrmion_runs")
    parser.add_argument("--skx-multiplier", type=int, default=1, help="Increase the number of SkX texture periods in the periodic unit cell")
    parser.add_argument("--sc-multiplier", type=int, default=1, help="Increase the number of square-cell texture periods in the periodic unit cell")
    parser.add_argument("--sp-multiplier", type=int, default=1, help="Increase the number of spiral texture periods in the periodic unit cell")
    
    args = parser.parse_args()
    
    compare_phases(
        H_scaled=args.H,
        A_scaled=args.A,
        L=args.L,
        npy_file=args.npy,
        plot_ansatz=args.plot_ansatz,
        live_plot=args.live_plot,
        live_mode=args.live_mode,
        max_dt=args.max_dt,
        cfl_factor=args.cfl,
        visualize_scaling=args.vis_scale,
        plot_groundstate=args.plot_groundstate,
        output_root=args.output_root,
        skx_multiplier=args.skx_multiplier,
        sc_multiplier=args.sc_multiplier,
        sp_multiplier=args.sp_multiplier,
    )
