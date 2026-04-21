import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_periodic_structure(spins_file, tiles_x=2, tiles_y=2, display_mode="quiver", cmap="coolwarm", scale_factor=0.8, ax=1.0, ay=1.0):
    """
    Load a .npy spin configuration and recursively tile it to 
    visualize the periodic boundary conditions.
    """
    if not os.path.exists(spins_file):
        print(f"Error: File '{spins_file}' not found.")
        return
        
    # Load original (L x L x 3) spins
    if spins_file.endswith('.npz'):
        data = np.load(spins_file)
        spins = data['spins']
        if 'ax' in data and ax == 1.0:
            ax = float(data['ax'])
        if 'ay' in data and ay == 1.0:
            ay = float(data['ay'])
    else:
        spins = np.load(spins_file)
        
    print(f"Loaded original spins with shape: {spins.shape}, ax={ax:.3f}, ay={ay:.3f}")
    L = spins.shape[0]
    
    # Tile it (tiles_x, tiles_y, 1) to extend the 2D grid while keeping spin depth (3) intact
    tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
    L_x = tiled_spins.shape[0]
    L_y = tiled_spins.shape[1]
    
    print(f"Tiled periodic surface is now {L_x} x {L_y}")

    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(L_x) * ax, np.arange(L_y) * ay)
    
    # Extract components (transpose to match meshgrid)
    U = tiled_spins[:, :, 0].T
    V = tiled_spins[:, :, 1].T
    Sz = tiled_spins[:, :, 2].T
    
    # Render Plot
    fig = plt.figure(figsize=(8, 8))
    
    scale_val = max(L_x, L_y) * scale_factor
    
    if display_mode == "heatmap":
        im = plt.imshow(Sz, cmap=cmap, vmin=-1, vmax=1, origin='lower', extent=[-0.5 * ax, (L_x - 0.5) * ax, -0.5 * ay, (L_y - 0.5) * ay])
        q = plt.quiver(X, Y, U, V, pivot='mid', scale=scale_val, width=0.005)
        plt.colorbar(im, label='Sz')
    elif display_mode == "quiver":
        q = plt.quiver(X, Y, U, V, Sz, cmap=cmap, pivot='mid', scale=scale_val, width=0.005)
        q.set_clim(-1, 1)
        plt.colorbar(q, label='Sz')
    
    # Draw dashed bounding boxes around the original unit cells to visualize the tiling borders
    for i in range(tiles_x):
        for j in range(tiles_y):
            rect = plt.Rectangle(((-0.5 + i*L) * ax, (-0.5 + j*L) * ay), L * ax, L * ay, 
                               fill=False, edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.5)
            plt.gca().add_patch(rect)
            
    plt.title(f'Periodic Tiling ({tiles_x}x{tiles_y}) of {os.path.basename(spins_file)}')
    plt.xlim(-ax, L_x * ax)
    plt.ylim(-ay, L_y * ay)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot periodic skyrmion lattices.")
    # The default file matches the output filename in MC_metropolis.py
    parser.add_argument("file", type=str, nargs='?', default="output/MC/npy/final_spins.npy", help="Path to .npy or .npz spin file")
    parser.add_argument("--tiles", type=int, default=2, help="Number of times to tile the lattice in both x and y directions")
    parser.add_argument("--mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for plotting")
    parser.add_argument("--ax", type=float, default=1.0, help="Lattice spacing in x")
    parser.add_argument("--ay", type=float, default=1.0, help="Lattice spacing in y")
    
    args = parser.parse_args()
    
    plot_periodic_structure(
        spins_file=args.file,
        tiles_x=args.tiles,
        tiles_y=args.tiles,
        display_mode=args.mode,
        ax=args.ax,
        ay=args.ay
    )

