import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_periodic_structure(spins_file, tiles_x=2, tiles_y=2, display_mode="quiver", cmap="coolwarm", scale_factor=0.8):
    """
    Load a .npy spin configuration and recursively tile it to 
    visualize the periodic boundary conditions.
    """
    if not os.path.exists(spins_file):
        print(f"Error: File '{spins_file}' not found.")
        return
        
    # Load original (L x L x 3) spins
    spins = np.load(spins_file)
    print(f"Loaded original spins with shape: {spins.shape}")
    L = spins.shape[0]
    
    # Tile it (tiles_x, tiles_y, 1) to extend the 2D grid while keeping spin depth (3) intact
    tiled_spins = np.tile(spins, (tiles_x, tiles_y, 1))
    L_x = tiled_spins.shape[0]
    L_y = tiled_spins.shape[1]
    
    print(f"Tiled periodic surface is now {L_x} x {L_y}")

    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(L_x), np.arange(L_y))
    
    # Extract components (transpose to match meshgrid)
    U = tiled_spins[:, :, 0].T
    V = tiled_spins[:, :, 1].T
    Sz = tiled_spins[:, :, 2].T
    
    # Render Plot
    fig = plt.figure(figsize=(8, 8))
    
    if display_mode == "heatmap":
        im = plt.imshow(Sz, cmap=cmap, vmin=-1, vmax=1, origin='lower', extent=[-0.5, L_x-0.5, -0.5, L_y-0.5])
        q = plt.quiver(X, Y, U, V, pivot='mid', scale=max(L_x, L_y)*scale_factor, width=0.005)
        plt.colorbar(im, label='Sz')
    elif display_mode == "quiver":
        q = plt.quiver(X, Y, U, V, Sz, cmap=cmap, pivot='mid', scale=max(L_x, L_y)*scale_factor, width=0.005)
        q.set_clim(-1, 1)
        plt.colorbar(q, label='Sz')
    
    # Draw dashed bounding boxes around the original unit cells to visualize the tiling borders
    for i in range(tiles_x):
        for j in range(tiles_y):
            rect = plt.Rectangle((-0.5 + i*L, -0.5 + j*L), L, L, 
                               fill=False, edgecolor='black', linestyle='--', linewidth=1.5, alpha=0.5)
            plt.gca().add_patch(rect)
            
    plt.title(f'Periodic Tiling ({tiles_x}x{tiles_y}) of {os.path.basename(spins_file)}')
    plt.xlim(-1, L_x)
    plt.ylim(-1, L_y)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot periodic skyrmion lattices.")
    # The default file matches the output filename in MC_metropolis.py
    parser.add_argument("file", type=str, nargs='?', default="final_spins.npy", help="Path to .npy spin file")
    parser.add_argument("--tiles", type=int, default=2, help="Number of times to tile the lattice in both x and y directions")
    parser.add_argument("--mode", type=str, choices=["quiver", "heatmap"], default="quiver", help="Display mode for plotting")
    
    args = parser.parse_args()
    
    plot_periodic_structure(
        spins_file=args.file,
        tiles_x=args.tiles,
        tiles_y=args.tiles,
        display_mode=args.mode
    )

