import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

# --- Configuration ---
PATCHES_DIR = '../data_a2s2k/patches' # Directory containing the .npy patches
NUM_PATCHES_TO_SHOW = 5 # How many patches to display (adjust as needed)
# Choose how to select channels for RGB visualization
# Option 1: Use specific indices (e.g., for R, G, B)
# CHANNEL_INDICES = [50, 30, 10] # Example indices, adjust based on your data
# Option 2: Use the first 3 channels
CHANNEL_INDICES = [0, 1, 2]
# Option 3: Use mean if fewer than 3 channels (set CHANNEL_INDICES = None)
# CHANNEL_INDICES = None
# --- End Configuration ---

def visualize_patches(patches_dir, num_to_show, channel_indices):
    """Loads and displays a grid of .npy patches."""
    patch_files = glob.glob(os.path.join(patches_dir, '*.npy'))

    if not patch_files:
        print(f"Error: No .npy files found in '{patches_dir}'")
        return

    num_to_show = min(num_to_show, len(patch_files))
    if num_to_show == 0:
        print("No patches to show.")
        return

    # Select a subset of files to show
    files_to_show = patch_files[:num_to_show]

    # Determine grid size
    cols = math.ceil(math.sqrt(num_to_show))
    rows = math.ceil(num_to_show / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() # Flatten to easily iterate

    print(f"Loading and displaying {num_to_show} patches...")

    for i, patch_file in enumerate(files_to_show):
        try:
            # Load the patch data (assuming H, W, C format)
            patch_data = np.load(patch_file)

            if patch_data.ndim != 3:
                print(f"Warning: Skipping {os.path.basename(patch_file)}, expected 3 dimensions, got {patch_data.ndim}")
                axes[i].set_title(f"{os.path.basename(patch_file)}\n(Invalid Dims)")
                axes[i].axis('off')
                continue

            h, w, c = patch_data.shape

            # Select channels for display
            if channel_indices and len(channel_indices) == 3:
                if c >= max(channel_indices) + 1:
                    display_patch = patch_data[:, :, channel_indices]
                else:
                    print(f"Warning: Not enough channels in {os.path.basename(patch_file)} for specified indices. Using first 3 or mean.")
                    if c >= 3:
                        display_patch = patch_data[:, :, :3]
                    elif c > 0:
                        display_patch = np.mean(patch_data, axis=2, keepdims=True)
                        display_patch = np.concatenate([display_patch] * 3, axis=-1)
                    else:
                        display_patch = np.zeros((h, w, 3)) # Empty patch
            elif c >= 3: # Default to first 3 if no indices specified
                 display_patch = patch_data[:, :, :3]
            elif c > 0: # Handle 1 or 2 channels by repeating/averaging
                 display_patch = np.mean(patch_data, axis=2, keepdims=True)
                 display_patch = np.concatenate([display_patch] * 3, axis=-1)
            else: # Handle 0 channels
                 display_patch = np.zeros((h, w, 3))

            # Normalize for display (0-1 range)
            min_val, max_val = np.min(display_patch), np.max(display_patch)
            if max_val > min_val:
                display_patch = (display_patch - min_val) / (max_val - min_val)
            else:
                display_patch = np.zeros_like(display_patch) # Handle constant image
            display_patch = np.clip(display_patch, 0, 1)

            # Display the patch
            ax = axes[i]
            ax.imshow(display_patch)
            ax.set_title(os.path.basename(patch_file), fontsize=8)
            ax.axis('off')

        except Exception as e:
            print(f"Error loading/displaying {os.path.basename(patch_file)}: {e}")
            axes[i].set_title(f"{os.path.basename(patch_file)}\n(Error)")
            axes[i].axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_patches(PATCHES_DIR, NUM_PATCHES_TO_SHOW, CHANNEL_INDICES)