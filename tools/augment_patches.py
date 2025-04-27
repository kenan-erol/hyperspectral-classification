import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt # <-- Add matplotlib import
import sys # <-- Add sys import

# --- Add src directory to sys.path if needed ---
# (Assuming log_utils is in src relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes tools/ is one level down from root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
     sys.path.append(src_path)
try:
    # Import your HSI display function
    from log_utils import hsi_to_rgb_display
except ImportError:
    print("Warning: Could not import hsi_to_rgb_display from src.log_utils. Visualization saving will fail.")
    hsi_to_rgb_display = None
# --- End Path Setup ---


def add_gaussian_noise(patch, mean=0.0, std_dev=0.1):
    """
    Adds Gaussian noise to a patch.
    Assumes patch is a numpy array (H, W, C) with float values.
    """
    if patch is None or patch.size == 0:
        return None

    # Generate noise with the same shape as the patch
    noise = np.random.normal(mean, std_dev, patch.shape).astype(patch.dtype)

    # Add noise to the patch
    noisy_patch = patch + noise

    # Clip values to maintain a reasonable range (e.g., 0 to 1 if normalized, or original range)
    # Determine min/max based on original patch if needed, or assume a common range like [0, 1] or [0, inf)
    # For simplicity, let's clip based on the original patch's min/max if possible, otherwise >= 0
    patch_min = np.min(patch) if patch.size > 0 else 0
    patch_max = np.max(patch) if patch.size > 0 else 1 # Assume max of 1 if min is 0, otherwise needs context
    if patch_min >= 0 and patch_max > patch_min: # If range seems like [0, X]
         noisy_patch = np.clip(noisy_patch, patch_min, patch_max)
    elif patch_min < 0: # If data can be negative, clip less aggressively or based on context
         # This part might need adjustment based on your data's actual range
         pass # Or clip based on some known bounds if available
    else: # Fallback: ensure non-negative if original was non-negative
        if patch_min >= 0:
            noisy_patch = np.clip(noisy_patch, 0, None)


    return noisy_patch

def main(args):
    print(f"Starting augmentation process...")
    print(f"Input directory: {args.input_dir}")
    # --- Use input_label_filename ---
    print(f"Input label file: {args.input_label_filename}")
    # --- End ---
    print(f"Output directory: {args.output_dir}")
    print(f"Noise standard deviation: {args.noise_std_dev}")
    # --- Print visualization info ---
    if args.visualize_count > 0:
        if not hsi_to_rgb_display:
            print("Warning: Cannot visualize because hsi_to_rgb_display function is not available.")
            args.visualize_count = 0 # Disable visualization
        elif not args.visualize_dir:
            print("Warning: --visualize_dir not specified. Visualizations will not be saved.")
            args.visualize_count = 0 # Disable visualization
        else:
            print(f"Saving {args.visualize_count} visualization(s) to: {args.visualize_dir}")
            os.makedirs(args.visualize_dir, exist_ok=True)
    # --- End visualization info ---

    # --- Construct input label file path using the argument ---
    input_label_file = os.path.join(args.input_dir, args.input_label_filename)
    # --- End ---
    output_label_file = os.path.join(args.output_dir, 'labels.txt') # Keep output name standard
    output_patches_base_dir = os.path.join(args.output_dir) # Save patches directly in output_dir mirroring structure

    os.makedirs(output_patches_base_dir, exist_ok=True) # Ensure output base exists

    augmented_samples = []
    processed_count = 0
    viz_saved_count = 0 # <-- Keep track of saved visualizations

    try:
        with open(input_label_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input label file not found at {input_label_file}")
        exit(1)

    # Determine number of patches to process
    num_to_process = len(lines)
    indices_to_process = list(range(len(lines)))
    if args.max_patches is not None and args.max_patches > 0:
        if args.max_patches < len(lines):
            print(f"Processing a random subset of {args.max_patches} patches.")
            indices_to_process = random.sample(range(len(lines)), args.max_patches)
            num_to_process = args.max_patches
        else:
            print(f"Processing all {len(lines)} patches (max_patches >= total patches).")
    else:
        print(f"Processing all {len(lines)} patches.")


    for idx in tqdm(indices_to_process, desc="Augmenting Patches"):
        line = lines[idx].strip()
        if not line: continue

        try:
            relative_patch_path, label_str = line.rsplit(' ', 1)
            label = int(label_str)

            # Construct full path to the original patch (relative to input_dir)
            # Assumes relative_patch_path in label file is like 'patches/Drug/...'
            full_input_patch_path = os.path.join(args.input_dir, relative_patch_path)

            if not os.path.exists(full_input_patch_path):
                print(f"Warning: Original patch not found: {full_input_patch_path}, skipping.")
                continue

            # Load the original patch
            original_patch = np.load(full_input_patch_path)

            # Apply augmentation (Gaussian noise)
            augmented_patch = add_gaussian_noise(original_patch, std_dev=args.noise_std_dev)

            if augmented_patch is None:
                print(f"Warning: Augmentation failed for {full_input_patch_path}, skipping.")
                continue

            # --- Save Visualization (if requested) ---
            save_this_viz = viz_saved_count < args.visualize_count and hsi_to_rgb_display is not None
            if save_this_viz:
                try:
                    original_rgb = hsi_to_rgb_display(original_patch)
                    augmented_rgb = hsi_to_rgb_display(augmented_patch)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(original_rgb)
                    axes[0].set_title(f"Original - Idx {idx}")
                    axes[0].axis('off')

                    axes[1].imshow(augmented_rgb)
                    axes[1].set_title(f"Augmented (Noise std={args.noise_std_dev})")
                    axes[1].axis('off')

                    plt.tight_layout()
                    base_filename = os.path.splitext(os.path.basename(relative_patch_path))[0]
                    viz_filename = f"viz_{idx}_{base_filename}.png"
                    full_viz_path = os.path.join(args.visualize_dir, viz_filename)
                    plt.savefig(full_viz_path)
                    plt.close(fig)
                    viz_saved_count += 1
                except Exception as viz_e:
                    print(f"\nWarning: Failed to save visualization for index {idx}: {viz_e}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                         plt.close(fig)
            # --- End Save Visualization ---


            # --- Save the augmented patch ---
            # Define output path, mirroring the subdirectory structure within output_dir
            # relative_patch_path is like 'patches/DrugName/Group/Mxxx/measurement_patch_y.npy'
            output_relative_path = relative_patch_path # Keep the same relative path structure
            full_output_patch_path = os.path.join(args.output_dir, output_relative_path)

            # Create the necessary subdirectories in the output folder
            output_subdir = os.path.dirname(full_output_patch_path)
            os.makedirs(output_subdir, exist_ok=True)

            # Save the augmented patch
            np.save(full_output_patch_path, augmented_patch.astype(np.float32))

            # Store info for the new label file (using the same relative path)
            augmented_samples.append((output_relative_path, label))
            processed_count += 1

        except Exception as e:
            print(f"\nError processing line '{line}': {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nFinished augmenting {processed_count} patches.")
    if viz_saved_count > 0:
        print(f"Saved {viz_saved_count} visualization(s) to {args.visualize_dir}")

    if not augmented_samples:
        print("Error: No patches were successfully augmented.")
        exit(1)

    # Save the new label file
    print(f"Saving new label file to {output_label_file}")
    try:
        with open(output_label_file, 'w') as f:
            for rel_path, lbl in augmented_samples:
                f.write(f"{rel_path} {lbl}\n")
    except Exception as e:
        print(f"Error writing new label file {output_label_file}: {e}")
        exit(1)

    print("Augmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment preprocessed hyperspectral patches with noise.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the patch subdirectories and the input label file.') # Modified help text
    # --- Add input_label_filename argument ---
    parser.add_argument('--input_label_filename', type=str, default='labels.txt',
                        help='Filename of the input label file within input_dir (default: labels.txt)')
    # --- End ---
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the augmented patches and the new labels.txt')
    parser.add_argument('--noise_std_dev', type=float, default=0.05,
                        help='Standard deviation of the Gaussian noise to add.')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum number of patches to augment (default: process all)')
    # --- Add visualization arguments ---
    parser.add_argument('--visualize_count', type=int, default=0,
                        help='Number of sample patches to visualize (original vs. augmented). Default: 0')
    parser.add_argument('--visualize_dir', type=str, default=None,
                        help='Directory to save visualization plots. Required if visualize_count > 0.')
    # --- End visualization arguments ---

    args = parser.parse_args()

    # --- Add validation for visualization args ---
    if args.visualize_count > 0 and not args.visualize_dir:
        parser.error("--visualize_dir is required when --visualize_count > 0")
    # --- End validation ---

    main(args)

# --- Example Command ---
# python tools/augment_patches.py \
#   --input_dir ./data_processed_patch \
#   --input_label_filename labels_patches.txt \
#   --output_dir ./data_augmented_noise_0.05 \
#   --noise_std_dev 0.05 \
#   --visualize_count 5 \
#   --visualize_dir ./data_augmented_noise_0.05/visualizations
# --- End Example Command ---