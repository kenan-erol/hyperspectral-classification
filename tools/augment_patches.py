import os
import argparse
import numpy as np
from tqdm import tqdm
import random

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
    print(f"Output directory: {args.output_dir}")
    print(f"Noise standard deviation: {args.noise_std_dev}")

    input_label_file = os.path.join(args.input_dir, 'labels.txt')
    output_label_file = os.path.join(args.output_dir, 'labels.txt')
    output_patches_base_dir = os.path.join(args.output_dir, 'patches') # Mirror structure

    os.makedirs(output_patches_base_dir, exist_ok=True)

    augmented_samples = []
    processed_count = 0

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

            # Construct full path to the original patch
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

            # --- Save the augmented patch ---
            # Define output path, mirroring the subdirectory structure
            # relative_patch_path is like 'patches/DrugName/Group/Mxxx/measurement_patch_y.npy'
            output_relative_path = relative_patch_path # Keep the same relative path structure
            full_output_patch_path = os.path.join(args.output_dir, output_relative_path)

            # Create the necessary subdirectories in the output folder
            output_subdir = os.path.dirname(full_output_patch_path)
            os.makedirs(output_subdir, exist_ok=True)

            # Save the augmented patch
            np.save(full_output_patch_path, augmented_patch.astype(np.float32))

            # Store info for the new label file
            augmented_samples.append((output_relative_path, label))
            processed_count += 1

        except Exception as e:
            print(f"\nError processing line '{line}': {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nFinished augmenting {processed_count} patches.")

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
                        help='Directory containing the preprocessed patches and labels.txt (output of preproc_patch.py)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the augmented patches and the new labels.txt')
    parser.add_argument('--noise_std_dev', type=float, default=0.05,
                        help='Standard deviation of the Gaussian noise to add.')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum number of patches to augment (default: process all)')

    args = parser.parse_args()
    main(args)