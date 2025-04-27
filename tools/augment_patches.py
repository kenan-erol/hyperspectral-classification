import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import random
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt # <-- Add matplotlib import
import sys # <-- Add sys import
from PIL import Image, ImageEnhance # <-- Add PIL imports for color jitter/hue

# --- Add src directory to sys.path if needed ---
# (Assuming log_utils is in src relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes tools/ is one level down from root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
     sys.path.append(src_path)
# --- We will define our own visualization function locally ---
# try:
#     # Import your HSI display function
#     from preproc_patch import hsi_to_rgb_display # Keep for reference if needed, but use local one
# except ImportError:
#     print("Warning: Could not import hsi_to_rgb_display from src.log_utils. Visualization saving will fail.")
#     hsi_to_rgb_display = None
# --- End Path Setup ---

# --- New HSI to RGB Visualization Function ---
def visualize_hsi_patch(hsi_patch_np):
    """
    Creates a displayable RGB image (uint8) from an HSI patch (float32/64)
    using the mean across spectral channels and robust normalization.
    Handles potential edge cases like zero channels or constant values.

    Args:
        hsi_patch_np (np.ndarray): The hyperspectral patch (H, W, C).

    Returns:
        np.ndarray: An RGB image (H, W, 3) of type uint8, or None if input is invalid.
    """
    if hsi_patch_np is None or hsi_patch_np.ndim < 2:
        print("Warning: Invalid input to visualize_hsi_patch.")
        return None

    # Ensure 3 dimensions (H, W, C) even if C=1
    if hsi_patch_np.ndim == 2:
        hsi_patch_np = np.expand_dims(hsi_patch_np, axis=-1)

    img_h, img_w, img_c = hsi_patch_np.shape

    # Handle case with zero channels
    if img_c == 0:
        print("Warning: HSI patch has zero channels.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Calculate the mean across the spectral dimension
    try:
        # Use float64 for intermediate mean calculation to avoid potential overflow/precision issues
        display_img = np.mean(hsi_patch_np, axis=2, dtype=np.float64)
    except Exception as e:
        print(f"Warning: Error calculating mean in visualize_hsi_patch: {e}")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8) # Return black image on error

    # Normalize the mean image to 0-255 range
    min_val = np.min(display_img)
    max_val = np.max(display_img)

    if max_val > min_val:
        # Perform normalization robustly
        display_img = (display_img - min_val) / (max_val - min_val) * 255.0
    elif max_val == min_val:
        # Handle constant value image - map to mid-gray or black/white
        # Mapping to 0 avoids potential issues if the constant value was negative after mean
        display_img = np.zeros((img_h, img_w), dtype=np.float64)
    else: # Should not happen if min/max are calculated correctly, but as a fallback
        display_img = np.zeros((img_h, img_w), dtype=np.float64)

    # Convert to uint8 and stack to create 3-channel RGB
    display_img_uint8 = display_img.astype(np.uint8)
    rgb_display = np.stack([display_img_uint8] * 3, axis=-1)

    return rgb_display
# --- End New HSI to RGB Visualization Function ---


def add_gaussian_noise(patch, mean=0.0, std_dev=0.1):
    """
    Adds Gaussian noise to a raw HSI patch.
    Operates directly on the hyperspectral data before saving.
    Assumes patch is a numpy array (H, W, C) with float values.
    """
    if patch is None or patch.size == 0:
        return None

    # Generate noise with the same shape as the patch
    noise = np.random.normal(mean, std_dev, patch.shape).astype(patch.dtype)

    # Add noise to the patch
    noisy_patch = patch + noise

    # Clip values to maintain a reasonable range.
    # Since HSI data might not be normalized to [0,1] and can have negative values
    # after processing, a simple clip to [0, max] might be incorrect.
    # A more robust approach might involve clipping based on original range percentiles,
    # but for simplicity, we'll just ensure values don't become excessively large/small
    # relative to the original range, or simply clip >= 0 if the original was non-negative.
    patch_min = np.min(patch) if patch.size > 0 else 0
    # patch_max = np.max(patch) if patch.size > 0 else 1 # Max might not be meaningful

    if patch_min >= 0:
        # If original data was non-negative, keep augmented data non-negative
        noisy_patch = np.clip(noisy_patch, 0, None)
    # else:
        # If original data could be negative, avoid aggressive clipping unless bounds are known.
        # Consider clipping based on N standard deviations from original mean if needed.
        # For now, no clipping if original could be negative.
        # pass

    return noisy_patch

# --- New Augmentation Functions for Visualization ---
def apply_random_color_jitter(image_pil, brightness=0.2, contrast=0.2, saturation=0.2):
    """
    Applies random color jitter (brightness, contrast, saturation) to a PIL Image.
    Used ONLY for the visualization, not saved in the .npy file.
    """
    if image_pil is None: return None

    # Brightness
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(random.uniform(1 - brightness, 1 + brightness))

    # Contrast
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(random.uniform(1 - contrast, 1 + contrast))

    # Saturation
    enhancer = ImageEnhance.Color(image_pil) # Saturation is called Color in PIL
    image_pil = enhancer.enhance(random.uniform(1 - saturation, 1 + saturation))

    return image_pil

def apply_random_hue_shift(image_pil, max_hue_delta=0.1):
    """
    Applies a random hue shift to a PIL Image.
    Used ONLY for the visualization, not saved in the .npy file.
    max_hue_delta is fraction of 360 degrees (e.g., 0.1 means +/- 36 degrees).
    """
    if image_pil is None: return None

    try:
        # Convert PIL image to HSV numpy array
        img_hsv = np.array(image_pil.convert('HSV'))

        # Calculate hue shift amount (delta is in range [0, 255] for uint8 HSV)
        hue_delta = int(random.uniform(-max_hue_delta, max_hue_delta) * 255)

        # Apply hue shift with wrap-around using modulo arithmetic
        # Ensure intermediate calculations use a larger type to prevent overflow before modulo
        hue_channel = img_hsv[:, :, 0].astype(np.int32)
        shifted_hue = (hue_channel + hue_delta) % 256 # Modulo 256 for uint8 range
        img_hsv[:, :, 0] = shifted_hue.astype(np.uint8)

        # Convert back to PIL Image
        image_pil = Image.fromarray(img_hsv, 'HSV').convert('RGB')
    except Exception as e:
        print(f"Warning: Error applying hue shift: {e}")
        # Return original image if conversion fails
        return image_pil.convert('RGB') # Ensure it's RGB

    return image_pil
# --- End New Augmentation Functions ---


def main(args):
    print(f"Starting augmentation process...")
    print(f"Input directory: {args.input_dir}")
    print(f"Input label file: {args.input_label_filename}")
    print(f"Output directory: {args.output_dir}")
    print(f"Noise standard deviation: {args.noise_std_dev}")
    # --- Print visualization info ---
    if args.visualize_count > 0:
        if not args.visualize_dir:
            print("Warning: --visualize_dir not specified. Visualizations will not be saved.")
            args.visualize_count = 0 # Disable visualization
        else:
            print(f"Saving {args.visualize_count} visualization(s) to: {args.visualize_dir}")
            print(f"Visualization includes noise, random jitter, and hue shift.")
            os.makedirs(args.visualize_dir, exist_ok=True)
    # --- End visualization info ---

    input_label_file = os.path.join(args.input_label_filename)
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

            # --- Apply Gaussian noise augmentation to the raw HSI data ---
            # This is the primary augmentation saved to the .npy file.
            augmented_patch_hsi = add_gaussian_noise(original_patch, std_dev=args.noise_std_dev)

            if augmented_patch_hsi is None:
                print(f"Warning: Noise augmentation failed for {full_input_patch_path}, skipping.")
                continue

            # --- Save Visualization (if requested) ---
            save_this_viz = viz_saved_count < args.visualize_count
            if save_this_viz:
                try:
                    # Convert original and augmented HSI to RGB for display using the local function
                    original_rgb = visualize_hsi_patch(original_patch)
                    augmented_rgb_noise_only = visualize_hsi_patch(augmented_patch_hsi) # Visualize effect of noise

                    if original_rgb is not None and augmented_rgb_noise_only is not None:
                        # --- Apply Jitter and Hue Shift ONLY to the augmented visualization ---
                        # Convert augmented RGB to PIL Image for these transforms
                        augmented_pil = Image.fromarray(augmented_rgb_noise_only)

                        # Apply color jitter (brightness, contrast, saturation)
                        augmented_pil_jittered = apply_random_color_jitter(augmented_pil)

                        # Apply hue shift
                        augmented_pil_final_viz = apply_random_hue_shift(augmented_pil_jittered)

                        # Convert final visualization back to numpy array
                        augmented_rgb_final_viz = np.array(augmented_pil_final_viz)
                        # --- End Jitter/Hue application for visualization ---

                        # Create plot
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(original_rgb)
                        axes[0].set_title('Original Patch (RGB Vis)')
                        axes[0].axis('off')

                        axes[1].imshow(augmented_rgb_final_viz)
                        axes[1].set_title('Augmented Vis (Noise+Jitter+Hue)')
                        axes[1].axis('off')

                        plt.suptitle(f"Sample {viz_saved_count}: {os.path.basename(relative_patch_path)}\nLabel: {label}, Noise StdDev: {args.noise_std_dev}", fontsize=10)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

                        # Construct visualization filename
                        base_filename = os.path.splitext(os.path.basename(relative_patch_path))[0]
                        viz_filename = os.path.join(args.visualize_dir, f"{base_filename}_augmentation_viz_{viz_saved_count}.png")
                        plt.savefig(viz_filename, dpi=150)
                        plt.close(fig)
                        viz_saved_count += 1
                    else:
                        print(f"Warning: Could not generate RGB for visualization for {relative_patch_path}")

                except Exception as viz_e:
                    print(f"\nError creating visualization for {relative_patch_path}: {viz_e}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig) # Close plot if open
            # --- End Save Visualization ---


            # --- Save the augmented HSI patch (with noise only) ---
            # Define output path, mirroring the subdirectory structure within output_dir
            output_relative_path = relative_patch_path # Keep the same relative path structure
            full_output_patch_path = os.path.join(args.output_dir, output_relative_path)

            # Create the necessary subdirectories in the output folder
            output_subdir = os.path.dirname(full_output_patch_path)
            os.makedirs(output_subdir, exist_ok=True)

            # Save the augmented patch (ensure float32)
            np.save(full_output_patch_path, augmented_patch_hsi.astype(np.float32))

            # Store info for the new label file (using the same relative path)
            augmented_samples.append((output_relative_path, label))
            processed_count += 1

        except Exception as e:
            print(f"\nError processing line '{line}': {e}")
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
        # Sort lines for consistency before writing
        augmented_samples.sort()
        with open(output_label_file, 'w') as f:
            for rel_path, lbl in augmented_samples:
                 # Use as_posix() for consistent forward slashes in the label file
                 f.write(f"{Path(rel_path).as_posix()} {lbl}\n")
    except Exception as e:
        print(f"Error writing new label file {output_label_file}: {e}")
        exit(1)

    print("Augmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment preprocessed hyperspectral patches with noise.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the patch subdirectories.') # Modified help text
    parser.add_argument('--input_label_filename', type=str, default='labels.txt',
                        help='Filename of the input label file relative to the project root or an absolute path (default: labels.txt)') # Clarified path expectation
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the augmented patches and the new labels.txt')
    parser.add_argument('--noise_std_dev', type=float, default=0.05,
                        help='Standard deviation of the Gaussian noise to add to raw HSI data.')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum number of patches to augment (default: process all)')
    parser.add_argument('--visualize_count', type=int, default=0,
                        help='Number of sample patches to visualize (original vs. augmented with noise+jitter+hue). Default: 0')
    parser.add_argument('--visualize_dir', type=str, default=None,
                        help='Directory to save visualization plots. Required if visualize_count > 0.')

    args = parser.parse_args()

    # --- Add validation for visualization args ---
    if args.visualize_count > 0 and not args.visualize_dir:
        parser.error("--visualize_dir is required when --visualize_count > 0")
    # --- End validation ---

    # --- Resolve input label file path ---
    # Check if it's absolute, otherwise assume relative to project root
    if not os.path.isabs(args.input_label_filename):
        args.input_label_filename = os.path.join(project_root, args.input_label_filename)
        print(f"Resolved input label file path to: {args.input_label_filename}")
    # --- End resolve ---


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