import os
import numpy as np
import random
import argparse
import sys
from PIL import Image, ImageEnhance
from tqdm import tqdm
from collections import defaultdict
import math
import matplotlib.pyplot as plt # <-- Add plt import
from pathlib import Path # <-- Add Path import

# --- Add src directory to sys.path if needed ---
# (Assuming log_utils is in src relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes tools/ is one level down from root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
     sys.path.append(src_path)
# --- End Path Setup ---

# --- HSI to RGB Visualization Function ---
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
        print("Warning: Input patch has zero channels.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Calculate the mean across the spectral dimension
    try:
        # Use nanmean to handle potential NaNs if they exist in data
        display_img = np.nanmean(hsi_patch_np, axis=2)
    except Exception as e:
        print(f"Warning: Error calculating mean across channels: {e}")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Check if result is all NaN (can happen if input was all NaN)
    if np.all(np.isnan(display_img)):
        print("Warning: Mean across channels resulted in all NaNs.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Normalize the mean image to 0-255 range
    # Use nanmin/nanmax to ignore NaNs during normalization
    min_val = np.nanmin(display_img)
    max_val = np.nanmax(display_img)

    # Check if min/max are valid numbers
    if np.isnan(min_val) or np.isnan(max_val):
         print("Warning: Could not determine valid min/max for normalization (NaNs present).")
         display_img = np.zeros((img_h, img_w), dtype=np.uint8) # Return black image
    elif max_val > min_val:
        # Normalize, ensuring we handle potential NaNs from input
        display_img = np.nan_to_num(display_img, nan=min_val) # Replace NaNs with min_val before scaling
        display_img = ((display_img - min_val) / (max_val - min_val) * 255.0)
    elif max_val == min_val:
        # Handle constant value image - map to mid-gray or black
        # print("Warning: Patch has constant value across spatial and spectral dimensions.")
        display_img = np.full((img_h, img_w), 128 if min_val != 0 else 0, dtype=float) # Mid-gray or black
    else: # max_val < min_val (should not happen with nanmin/max) or other issues
        print("Warning: Unexpected min/max values during normalization.")
        display_img = np.zeros((img_h, img_w), dtype=np.uint8) # Return black image

    # Convert to uint8 and stack to create 3-channel RGB
    display_img_uint8 = display_img.astype(np.uint8)
    rgb_display = np.stack([display_img_uint8] * 3, axis=-1)

    return rgb_display
# --- End HSI to RGB Visualization Function ---


# --- Modified Gaussian Noise Function ---
def add_gaussian_noise(patch, mean=0.0, std_dev=0.1, start_band=None, end_band=None):
    """
    Adds Gaussian noise to a raw HSI patch, optionally targeting specific bands.
    Operates directly on the hyperspectral data before saving.
    Assumes patch is a numpy array (H, W, C) with float values.

    Args:
        patch (np.ndarray): The input HSI patch (H, W, C).
        mean (float): Mean of the Gaussian noise.
        std_dev (float): Standard deviation of the Gaussian noise.
        start_band (int, optional): The starting index (inclusive) of the band range to apply noise.
        end_band (int, optional): The ending index (exclusive) of the band range to apply noise.

    Returns:
        np.ndarray: The patch with added noise.
    """
    if patch is None or patch.size == 0:
        print("Warning: Skipping noise addition for empty patch.")
        return patch

    noisy_patch = patch.copy() # Work on a copy

    # Determine the slice to apply noise
    if start_band is not None and end_band is not None:
        if start_band < 0 or end_band > patch.shape[2] or start_band >= end_band:
             print(f"Warning: Invalid band range [{start_band}:{end_band}] for patch with {patch.shape[2]} channels. Applying noise to all bands.")
             target_slice = noisy_patch # Apply to all if range is invalid
             noise_shape = patch.shape
        else:
             target_slice = noisy_patch[..., start_band:end_band]
             noise_shape = target_slice.shape
             # print(f"Debug: Applying noise to bands {start_band} to {end_band-1}") # Optional debug
    else:
        target_slice = noisy_patch # Apply to all bands if range not specified
        noise_shape = patch.shape
        # print("Debug: Applying noise to all bands") # Optional debug


    # Generate noise with the shape of the target slice
    noise = np.random.normal(mean, std_dev, noise_shape).astype(patch.dtype)

    # Add noise to the target slice
    if start_band is not None and end_band is not None and target_slice is not noisy_patch:
         noisy_patch[..., start_band:end_band] += noise
    else:
         noisy_patch += noise # Add to the whole array if target_slice is the whole array

    # Clip values to maintain a reasonable range (e.g., non-negative for reflectance)
    # Apply clipping to the entire patch, assuming non-negativity is desired globally
    patch_min_original = np.min(patch) if patch.size > 0 else 0
    if patch_min_original >= 0:
        noisy_patch = np.clip(noisy_patch, 0, None) # Clip only bottom at 0
    # else:
        # If original data could be negative, avoid aggressive clipping unless bounds are known.
        # pass

    return noisy_patch
# --- End Modified Gaussian Noise Function ---

# --- START: New HSI Augmentation Functions ---
def apply_random_intensity_scaling(patch, min_factor=0.8, max_factor=1.2):
    """
    Applies random intensity scaling globally to an HSI patch.

    Args:
        patch (np.ndarray): The input HSI patch (H, W, C).
        min_factor (float): Minimum scaling factor.
        max_factor (float): Maximum scaling factor.

    Returns:
        np.ndarray: The patch with applied scaling.
    """
    if patch is None or patch.size == 0:
        return patch

    scale_factor = random.uniform(min_factor, max_factor)
    scaled_patch = patch * scale_factor

    # Clip values if original data was non-negative
    patch_min_original = np.min(patch) if patch.size > 0 else 0
    if patch_min_original >= 0:
        scaled_patch = np.clip(scaled_patch, 0, None)

    return scaled_patch

def apply_random_intensity_offset(patch, min_offset=-0.05, max_offset=0.05):
    """
    Applies a random intensity offset globally to an HSI patch.

    Args:
        patch (np.ndarray): The input HSI patch (H, W, C).
        min_offset (float): Minimum offset value.
        max_offset (float): Maximum offset value.

    Returns:
        np.ndarray: The patch with applied offset.
    """
    if patch is None or patch.size == 0:
        return patch

    offset = random.uniform(min_offset, max_offset)
    offset_patch = patch + offset

    # Clip values if original data was non-negative
    patch_min_original = np.min(patch) if patch.size > 0 else 0
    if patch_min_original >= 0:
        offset_patch = np.clip(offset_patch, 0, None)

    return offset_patch
# --- END: New HSI Augmentation Functions ---


# --- Augmentation Functions for Visualization (Keep as before) ---
def apply_random_color_jitter(image_pil, brightness=0.2, contrast=0.2, saturation=0.2):
    """
    Applies random color jitter (brightness, contrast, saturation) to a PIL Image.
    Used ONLY for the visualization, not saved in the .npy file.
    """
    if image_pil is None:
        return None

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
    max_hue_delta is fraction of 1.0 (PIL hue range is 0-1).
    """
    if image_pil is None:
        return None

    try:
        hue_factor = random.uniform(-max_hue_delta, max_hue_delta)
        # PIL's enhance expects factor 0..2 for hue? No, adjust_hue takes factor -0.5 to 0.5
        # Let's use functional API for clarity
        from torchvision.transforms import functional as F_vision
        image_pil = F_vision.adjust_hue(image_pil, hue_factor)

    except Exception as e:
        print(f"Warning: Could not apply hue shift: {e}")

    return image_pil
# --- End New Augmentation Functions ---


def main(args):
    print(f"Starting augmentation process...")
    print(f"Input directory: {args.input_dir}")
    print(f"Input label file: {args.input_label_filename}")
    print(f"Output directory: {args.output_dir}")
    print(f"Noise standard deviation: {args.noise_std_dev}")
    print(f"Applying noise to bands: 80 to 130 (inclusive indices)") # Clarify band range
    # --- Print info about new augmentations ---
    if args.apply_scaling:
        print(f"Applying random intensity scaling (Factor: {args.scale_factor_range})")
    if args.apply_offset:
        print(f"Applying random intensity offset (Range: {args.offset_range})")
    # --- End print info ---

    # --- Print visualization info ---
    if args.visualize_count > 0:
        print(f"Saving {args.visualize_count} visualization comparisons to: {args.visualize_dir}")
        os.makedirs(args.visualize_dir, exist_ok=True)
        # Clear existing visualization files
        print(f"Clearing existing files in {args.visualize_dir}...")
        for filename in os.listdir(args.visualize_dir):
             file_path = os.path.join(args.visualize_dir, filename)
             try:
                 if os.path.isfile(file_path) or os.path.islink(file_path):
                     os.unlink(file_path)
             except Exception as e:
                 print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print("Visualization disabled.")
    # --- End visualization info ---

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_label_file = os.path.join(args.input_label_filename) # Assume relative to CWD or use absolute path
    output_label_file = os.path.join(args.output_dir, 'labels_augmented.txt') # Different name for augmented labels
    output_patches_base_dir = os.path.join(args.output_dir, 'patches_augmented') # Subdir for augmented patches

    os.makedirs(output_patches_base_dir, exist_ok=True) # Ensure output base exists

    augmented_samples = []
    processed_count = 0
    viz_saved_count = 0 # <-- Keep track of saved visualizations

    # --- Load and Group All Samples ---
    all_samples_by_label = defaultdict(list)
    total_samples_read = 0
    try:
        with open(input_label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # --- MODIFICATION HERE ---
                # parts = line.split() # Old way, breaks with spaces in path
                parts = line.rsplit(maxsplit=1) # New way: split only at the last space
                # --- END MODIFICATION ---
                if len(parts) == 2:
                    rel_path, label_str = parts
                    try:
                        label = int(label_str)
                        # Store the relative path and label
                        all_samples_by_label[label].append(rel_path)
                        total_samples_read += 1
                    except ValueError:
                        print(f"Warning: Skipping line with non-integer label: {line}")
                else:
                    # This warning will now correctly catch lines that genuinely don't have path + label
                    print(f"Warning: Skipping malformed line (expected 'path label'): {line}")
    except FileNotFoundError:
        print(f"Error: Input label file not found at {input_label_file}")
        sys.exit(1)
    except Exception as read_e:
        print(f"Error reading input label file: {read_e}")
        sys.exit(1)

    if total_samples_read == 0:
        print("Error: No valid samples read from the input label file.")
        sys.exit(1)
    print(f"Read {total_samples_read} samples from label file.")
    # --- End Load and Group ---

    # --- Stratified Sampling (50%) ---
    selected_samples_for_processing = []
    print("Selecting approximately 50% of samples stratified by class...")
    for label, paths_in_class in all_samples_by_label.items():
        num_to_select = math.ceil(len(paths_in_class) * 0.5) # Select 50%, rounding up
        selected_paths = random.sample(paths_in_class, num_to_select)
        for path in selected_paths:
            selected_samples_for_processing.append((path, label)) # Keep label info

    print(f"Total samples selected for processing: {len(selected_samples_for_processing)}")

    # Shuffle the selected list to ensure random order during processing/visualization
    random.shuffle(selected_samples_for_processing)
    # --- End Stratified Sampling ---

    # --- Process Only Selected Samples ---
    if args.max_patches is not None:
        print(f"Warning: --max_patches argument is ignored. Processing {len(selected_samples_for_processing)} selected samples (~50% stratified).")

    for relative_path, label in tqdm(selected_samples_for_processing, desc="Augmenting Patches"):
        original_patch_full_path = os.path.join(args.input_dir, relative_path) # Path relative to input_dir

        try:
            # Load original patch
            original_patch_np = np.load(original_patch_full_path)

            # --- Apply HSI Augmentations ---
            # 1. Targeted Gaussian Noise
            augmented_patch_np = add_gaussian_noise(
                original_patch_np,
                mean=0.0,
                std_dev=args.noise_std_dev,
                start_band=80, # Python index for 81st band
                end_band=131  # Python index for 131st band (exclusive)
            )

            # 2. Random Intensity Scaling (Optional)
            if args.apply_scaling:
                augmented_patch_np = apply_random_intensity_scaling(
                    augmented_patch_np,
                    min_factor=args.scale_factor_range[0],
                    max_factor=args.scale_factor_range[1]
                )

            # 3. Random Intensity Offset (Optional)
            if args.apply_offset:
                augmented_patch_np = apply_random_intensity_offset(
                    augmented_patch_np,
                    min_offset=args.offset_range[0],
                    max_offset=args.offset_range[1]
                )
            # --- End HSI Augmentations ---

            # Define output path for the augmented .npy file
            # Maintain the same relative structure within the output directory
            output_relative_path = relative_path # Use the same relative path
            output_patch_full_path = os.path.join(output_patches_base_dir, output_relative_path)

            # Ensure the subdirectory structure exists in the output
            os.makedirs(os.path.dirname(output_patch_full_path), exist_ok=True)

            # Save the augmented patch
            np.save(output_patch_full_path, augmented_patch_np)

            # Add entry for the new label file (path relative to output_dir/patches_augmented)
            augmented_samples.append((output_relative_path, label)) # Store relative path and label

            # --- Generate and Save Visualization (if enabled and count not reached) ---
            if args.visualize_count > 0 and viz_saved_count < args.visualize_count:
                try:
                    # Visualize original
                    original_viz_rgb = visualize_hsi_patch(original_patch_np)

                    # Visualize augmented (base) - AFTER HSI augmentations
                    augmented_viz_rgb_base = visualize_hsi_patch(augmented_patch_np)

                    if original_viz_rgb is not None and augmented_viz_rgb_base is not None:
                        # Apply visual jitter/hue only to the augmented visualization
                        aug_viz_pil = Image.fromarray(augmented_viz_rgb_base)
                        aug_viz_pil = apply_random_color_jitter(aug_viz_pil)
                        aug_viz_pil = apply_random_hue_shift(aug_viz_pil)
                        augmented_viz_final_rgb = np.array(aug_viz_pil)

                        # Create comparison plot
                        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                        # --- Update plot title ---
                        aug_desc = f"Noise[{81}-{131}]"
                        if args.apply_scaling: aug_desc += "+Scale"
                        if args.apply_offset: aug_desc += "+Offset"
                        aug_desc += " (+Vis Jitter/Hue)"
                        fig.suptitle(f"Augmentation: {os.path.basename(relative_path)} (Label: {label})", fontsize=10)
                        # --- End update plot title ---

                        axes[0].imshow(original_viz_rgb)
                        axes[0].set_title("Original", fontsize=8)
                        axes[0].axis('off')

                        axes[1].imshow(augmented_viz_final_rgb)
                        # --- Update subplot title ---
                        axes[1].set_title(f"Augmented ({aug_desc})", fontsize=8)
                        # --- End update subplot title ---
                        axes[1].axis('off')

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

                        # Sanitize filename for saving
                        safe_filename_base = os.path.splitext(relative_path.replace(os.sep, '_'))[0]
                        viz_save_path = os.path.join(args.visualize_dir, f"viz_{viz_saved_count}_{safe_filename_base}.png")
                        plt.savefig(viz_save_path)
                        plt.close(fig)
                        viz_saved_count += 1
                    else:
                        print(f"Warning: Skipping visualization for {relative_path} due to error in RGB conversion.")

                except Exception as viz_e:
                    print(f"Warning: Failed to create/save visualization for {relative_path}: {viz_e}")
            # --- End Visualization ---

            processed_count += 1

        except FileNotFoundError:
            print(f"Warning: Original patch file not found: {original_patch_full_path}. Skipping.")
        except Exception as e:
            print(f"Error processing patch {relative_path}: {e}")
            # Decide whether to continue or stop on error
            # continue
    # --- End Processing Loop ---

    print(f"\nFinished augmenting {processed_count} patches.")
    if args.visualize_count > 0:
        print(f"Saved {viz_saved_count} visualization comparisons.")

    if not augmented_samples:
        print("Error: No patches were successfully augmented.")
        sys.exit(1)

    # Save the new label file
    print(f"Saving new label file to {output_label_file}")
    try:
        # Sort for consistency
        augmented_samples.sort()
        with open(output_label_file, 'w') as f:
            for rel_path, label in augmented_samples:
                # Ensure forward slashes for consistency in the label file
                f.write(f"{Path(rel_path).as_posix()} {label}\n")
    except Exception as e:
        print(f"Error writing augmented label file: {e}")
        sys.exit(1)

    print("Augmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment preprocessed hyperspectral patches with noise, processing ~50% stratified by class, and visualize.")
    parser.add_argument('--input_dir', type=str, required=True, help="Base directory containing the original preprocessed patches (e.g., './data_processed_patch/patches')")
    parser.add_argument('--input_label_filename', type=str, required=True, help="Name of the label file corresponding to the input patches (e.g., 'labels_patches.txt', expected inside input_dir or provide full path)")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory to save the augmented patches and the new label file.")
    parser.add_argument('--noise_std_dev', type=float, default=0.05, help="Standard deviation for the Gaussian noise applied to bands 81-131.")
    # --- Add arguments for new augmentations ---
    parser.add_argument('--apply_scaling', action='store_true', help="Apply random intensity scaling to HSI data.")
    parser.add_argument('--scale_factor_range', type=float, nargs=2, default=[0.8, 1.2], help="Range [min, max] for random intensity scaling factor.")
    parser.add_argument('--apply_offset', action='store_true', help="Apply random intensity offset to HSI data.")
    parser.add_argument('--offset_range', type=float, nargs=2, default=[-0.05, 0.05], help="Range [min, max] for random intensity offset.")
    # --- End add arguments ---
    parser.add_argument('--seed', type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument('--visualize_count', type=int, default=10, help="Number of original vs. augmented comparison images to save. 0 to disable.")
    parser.add_argument('--visualize_dir', type=str, default=None, help="Directory to save visualization images. Defaults to '[output_dir]/visualizations'.")
    parser.add_argument('--max_patches', type=int, default=None, help="(IGNORED) Maximum number of patches to process. Script now processes ~50% stratified.")

    args = parser.parse_args()

    # Default visualization dir if not provided
    if args.visualize_dir is None:
        args.visualize_dir = os.path.join(args.output_dir, 'visualizations')

    # Adjust input label file path if it's just a name
    if not os.path.isabs(args.input_label_filename) and not os.path.dirname(args.input_label_filename):
         # If it's just a filename, assume it's relative to the input_dir's parent
         # Or adjust logic if it's expected *inside* input_dir
         input_dir_parent = os.path.dirname(args.input_dir) if args.input_dir != '.' else '.'
         potential_path = os.path.join(input_dir_parent, args.input_label_filename)
         if os.path.exists(potential_path):
              args.input_label_filename = potential_path
              print(f"Interpreted input label file path as: {args.input_label_filename}")
         # Add more robust path checking if needed

    main(args)


# --- Example Command ---
# python tools/augment_patches.py \
#   --input_dir ./data_processed_patch/patches \
#   --input_label_filename ./data_processed_patch/labels_patches.txt \
#   --output_dir ./data_augmented_50pct_noise_scale_offset \
#   --noise_std_dev 0.05 \
#   --apply_scaling \
#   --scale_factor_range 0.9 1.1 \
#   --apply_offset \
#   --offset_range -0.02 0.02 \
#   --seed 123 \
#   --visualize_count 10
# --- End Example Command ---