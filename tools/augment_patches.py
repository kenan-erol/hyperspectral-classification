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
def visualize_hsi_patch(hsi_patch_np, start_band=None, end_band=None):
    """
    Creates a displayable RGB image (uint8) from an HSI patch (float32/64)
    using the mean across a specified range of spectral channels and robust normalization.
    Handles potential edge cases like zero channels or constant values.

    Args:
        hsi_patch_np (np.ndarray): The hyperspectral patch (H, W, C).
        start_band (int, optional): The starting band index (inclusive). Defaults to 0.
        end_band (int, optional): The ending band index (exclusive). Defaults to include all bands.

    Returns:
        Tuple[np.ndarray, str]: An RGB image (H, W, 3) of type uint8, and the band range string used.
                                Returns (zeros, "Invalid") if input is invalid.
    """
    if hsi_patch_np is None or hsi_patch_np.ndim < 2:
        print("Warning: Invalid input patch for visualization.")
        return np.zeros((50, 50, 3), dtype=np.uint8), "Invalid" # Return a small black square

    # Ensure 3 dimensions (H, W, C) even if C=1
    if hsi_patch_np.ndim == 2:
        hsi_patch_np = np.expand_dims(hsi_patch_np, axis=-1)

    img_h, img_w, img_c = hsi_patch_np.shape

    # Handle case with zero channels
    if img_c == 0:
        print("Warning: Patch has zero channels.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8), "0 Channels"

    # --- START: Band Slicing ---
    actual_start_band = start_band if start_band is not None else 0
    actual_end_band = end_band if end_band is not None else img_c

    # Validate band range
    if actual_start_band < 0 or actual_end_band > img_c or actual_start_band >= actual_end_band:
        print(f"Warning: Invalid band range [{actual_start_band}:{actual_end_band}] for patch with {img_c} channels. Using full range.")
        patch_to_process = hsi_patch_np
        band_range_str = f"All ({img_c})"
    else:
        patch_to_process = hsi_patch_np[:, :, actual_start_band:actual_end_band]
        # Use 1-based indexing for display string if desired, or keep 0-based
        band_range_str = f"{actual_start_band+1}-{actual_end_band}" # e.g., 81-131
        # Or use 0-based: band_range_str = f"{actual_start_band}-{actual_end_band-1}" # e.g., 80-130
    # --- END: Band Slicing ---

    # Check if the slice is empty
    if patch_to_process.shape[2] == 0:
        print(f"Warning: Selected band range [{actual_start_band}:{actual_end_band}] resulted in zero channels.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8), "Empty Range"


    # Calculate the mean across the spectral dimension of the processed patch
    try:
        # Use nanmean to handle potential NaNs if necessary
        display_img = np.nanmean(patch_to_process, axis=2)
    except Exception as e:
        print(f"Error calculating mean for visualization: {e}")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8), "Mean Error"

    # Check if result is all NaN (can happen if input was all NaN)
    if np.all(np.isnan(display_img)):
        print("Warning: Mean image is all NaN.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8), "All NaN"

    # Normalize the mean image to 0-255 range
    min_val = np.nanmin(display_img)
    max_val = np.nanmax(display_img)

    if np.isnan(min_val) or np.isnan(max_val):
         print("Warning: NaN found during normalization min/max calculation.")
         display_img = np.zeros((img_h, img_w), dtype=np.float32)
    elif max_val > min_val:
        display_img = (display_img - min_val) / (max_val - min_val) * 255.0
    elif max_val == min_val:
        display_img = np.zeros((img_h, img_w), dtype=np.float32)
    else:
         print("Warning: max_val <= min_val during normalization. Setting to 0.")
         display_img = np.zeros((img_h, img_w), dtype=np.float32)

    display_img = np.nan_to_num(display_img, nan=0.0)
    display_img = np.clip(display_img, 0, 255)

    display_img_uint8 = display_img.astype(np.uint8)
    rgb_display = np.stack([display_img_uint8] * 3, axis=-1)

    return rgb_display, band_range_str


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
def apply_random_intensity_scaling(patch, min_factor=0.8, max_factor=1.2, start_band=None, end_band=None):
    def apply_random_intensity_scaling(patch, min_factor=0.8, max_factor=1.2, start_band=None, end_band=None):
        """
    Applies random intensity scaling to an HSI patch, optionally targeting specific bands.

    Args:
        patch (np.ndarray): The input HSI patch (H, W, C).
        min_factor (float): Minimum scaling factor.
        max_factor (float): Maximum scaling factor.
        start_band (int, optional): Start band index (inclusive).
        end_band (int, optional): End band index (exclusive).

    Returns:
        Tuple[np.ndarray, float]: The patch with applied scaling and the scale factor used.
    """
    if patch is None or patch.size == 0:
        return patch, 1.0 # Return original patch and factor 1.0

    scaled_patch = patch.copy() # Work on a copy
    scale_factor = random.uniform(min_factor, max_factor)

    # Determine the slice to apply scaling
    if start_band is not None and end_band is not None:
        num_channels = patch.shape[2]
        valid_start = max(0, start_band)
        valid_end = min(num_channels, end_band)
        if valid_start >= valid_end:
            print(f"Warning: Invalid band range [{start_band}:{end_band}] for scaling. Applying globally.")
            scaled_patch *= scale_factor # Apply globally if range is invalid
        else:
            target_slice = scaled_patch[..., valid_start:valid_end]
            target_slice *= scale_factor # Apply scaling only to the slice
    else:
        scaled_patch *= scale_factor # Apply globally if range not specified

    # Clip values if original data was non-negative (apply globally after scaling)
    patch_min_original = np.min(patch) if patch.size > 0 else 0
    if patch_min_original >= 0:
        scaled_patch = np.clip(scaled_patch, 0, None)

    return scaled_patch, scale_factor # <-- Return both patch and factor

def apply_random_intensity_offset(patch, min_offset=-0.05, max_offset=0.05, start_band=None, end_band=None):
    """
    Applies a random intensity offset to an HSI patch, optionally targeting specific bands.

    Args:
        patch (np.ndarray): The input HSI patch (H, W, C).
        min_offset (float): Minimum offset value.
        max_offset (float): Maximum offset value.
        start_band (int, optional): Start band index (inclusive).
        end_band (int, optional): End band index (exclusive).

    Returns:
        Tuple[np.ndarray, float]: The patch with applied offset and the offset value used.
    """
    if patch is None or patch.size == 0:
        return patch, 0.0 # Return original patch and offset 0.0

    offset_patch = patch.copy() # Work on a copy
    offset_value = random.uniform(min_offset, max_offset) # Renamed variable

    # Determine the slice to apply offset
    if start_band is not None and end_band is not None:
        num_channels = patch.shape[2]
        valid_start = max(0, start_band)
        valid_end = min(num_channels, end_band)
        if valid_start >= valid_end:
            print(f"Warning: Invalid band range [{start_band}:{end_band}] for offset. Applying globally.")
            offset_patch += offset_value # Apply globally if range is invalid
        else:
            target_slice = offset_patch[..., valid_start:valid_end]
            target_slice += offset_value # Apply offset only to the slice
    offset_patch += offset_value # Apply globally if range not specified

    # Clip values if original data was non-negative (apply globally after offset)
    patch_min_original = np.min(patch) if patch.size > 0 else 0
    if patch_min_original >= 0:
        offset_patch = np.clip(offset_patch, 0, None)

    return offset_patch, offset_value
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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    visualize_dir = Path(args.visualize_dir) if args.visualize_dir else None
    input_label_file = Path(args.input_label_filename)

    # --- Directory and File Checks ---
    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)
    if not input_label_file.is_file():
        # Try resolving relative to input_dir parent if it's just a filename
        if not input_label_file.is_absolute() and not input_label_file.parent.name:
             potential_path = input_dir.parent / input_label_file.name
             if potential_path.is_file():
                  input_label_file = potential_path
                  print(f"Interpreted input label file as: {input_label_file}")
             else:
                  print(f"Error: Input label file '{args.input_label_filename}' not found.")
                  sys.exit(1)
        else:
             print(f"Error: Input label file '{input_label_file}' not found.")
             sys.exit(1)

    output_patches_dir = output_dir / 'patches_augmented' # Consistent subdir name
    output_label_file = output_dir / 'labels_augmented.txt' # Consistent label file name

    output_patches_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize_dir:
        visualize_dir.mkdir(parents=True, exist_ok=True)
        print(f"Clearing existing files in {visualize_dir}...")
        for filename in os.listdir(visualize_dir):
             file_path = os.path.join(visualize_dir, filename)
             try:
                 if os.path.isfile(file_path) or os.path.islink(file_path):
                     os.unlink(file_path)
             except Exception as e:
                 print(f'Failed to delete {file_path}. Reason: {e}')
    # --- End Directory and File Checks ---

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
                parts = line.rsplit(maxsplit=1) # Split only at the last space
                if len(parts) == 2:
                    rel_path, label_str = parts
                    try:
                        label = int(label_str)
                        all_samples_by_label[label].append(rel_path)
                        total_samples_read += 1
                    except ValueError:
                        print(f"Warning: Skipping line with non-integer label: {line}")
                else:
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
    random.shuffle(selected_samples_for_processing)
    # --- End Stratified Sampling ---

    # --- Process Only Selected Samples ---
    # if args.max_patches is not None:
    #     print(f"Warning: --max_patches argument is ignored. Processing {len(selected_samples_for_processing)} selected samples (~50% stratified).")

    # Define the target band range (Python indices)
    TARGET_START_BAND = 80  # Corresponds to 81st band
    TARGET_END_BAND = 131 # Corresponds to 131st band (exclusive)

    augmented_samples_list = [] # Store (rel_path, label) for the new label file
    processed_count = 0
    viz_saved_count = 0

    # --- Process Only Selected Samples ---
    for relative_path, label in tqdm(selected_samples_for_processing, desc="Augmenting Patches"):
        original_patch_full_path = input_dir / relative_path # Use Path object

        try:
            # Load original patch
            original_patch_np = np.load(original_patch_full_path)
            if original_patch_np.dtype == np.float64:
                 original_patch_np = original_patch_np.astype(np.float32)

            # --- Apply HSI Augmentations ---
            augmented_patch_np = original_patch_np.copy() # Start with a copy
            aug_desc_parts = []
            scale_factor = 1.0
            offset_value = 0.0

            # 1. Noise
            if args.noise_std_dev > 0:
                augmented_patch_np = add_gaussian_noise(
                    augmented_patch_np,
                    std_dev=args.noise_std_dev,
                    start_band=TARGET_START_BAND,
                    end_band=TARGET_END_BAND
                )
                noise_range = f"[{TARGET_START_BAND+1}:{TARGET_END_BAND}]" if TARGET_START_BAND is not None else "All"
                aug_desc_parts.append(f"N(σ={args.noise_std_dev}, bands={noise_range})")

            # 2. Scaling
            if args.apply_scaling:
                augmented_patch_np, scale_factor = apply_random_intensity_scaling(
                    augmented_patch_np,
                    min_factor=args.scale_factor_range[0],
                    max_factor=args.scale_factor_range[1],
                    start_band=TARGET_START_BAND, # Use specific scale/offset range
                    end_band=TARGET_END_BAND
                )
                if abs(scale_factor - 1.0) > 1e-6:
                    scale_range = f"[{TARGET_START_BAND+1}:{TARGET_END_BAND}]" if TARGET_START_BAND is not None else "All"
                    aug_desc_parts.append(f"S(x{scale_factor:.2f}, bands={scale_range})")

            # 3. Offset
            if args.apply_offset:
                augmented_patch_np, offset_value = apply_random_intensity_offset(
                    augmented_patch_np,
                    min_offset=args.offset_range[0],
                    max_offset=args.offset_range[1],
                    start_band=TARGET_START_BAND, # Use specific scale/offset range
                    end_band=TARGET_END_BAND
                )
                if abs(offset_value) > 1e-6:
                     offset_range = f"[{TARGET_START_BAND+1}:{TARGET_END_BAND}]" if TARGET_START_BAND is not None else "All"
                     aug_desc_parts.append(f"O({offset_value:+.3f}, bands={offset_range})")

            aug_desc = " + ".join(aug_desc_parts) if aug_desc_parts else "None"
            # --- End HSI Augmentations ---

            # Define output path for the augmented .npy file
            output_patch_full_path = output_patches_dir / relative_path # Use Path object
            output_patch_full_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the augmented patch
            np.save(output_patch_full_path, augmented_patch_np)

            # Add entry for the new label file (use relative path from output_dir)
            output_relative_path = output_patch_full_path.relative_to(output_dir)
            augmented_samples_list.append((output_relative_path.as_posix(), label))

            # --- Generate and Save Visualization (if enabled and count not reached) ---
            if visualize_dir and viz_saved_count < args.visualize_count:
                try:
                    # --- Generate RGB for specific band range ---
                    original_viz_rgb, orig_band_str = visualize_hsi_patch(original_patch_np, start_band=TARGET_START_BAND, end_band=TARGET_END_BAND)
                    augmented_viz_rgb, aug_band_str = visualize_hsi_patch(augmented_patch_np, start_band=TARGET_START_BAND, end_band=TARGET_END_BAND)
                    # ---

                    # --- REMOVE visual jitter/hue application ---
                    # aug_viz_pil = Image.fromarray(augmented_viz_rgb_base)
                    # aug_viz_pil = apply_random_color_jitter(aug_viz_pil)
                    # aug_viz_pil = apply_random_hue_shift(aug_viz_pil)
                    # augmented_viz_final_rgb = np.array(aug_viz_pil)
                    # --- Use the direct RGB conversion ---
                    augmented_viz_final_rgb = augmented_viz_rgb
                    # ---

                    if original_viz_rgb is not None and augmented_viz_final_rgb is not None:
                        # Create comparison plot
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        fig.suptitle(f"Patch: {relative_path} (Label: {label})", fontsize=10)

                        # --- Update plot titles ---
                        axes[0].imshow(original_viz_rgb)
                        axes[0].set_title(f"Original (Mean Bands {orig_band_str})", fontsize=8)
                        axes[0].axis('off')

                        axes[1].imshow(augmented_viz_final_rgb)
                        # Combine band info and augmentation description
                        axes[1].set_title(f"Augmented (Mean Bands {aug_band_str})\n{aug_desc}", fontsize=8)
                        axes[1].axis('off')
                        # --- End update plot titles ---

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                        # Create a more descriptive filename
                        viz_filename_base = Path(relative_path).name.replace('.npy', '.png')
                        viz_filename = visualize_dir / f"viz_{viz_saved_count:03d}_{viz_filename_base}"
                        plt.savefig(viz_filename, dpi=150)
                        plt.close(fig)
                        viz_saved_count += 1
                    else:
                        print(f"Warning: Skipping visualization for {relative_path} due to error in RGB conversion.")

                except Exception as viz_e:
                    print(f"Warning: Failed to create/save visualization for {relative_path}: {viz_e}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig) # Close figure on error
            # --- End Visualization ---

            processed_count += 1

        except FileNotFoundError:
            print(f"Warning: Original patch file not found: {original_patch_full_path}. Skipping.")
        except Exception as e:
            print(f"Error processing patch {relative_path}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # Decide whether to continue or stop on error
            # continue
    # --- End Processing Loop ---

    print(f"\nFinished augmenting {processed_count} patches.")
    if visualize_dir:
        print(f"Saved {viz_saved_count} visualization comparisons to {visualize_dir}")

    if not augmented_samples_list:
        print("Error: No patches were successfully augmented.")
        sys.exit(1)

    # Save the new label file
    print(f"Saving new label file to {output_label_file}")
    try:
        augmented_samples_list.sort() # Sort for consistency
        with open(output_label_file, 'w') as f:
            for rel_path, label in augmented_samples_list:
                f.write(f"{rel_path} {label}\n")
    except Exception as e:
        print(f"Error writing augmented label file: {e}")
        sys.exit(1)

    print("Augmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment hyperspectral patches with noise, scaling, offset, processing ~50% stratified by class, and visualize.")
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the original .npy patches (e.g., './data_processed_patch/patches').")
    parser.add_argument('--input_label_filename', type=str, required=True, help="Path to the label file containing relative paths of original patches (e.g., './data_processed_patch/labels_patches.txt').")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory to save the augmented patches ('patches_augmented/' subdir) and the new label file ('labels_augmented.txt').")

    # Noise Augmentation
    parser.add_argument('--noise_std_dev', type=float, default=0.0, help="Standard deviation for Gaussian noise. Set to 0 to disable.")
    # parser.add_argument('--noise_start_band', type=int, default=None, help="Start band index (inclusive, 0-based) for noise application (optional).")
    # parser.add_argument('--noise_end_band', type=int, default=None, help="End band index (exclusive, 0-based) for noise application (optional).")

    # Scaling Augmentation
    parser.add_argument('--apply_scaling', action='store_true', help="Apply random intensity scaling.")
    parser.add_argument('--scale_factor_range', type=float, nargs=2, default=[0.8, 1.2], help="Range [min, max] for random scaling factor.")

    # Offset Augmentation
    parser.add_argument('--apply_offset', action='store_true', help="Apply random intensity offset.")
    parser.add_argument('--offset_range', type=float, nargs=2, default=[-0.05, 0.05], help="Range [min, max] for random offset value.")

    # # Shared Band Range for Scaling/Offset (Optional)
    # parser.add_argument('--scale_offset_start_band', type=int, default=None, help="Start band index (inclusive, 0-based) for scaling/offset (optional).")
    # parser.add_argument('--scale_offset_end_band', type=int, default=None, help="End band index (exclusive, 0-based) for scaling/offset (optional).")

    # General Settings
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    # Visualization Settings
    parser.add_argument('--visualize_count', type=int, default=10, help="Number of random patches to visualize (0 for none).")
    parser.add_argument('--visualize_dir', type=str, default=None, help="Directory to save visualization images. Defaults to '[output_dir]/visualizations'.")

    # Ignored Argument (Kept for compatibility if scripts call it)
    parser.add_argument('--max_patches', type=int, default=None, help="(IGNORED) Script processes ~50% stratified.")

    args = parser.parse_args()

    # Default visualization dir if not provided
    if args.visualize_dir is None and args.visualize_count > 0:
        args.visualize_dir = os.path.join(args.output_dir, 'visualizations')

    # Validate ranges
    if args.apply_scaling and args.scale_factor_range[0] >= args.scale_factor_range[1]:
        raise ValueError("scale_factor_range min must be less than max.")
    if args.apply_offset and args.offset_range[0] >= args.offset_range[1]:
        raise ValueError("offset_range min must be less than max.")

    main(args)

# --- Example Command ---
# python tools/augment_patches.py \
#   --input_dir ./data_processed_patch/patches \
#   --input_label_filename ./data_processed_patch/labels_patches.txt \
#   --output_dir ./data_augmented_noise0.05_scale0.9-1.1_offset-0.02-0.02_bands81-131 \
#   --noise_std_dev 0.05 \
#   --noise_start_band 80 \
#   --noise_end_band 131 \
#   --apply_scaling \
#   --scale_factor_range 0.9 1.1 \
#   --apply_offset \
#   --offset_range -0.02 0.02 \
#   --scale_offset_start_band 80 \
#   --scale_offset_end_band 131 \
#   --seed 123 \
#   --visualize_count 20 \
#   --visualize_dir ./visualizations_bands81-131
# --- End Example Command ---