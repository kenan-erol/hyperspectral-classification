# ... (other imports) ...
import matplotlib.pyplot as plt # <-- Make sure this is imported
import random # <-- Make sure this is imported
from PIL import Image, ImageEnhance # <-- Make sure these are imported
import sys # <-- Make sure this is imported
import os # <-- Make sure this is imported
import numpy as np # <-- Make sure this is imported
import torch # <-- Make sure this is imported
import torch.nn.functional as F # <-- Make sure this is imported
from tqdm import tqdm # <-- Make sure this is imported
import argparse # <-- Make sure this is imported

# --- Add src directory to sys.path if needed ---
# (Assuming log_utils is in src relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes tools/ is one level down from root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
     sys.path.append(src_path)
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
        # Return a black image of the correct size
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Calculate the mean across the spectral dimension
    try:
        # Use nanmean to handle potential NaNs if they exist in data
        display_img = np.nanmean(hsi_patch_np, axis=2)
    except Exception as e:
        print(f"Warning: Error calculating mean for visualization: {e}")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8) # Return black on error

    # Check if result is all NaN (can happen if input was all NaN)
    if np.all(np.isnan(display_img)):
        print("Warning: Mean calculation resulted in all NaNs.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Normalize the mean image to 0-255 range
    # Use nanmin/nanmax to ignore NaNs during normalization
    min_val = np.nanmin(display_img)
    max_val = np.nanmax(display_img)

    if max_val > min_val:
        # Normalize, converting NaNs to 0 after normalization (or another value if preferred)
        display_img = (display_img - min_val) / (max_val - min_val)
        display_img = np.nan_to_num(display_img, nan=0.0) # Convert NaNs to 0
        display_img = (display_img * 255)
    elif max_val == min_val:
        # Handle constant value image (avoid division by zero)
        # Check if the constant value is NaN
        if np.isnan(min_val):
             display_img = np.zeros((img_h, img_w)) # All NaN -> black
        else:
             # Assign a mid-gray value, or scale if range is known
             # For simplicity, let's make it gray
             display_img = np.full((img_h, img_w), 128)
    else: # min_val > max_val should not happen with nanmin/nanmax unless all NaN
        display_img = np.zeros((img_h, img_w)) # Fallback to black

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
        print("Warning: Attempted to add noise to an empty or None patch.")
        return patch # Return original if invalid

    # Generate noise with the same shape as the patch
    noise = np.random.normal(mean, std_dev, patch.shape).astype(patch.dtype)

    # Add noise to the patch
    noisy_patch = patch + noise

    # Clip values to maintain a reasonable range.
    patch_min = np.min(patch) if patch.size > 0 else 0

    if patch_min >= 0:
        # If original data was non-negative, clip noisy result at 0
        noisy_patch = np.clip(noisy_patch, 0, None)
    # else:
        # If original data could be negative, avoid aggressive clipping unless bounds are known.
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
        hsv = np.array(image_pil.convert('HSV'))
        # Apply hue shift (scale 0-255 for uint8)
        hue_shift = random.uniform(-max_hue_delta, max_hue_delta) * 255
        # Add shift and wrap around using modulo arithmetic
        hsv[..., 0] = (hsv[..., 0].astype(np.float32) + hue_shift) % 255
        hsv = hsv.astype(np.uint8)
        # Convert back to PIL RGB
        image_pil = Image.fromarray(hsv, 'HSV').convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to apply hue shift: {e}")
        # Return original image on error

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
        print(f"Visualizing {args.visualize_count} samples.")
        if args.visualize_dir:
            os.makedirs(args.visualize_dir, exist_ok=True)
            print(f"Visualization directory: {args.visualize_dir}")
        else:
            print("Error: --visualize_dir is required when --visualize_count > 0.")
            exit(1)
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
        num_to_process = min(len(lines), args.max_patches)
        # Sample random indices if processing a subset
        indices_to_process = random.sample(range(len(lines)), num_to_process)
        print(f"Processing a random subset of {num_to_process} patches.")
    else:
        print(f"Processing all {num_to_process} patches.")


    for idx in tqdm(indices_to_process, desc="Augmenting Patches"):
        line = lines[idx].strip()
        if not line: continue

        try:
            relative_path, label_str = line.rsplit(' ', 1)
            label = int(label_str)
            # Construct full path relative to input_dir
            full_patch_path = os.path.join(args.input_dir, relative_path)

            if not os.path.exists(full_patch_path):
                print(f"Warning: Patch file not found: {full_patch_path}. Skipping.")
                continue

            # Load the original patch
            patch_np = np.load(full_patch_path)
            if patch_np.dtype == np.float64: patch_np = patch_np.astype(np.float32)

            # Apply Gaussian noise to the raw HSI data
            noisy_patch = add_gaussian_noise(patch_np, std_dev=args.noise_std_dev)

            # Define output path for the noisy patch
            output_patch_path = os.path.join(output_patches_base_dir, relative_path)
            os.makedirs(os.path.dirname(output_patch_path), exist_ok=True)

            # Save the noisy patch
            np.save(output_patch_path, noisy_patch)

            # Add entry for the new label file (using the same relative path)
            augmented_samples.append((relative_path, label))
            processed_count += 1

            # --- Visualization Logic ---
            if args.visualize_count > 0 and viz_saved_count < args.visualize_count:
                try:
                    # Convert original and noisy HSI patches to RGB for display
                    original_rgb = visualize_hsi_patch(patch_np)
                    noisy_rgb = visualize_hsi_patch(noisy_patch)

                    if original_rgb is not None and noisy_rgb is not None:
                        # --- Apply visualization-only augmentations (jitter, hue) ---
                        # Convert noisy_rgb to PIL image to apply these transforms
                        noisy_pil = Image.fromarray(noisy_rgb)
                        noisy_jittered_pil = apply_random_color_jitter(noisy_pil)
                        noisy_jittered_hue_pil = apply_random_hue_shift(noisy_jittered_pil)
                        # Convert back to numpy array for plotting
                        noisy_augmented_vis = np.array(noisy_jittered_hue_pil)
                        # --- End visualization-only augmentations ---

                        # Create side-by-side plot
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 row, 2 columns

                        # Plot original patch
                        axes[0].imshow(original_rgb)
                        axes[0].set_title("Original Patch")
                        axes[0].axis('off')

                        # Plot augmented patch (noise + vis-only jitter/hue)
                        axes[1].imshow(noisy_augmented_vis)
                        axes[1].set_title(f"Augmented (Noise={args.noise_std_dev:.2f} + Jitter/Hue)")
                        axes[1].axis('off')

                        plt.suptitle(f"Patch: {os.path.basename(relative_path)}", fontsize=10)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

                        # Save the plot
                        vis_filename = os.path.join(args.visualize_dir, f"{os.path.splitext(os.path.basename(relative_path))[0]}_augmentation_vis.png")
                        plt.savefig(vis_filename, dpi=150)
                        plt.close(fig) # Close the figure to free memory
                        viz_saved_count += 1
                    else:
                        print(f"Warning: Skipping visualization for {relative_path} due to conversion error.")

                except Exception as vis_e:
                    print(f"Error during visualization for {relative_path}: {vis_e}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                         plt.close(fig) # Ensure figure is closed on error
            # --- End Visualization Logic ---

        except Exception as e:
            import traceback
            print(f"\n--- ERROR ---")
            print(f"Error processing line '{line}' (Index {idx}):")
            print(f"Relative Path: {relative_path if 'relative_path' in locals() else 'N/A'}")
            print(f"Exception Type: {type(e)}")
            print(f"Exception Message: {e}")
            print("Traceback:")
            traceback.print_exc()
            print(f"-------------")
            continue # Skip to the next patch

    print(f"\nFinished augmenting {processed_count} patches.")
    if viz_saved_count > 0:
        print(f"Saved {viz_saved_count} visualization comparisons to {args.visualize_dir}")

    if not augmented_samples:
        print("Warning: No patches were successfully augmented.")
        # Decide whether to exit or continue to save an empty label file
        # exit(1) # Optional: exit if no patches were processed

    # Save the new label file
    print(f"Saving new label file to {output_label_file}")
    try:
        # Sort samples for consistency before saving
        augmented_samples.sort()
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
                        help='Base directory containing the patch subdirectories (e.g., ./data_processed_patch).') # Modified help text
    parser.add_argument('--input_label_filename', type=str, default='labels.txt',
                        help='Filename of the input label file relative to the project root or an absolute path (e.g., labels_patches.txt)') # Clarified path expectation
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
        print("Error: --visualize_dir must be specified when --visualize_count > 0")
        exit(1)
    # --- End validation ---

    # --- Resolve input label file path ---
    # Check if it's absolute, otherwise assume relative to project root
    if not os.path.isabs(args.input_label_filename):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes script is in tools/
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
#   --visualize_count 10 \
#   --visualize_dir ./data_augmented_noise_0.05/visualizations
# --- End Example Command ---