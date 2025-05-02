# filepath: tools/compare_real_fake_patches.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

# --- Add src directory to sys.path if needed ---
# (Assuming this script is in tools/ and utils are in src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
# --- End Path Setup ---

# --- HSI to RGB Visualization Function (Copy from augment_patches.py or import) ---
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
        print("Warning: Invalid input patch for visualization.")
        return np.zeros((50, 50, 3), dtype=np.uint8) # Return a small black square

    # Ensure 3 dimensions (H, W, C) even if C=1
    if hsi_patch_np.ndim == 2:
        hsi_patch_np = np.expand_dims(hsi_patch_np, axis=-1)

    img_h, img_w, img_c = hsi_patch_np.shape

    # Handle case with zero channels
    if img_c == 0:
        print("Warning: Patch has zero channels.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Calculate the mean across the spectral dimension
    try:
        # Use nanmean to handle potential NaNs if necessary
        display_img = np.nanmean(hsi_patch_np, axis=2)
    except Exception as e:
        print(f"Error calculating mean for visualization: {e}")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Check if result is all NaN (can happen if input was all NaN)
    if np.all(np.isnan(display_img)):
        print("Warning: Mean image is all NaN.")
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Normalize the mean image to 0-255 range
    # Use nanmin/nanmax to ignore NaNs during normalization
    min_val = np.nanmin(display_img)
    max_val = np.nanmax(display_img)

    # Check if min/max are valid numbers
    if np.isnan(min_val) or np.isnan(max_val):
         print("Warning: NaN found during normalization min/max calculation.")
         # Default to black image if normalization fails
         display_img = np.zeros((img_h, img_w), dtype=np.float32)
    elif max_val > min_val:
        # Normalize to 0-1, then scale to 0-255
        display_img = (display_img - min_val) / (max_val - min_val) * 255.0
    elif max_val == min_val:
        # Handle constant value image - map to mid-gray or black/white
        # Mapping to 0 avoids issues if the constant value was negative after augmentation
        display_img = np.zeros((img_h, img_w), dtype=np.float32)
        # print(f"Warning: Constant value ({min_val}) in mean image, setting to 0.")
    else: # Should not happen if min/max are valid numbers
         print("Warning: max_val <= min_val during normalization. Setting to 0.")
         display_img = np.zeros((img_h, img_w), dtype=np.float32)

    # Clip to ensure values are within [0, 255] after potential float inaccuracies
    display_img = np.clip(display_img, 0, 255)

    # Convert to uint8 and stack to create 3-channel RGB
    display_img_uint8 = display_img.astype(np.uint8)
    rgb_display = np.stack([display_img_uint8] * 3, axis=-1)

    return rgb_display
# --- End HSI to RGB Visualization Function ---


def main(args):
    try:
        print(f"Loading real patch: {args.real_patch_path}")
        real_patch = np.load(args.real_patch_path)
        print(f"Loading fake patch: {args.fake_patch_path}")
        fake_patch = np.load(args.fake_patch_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during file loading: {e}")
        sys.exit(1)

    if real_patch.shape != fake_patch.shape:
        print(f"Error: Shape mismatch! Real: {real_patch.shape}, Fake: {fake_patch.shape}")
        # Optionally add resizing logic here if needed, but ideally they should match
        sys.exit(1)

    print(f"Patch shape: {real_patch.shape}")
    img_h, img_w, num_bands = real_patch.shape

    # Calculate difference
    diff_patch = fake_patch - real_patch

    # --- Visualization ---
    real_viz = visualize_hsi_patch(real_patch)
    fake_viz = visualize_hsi_patch(fake_patch)
    diff_viz = visualize_hsi_patch(diff_patch) # Visualize the difference

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(real_viz)
    axes[0].set_title(f"Real Patch\n{os.path.basename(args.real_patch_path)}")
    axes[0].axis('off')

    axes[1].imshow(fake_viz)
    axes[1].set_title(f"Fake (Augmented) Patch\n{os.path.basename(args.fake_patch_path)}")
    axes[1].axis('off')

    axes[2].imshow(diff_viz)
    axes[2].set_title("Difference (Fake - Real)")
    axes[2].axis('off')

    plt.tight_layout()
    # Save the comparison figure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.real_patch_path))[0]
    vis_filename = output_dir / f"{base_name}_comparison.png"
    plt.savefig(vis_filename)
    print(f"Saved patch comparison visualization to: {vis_filename}")
    plt.close(fig)


    # --- Spectral Plotting ---
    # Define pixels to plot (e.g., center, corners, or from args)
    pixels_to_plot = []
    if args.pixels:
        for p_str in args.pixels:
            try:
                y, x = map(int, p_str.split(','))
                if 0 <= y < img_h and 0 <= x < img_w:
                    pixels_to_plot.append((y, x))
                else:
                    print(f"Warning: Pixel ({y},{x}) is out of bounds ({img_h}x{img_w}). Skipping.")
            except ValueError:
                print(f"Warning: Invalid pixel format '{p_str}'. Use 'y,x'. Skipping.")
    else:
        # Default to center pixel if none provided
        center_y, center_x = img_h // 2, img_w // 2
        pixels_to_plot.append((center_y, center_x))
        # Add corners if image is large enough
        if img_h > 1 and img_w > 1:
             pixels_to_plot.append((0, 0))
             pixels_to_plot.append((img_h - 1, img_w - 1))


    if not pixels_to_plot:
        print("No valid pixels selected for spectral plotting.")
        return

    print(f"Plotting spectra for pixels: {pixels_to_plot}")
    band_indices = np.arange(num_bands)

    num_plots = len(pixels_to_plot)
    fig_spec, axes_spec = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), squeeze=False)

    for i, (y, x) in enumerate(pixels_to_plot):
        real_spectrum = real_patch[y, x, :]
        fake_spectrum = fake_patch[y, x, :]
        diff_spectrum = diff_patch[y, x, :]

        ax = axes_spec[i, 0]
        ax.plot(band_indices, real_spectrum, label='Real', alpha=0.7)
        ax.plot(band_indices, fake_spectrum, label='Fake', alpha=0.7)
        ax.plot(band_indices, diff_spectrum, label='Difference', linestyle='--', color='red', alpha=0.8)
        ax.set_title(f"Spectral Profile at Pixel ({y}, {x})")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Intensity / Reflectance")
        ax.legend()
        ax.grid(True, linestyle=':')

    plt.tight_layout()
    # Save the spectral plot figure
    spec_filename = output_dir / f"{base_name}_spectra.png"
    plt.savefig(spec_filename)
    print(f"Saved spectral plots to: {spec_filename}")
    plt.close(fig_spec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare a real HSI patch with its augmented (fake) counterpart.")
    parser.add_argument('--real_patch_path', type=str, required=True, help="Path to the real .npy patch file.")
    parser.add_argument('--fake_patch_path', type=str, required=True, help="Path to the corresponding fake .npy patch file.")
    parser.add_argument('--output_dir', type=str, default="./comparison_output", help="Directory to save comparison images and plots.")
    parser.add_argument('--pixels', type=str, nargs='+', help="List of pixel coordinates 'y,x' to plot spectra for (e.g., --pixels 112,112 0,0). Defaults to center.")

    args = parser.parse_args()
    main(args)
