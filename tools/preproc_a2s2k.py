import os
import argparse
import numpy as np
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm # For progress bar

# --- Need to import SAM2 components ---
# Add the project root to sys.path to allow importing sam2 and other modules
#import sys
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#f project_root not in sys.path:
#    sys.path.insert(0, project_root)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# --------------------------------------

def _adjust_bbox(bbox, img_shape):
    """Make bbox square, enlarge slightly, and clip to image bounds."""
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]

    # Center
    cx = x + w / 2
    cy = y + h / 2

    # New side (max dimension + 10% padding)
    side = max(w, h) * 1.1

    # New square coordinates
    new_x = int(cx - side / 2)
    new_y = int(cy - side / 2)
    new_w = int(side)
    new_h = int(side)

    # Clip to image boundaries
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(img_w - new_x, new_w)
    new_h = min(img_h - new_y, new_h)

    # Ensure width and height are positive
    if new_w <= 0 or new_h <= 0:
        return None # Invalid bbox after clipping

    return new_x, new_y, new_w, new_h

def resize_patch(patch_np, target_size, device):
    """Resize a numpy patch (H, W, C) to target_size using F.interpolate."""
    if patch_np.size == 0:
        return None
    # Convert to tensor (C, H, W) and add batch dim (N, C, H, W)
    patch_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    # Interpolate
    resized_tensor = F.interpolate(
        patch_tensor,
        size=target_size, # target_size should be (H, W) tuple
        mode='bilinear',
        align_corners=False
    )
    # Remove batch dim, move to CPU, convert back to numpy (H, W, C)
    resized_np = resized_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    return resized_np

def main(args):
    print("Starting preprocessing for A2S2K format...")
    device = torch.device(args.device)

    # --- Load SAM2 Model ---
    print("Loading SAM2 model...")
    # Determine config name (copied logic from train_classification_hyper.py)
    base_config_name = "configs"
    sam2_checkpoint = args.sam2_checkpoint_path
    if "sam2.1_hiera_tiny" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_t.yaml")
    elif "sam2.1_hiera_small" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_s.yaml")
    elif "sam2.1_hiera_base_plus" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_b+.yaml")
    elif "sam2.1_hiera_large" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_l.yaml")
    else: # Add fallbacks for older models or default
        print(f"Warning: Could not determine specific config name for {sam2_checkpoint}. Add more checks or using default.")
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_l.yaml") # Defaulting to large

    model_cfg_name = model_cfg_name.replace(".yaml", "")
    print(f"Using SAM2 config name: {model_cfg_name}")

    sam2 = build_sam2(model_cfg_name, sam2_checkpoint, device=device, apply_postprocessing=False)
    # Use generate_masks method which returns list of bboxes directly
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    print("SAM2 model loaded.")
    # -----------------------

    # --- Prepare Output Directories ---
    patches_output_dir = os.path.join(args.output_dir, "patches")
    os.makedirs(patches_output_dir, exist_ok=True)
    print(f"Output directory for patches: {patches_output_dir}")
    print(f"Output directory for labels/paths: {args.output_dir}")
    # --------------------------------

    all_patch_paths = []
    all_labels = []
    patch_counter = 0

    print("Reading label file and processing images...")
    try:
        with open(args.label_file, 'r') as f:
            lines = f.readlines()

        # Use tqdm for progress bar
        for line in tqdm(lines, desc="Processing Images"):
            line = line.strip()
            if not line: continue

            try:
                relative_path, label_str = line.rsplit(' ', 1)
                label = int(label_str)
                full_image_path = os.path.join(args.data_dir, relative_path)

                if not os.path.exists(full_image_path):
                    print(f"Warning: Image file not found {full_image_path}, skipping.")
                    continue

                # Load the full hyperspectral image
                image = np.load(full_image_path)
                if image.ndim == 2: # Add channel dim if grayscale
                    image = np.expand_dims(image, axis=-1)
                if image.dtype == np.float64: # Convert to float32 if needed
                    image = image.astype(np.float32)
                img_h, img_w, img_c = image.shape # Get original dimensions

                # --- Convert hyperspectral to RGB for SAM2 ---
                # Option 1: Select specific bands (e.g., R=50, G=30, B=10)
                # if img_c >= 50:
                #     rgb_image_for_sam = image[:, :, [50, 30, 10]]
                # else: # Fallback
                #     rgb_image_for_sam = np.mean(image, axis=2)
                #     rgb_image_for_sam = np.stack([rgb_image_for_sam]*3, axis=-1)

                # Option 2: Use mean across channels (simple grayscale representation)
                rgb_image_for_sam = np.mean(image, axis=2)
                # Normalize to 0-255 uint8 for SAM2
                min_val, max_val = np.min(rgb_image_for_sam), np.max(rgb_image_for_sam)
                if max_val > min_val:
                    rgb_image_for_sam = ((rgb_image_for_sam - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    rgb_image_for_sam = np.zeros((img_h, img_w), dtype=np.uint8) # Handle constant image
                rgb_image_for_sam = np.stack([rgb_image_for_sam]*3, axis=-1) # Convert grayscale to 3-channel RGB
                # --- End RGB Conversion ---

                # Generate masks using SAM2 on the RGB image
                # Add debug print for the input shape to generate
                # print(f"DEBUG: Input shape to mask_generator.generate: {rgb_image_for_sam.shape}, dtype: {rgb_image_for_sam.dtype}")
                masks_data = mask_generator.generate(rgb_image_for_sam) # Pass the 3-channel uint8 image
                # print(f"DEBUG: Number of masks found: {len(masks_data)}") # Debug print

                if not masks_data:
                    # print(f"Warning: No masks found for {full_image_path}. Skipping patch generation for this image.")
                    continue

                # Select masks (e.g., based on area, score, or just take some)
                masks_data.sort(key=lambda x: x['area'], reverse=True)
                selected_masks = masks_data[:args.num_patches_per_image]

                for i, mask_info in enumerate(selected_masks):
                    bbox = mask_info['bbox'] # Get bbox [x, y, w, h]
                    adjusted_bbox = _adjust_bbox(bbox, image.shape) # Adjust based on ORIGINAL image shape

                    if adjusted_bbox is None:
                        # print(f"Warning: Invalid adjusted bbox for a mask in {full_image_path}. Skipping this patch.")
                        continue

                    adj_x, adj_y, adj_w, adj_h = adjusted_bbox
                    # Crop from the ORIGINAL hyperspectral image
                    patch_np = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]
                    # print(f"DEBUG: Cropped patch shape: {patch_np.shape}") # Debug print

                    # Resize the ORIGINAL hyperspectral patch
                    resized_patch = resize_patch(patch_np, (args.patch_size, args.patch_size), device)
                    # print(f"DEBUG: Resized patch shape: {resized_patch.shape if resized_patch is not None else 'None'}") # Debug print


                    if resized_patch is None:
                        # print(f"Warning: Empty patch after extraction/resizing for {full_image_path}. Skipping.")
                        continue

                    # Define output path for the patch
                    base_filename = os.path.splitext(os.path.basename(relative_path))[0]
                    patch_filename = f"{base_filename}_patch_{i}.npy"
                    output_patch_path = os.path.join(patches_output_dir, patch_filename)

                    # Save the patch
                    np.save(output_patch_path, resized_patch.astype(np.float32)) # Ensure saving as float32

                    # Store the relative path (from output_dir) and label
                    relative_patch_path = os.path.join("patches", patch_filename)
                    all_patch_paths.append(relative_patch_path)
                    all_labels.append(label)
                    patch_counter += 1

            except Exception as e:
                # --- Improved Error Reporting ---
                import traceback
                print(f"\n--- ERROR ---")
                print(f"Error processing line '{line}' corresponding to image '{full_image_path}':")
                print(f"Exception Type: {type(e)}")
                print(f"Exception Message: {e}")
                print("Traceback:")
                traceback.print_exc()
                print(f"-------------")
                # --- End Improved Error Reporting ---
                # Decide if you want to stop or continue
                # raise e # Uncomment to stop on error
                continue # Comment out to stop on error

    except FileNotFoundError:
        print(f"Error: Label file not found at {args.label_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading label file {args.label_file}: {e}")
        exit(1)

    print(f"\nFinished processing. Generated {patch_counter} patches.")

    if not all_patch_paths:
        print("Error: No patches were generated. Check input data and SAM2 setup.")
        exit(1)

    # Convert lists to numpy arrays
    patch_paths_array = np.array(all_patch_paths, dtype=str)
    labels_array = np.array(all_labels, dtype=int)

    # Save the final arrays
    paths_save_path = os.path.join(args.output_dir, 'patch_paths.npy')
    labels_save_path = os.path.join(args.output_dir, 'labels.npy')

    print(f"Saving patch paths to {paths_save_path}")
    np.save(paths_save_path, patch_paths_array)
    print(f"Saving labels to {labels_save_path}")
    np.save(labels_save_path, labels_array)

    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess hyperspectral data using SAM2 for A2S2K format.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing original hyperspectral image .npy files')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the text file containing relative image paths and labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed patches and the final patch_paths.npy and labels.npy')
    parser.add_argument('--sam2_checkpoint_path', type=str, required=True, help='Path to the SAM2 checkpoint file (.pt)')
    parser.add_argument('--patch_size', type=int, default=224, help='Target size (height and width) for the output patches')
    parser.add_argument('--num_patches_per_image', type=int, default=5, help='Maximum number of patches to extract per original image (sorted by mask area)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for SAM2 and resizing (cuda or cpu)')

    cli_args = parser.parse_args()
    main(cli_args)
