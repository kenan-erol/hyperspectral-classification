import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import statistics # Added for median calculation
import copy       # Added for deepcopy

# --- Constants from datasets.py ---
MIN_PILL_AREA = 50
MAX_PILL_AREA = 170
MIN_IOU_SCORE = 0.5

TARGET_MASK_COUNT = 100
MIN_MASKS_FOR_ADJUSTMENT = 51 # More than 50
# --- End Constants ---

# Import SAM2 components
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
        return None  # Invalid bbox after clipping

    return new_x, new_y, new_w, new_h

def resize_patch(patch_np, target_size, device):
    """Resize a numpy patch (H, W, C) to target_size using F.interpolate."""
    if patch_np is None or patch_np.size == 0:
        return None

    # Convert to tensor (C, H, W) and add batch dim (N, C, H, W)
    patch_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float().unsqueeze(0).to(device)

    # Interpolate
    resized_tensor = F.interpolate(
        patch_tensor,
        size=target_size,  # target_size should be (H, W) tuple
        mode='bilinear',
        align_corners=False
    )

    # Remove batch dim, move to CPU, convert back to numpy (H, W, C)
    resized_np = resized_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    return resized_np

def save_visualization(patch, output_path, title="Hyperspectral Patch"):
    """Save a visualization of a hyperspectral patch as a PNG image."""
    if patch is None or patch.size == 0:
        print(f"Warning: Cannot save visualization for empty patch: {output_path}")
        return
    # Create a RGB visualization from the hyperspectral data
    if patch.shape[2] >= 3:
        # Simple RGB using first 3 channels or specific bands if known
        # bands = [patch.shape[2]//4, patch.shape[2]//2, 3*patch.shape[2]//4] # Example bands
        bands = [0, patch.shape[2]//2, patch.shape[2]-1] # Another example: first, middle, last
        rgb = patch[:, :, bands]
    elif patch.shape[2] == 2:
         # Create a 3-channel image by repeating one channel and adding the second
         rgb = np.stack([patch[:,:,0], patch[:,:,1], patch[:,:,0]], axis=-1)
    else: # Single channel
        rgb = np.repeat(patch[:, :, :1], 3, axis=2)

    # Normalize for visualization
    rgb = rgb.astype(np.float32) # Ensure float for calculations
    rgb_min, rgb_max = np.min(rgb), np.max(rgb)
    if rgb_max > rgb_min:
        rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb_norm = np.zeros_like(rgb)

    # Clip values to be safe, although normalization should handle it
    rgb_norm = np.clip(rgb_norm, 0, 1)

    # Create figure and save
    try:
        plt.figure(figsize=(6, 6)) # Smaller figure size
        plt.imshow(rgb_norm)
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    except Exception as e:
        print(f"Error saving visualization {output_path}: {e}")

# Removed filter_masks_by_size function, logic integrated into main loop

def main(args):
    print("Starting preprocessing for hyperspectral classification...")
    device = torch.device(args.device)

    # --- Load SAM2 Model ---
    print("Loading SAM2 model...")
    # Determine config name based on checkpoint filename
    base_config_name = "configs" # Assuming configs are relative to sam2 directory or cwd
    sam2_checkpoint = args.sam2_checkpoint_path
    model_cfg_rel_path = None

    ckpt_basename = os.path.basename(sam2_checkpoint)
    if "sam2.1_hiera_tiny" in ckpt_basename:
        model_cfg_rel_path = "sam2.1/sam2.1_hiera_t.yaml"
    elif "sam2.1_hiera_small" in ckpt_basename:
        model_cfg_rel_path = "sam2.1/sam2.1_hiera_s.yaml"
    elif "sam2.1_hiera_base_plus" in ckpt_basename:
        model_cfg_rel_path = "sam2.1/sam2.1_hiera_b+.yaml"
    elif "sam2.1_hiera_large" in ckpt_basename:
        model_cfg_rel_path = "sam2.1/sam2.1_hiera_l.yaml"
    # Add older configs if needed
    elif "sam2_hiera_tiny" in ckpt_basename:
         model_cfg_rel_path = "sam2/sam2_hiera_t.yaml"
    elif "sam2_hiera_small" in ckpt_basename:
         model_cfg_rel_path = "sam2/sam2_hiera_s.yaml"
    elif "sam2_hiera_base_plus" in ckpt_basename:
         model_cfg_rel_path = "sam2/sam2_hiera_b+.yaml"
    elif "sam2_hiera_large" in ckpt_basename:
         model_cfg_rel_path = "sam2/sam2_hiera_l.yaml"

    if model_cfg_rel_path:
        # Construct path relative to the script or a known base like 'sam2' directory
        # This assumes the script is run from the project root or sam2/ directory structure is accessible
        # Adjust base path if necessary
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir) # Assumes tools/ is one level down
        sam2_dir = os.path.join(project_root, 'sam2', 'sam2') # Path to sam2 library code
        model_cfg_path = os.path.join(sam2_dir, base_config_name, model_cfg_rel_path)
        model_cfg_name = os.path.join(base_config_name, model_cfg_rel_path).replace(".yaml", "") # Name for build_sam2
        print(f"Using SAM2 config path: {model_cfg_path}")
        print(f"Using SAM2 config name for build: {model_cfg_name}")
    else:
        print(f"Warning: Could not determine specific config name for {ckpt_basename}. Using default.")
        # Provide a sensible default relative path
        default_cfg_rel_path = "sam2.1/sam2.1_hiera_b+.yaml"
        model_cfg_path = os.path.join(sam2_dir, base_config_name, default_cfg_rel_path)
        model_cfg_name = os.path.join(base_config_name, default_cfg_rel_path).replace(".yaml", "")
        print(f"Using default SAM2 config path: {model_cfg_path}")
        print(f"Using default SAM2 config name for build: {model_cfg_name}")

    if not os.path.exists(model_cfg_path):
         print(f"ERROR: SAM2 config file not found at {model_cfg_path}. Check paths.")
         exit(1)

    try:
        sam2 = build_sam2(model_cfg_name, sam2_checkpoint, device=device, apply_postprocessing=False)
        # Use SAM2 Automatic Mask Generator parameters similar to example/defaults
        # These parameters control the initial generation, filtering happens later
        mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2,
                    points_per_side=32, # Default: 32
                    points_per_batch=64, # Default: 64
                    pred_iou_thresh=0.8, # Default: 0.8 - Keep good initial masks
                    stability_score_thresh=0.9, # Default: 0.95 - Slightly lower for more variety initially
                    stability_score_offset=1.0, # Default: 1.0
                    box_nms_thresh=0.7, # Default: 0.7
                    crop_n_layers=0, # Default: 0 - No cropping for simplicity unless needed
                    crop_nms_thresh = 0.7, # Default: 0.7
                    # crop_overlap_ratio = 512 / 1500, # Default
                    # crop_n_points_downscale_factor = 1, # Default
                    min_mask_region_area=25, # Default: 0 - Filter small disconnected regions
                    output_mode = "binary_mask", # Default
                    # use_m2m=False, # Default: False
                    multimask_output = True, # Default: True - Get multiple masks per point
                )
        print("SAM2 model and generator initialized.")
    except Exception as e:
        import traceback
        print(f"Error initializing SAM2: {e}")
        traceback.print_exc()
        exit(1)
    # -----------------------

    # --- Prepare Output Directories ---
    output_patches_dir = os.path.join(args.output_dir, "patches")
    output_vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(output_patches_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)
    print(f"Output directory for patches: {output_patches_dir}")
    print(f"Output directory for visualizations: {output_vis_dir}")
    # --------------------------------

    # Path to save patch paths and labels
    new_label_file_path = os.path.join(args.output_dir, 'labels.txt')
    all_patch_entries = [] # Store tuples of (relative_patch_path, label)
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

                # Create directory structure mirroring input
                rel_dir = os.path.dirname(relative_path)
                output_patch_subdir = os.path.join(output_patches_dir, rel_dir)
                output_vis_subdir = os.path.join(output_vis_dir, rel_dir)
                os.makedirs(output_patch_subdir, exist_ok=True)
                os.makedirs(output_vis_subdir, exist_ok=True)

                # Load the full hyperspectral image
                image = np.load(full_image_path)
                if image.ndim == 2:  # Add channel dim if grayscale
                    image = np.expand_dims(image, axis=-1)
                if image.dtype == np.float64:  # Convert to float32 if needed
                    image = image.astype(np.float32)
                img_h, img_w, img_c = image.shape  # Get original dimensions

                # --- Convert hyperspectral to RGB for SAM2 ---
                rgb_image_for_sam = np.mean(image, axis=2)
                min_val, max_val = np.min(rgb_image_for_sam), np.max(rgb_image_for_sam)
                if max_val > min_val:
                    rgb_image_for_sam = ((rgb_image_for_sam - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    rgb_image_for_sam = np.zeros((img_h, img_w), dtype=np.uint8)
                rgb_image_for_sam = np.stack([rgb_image_for_sam]*3, axis=-1)
                # --- End RGB Conversion ---

                # Generate masks using SAM2 on the RGB image
                # print(f"Generating masks for {full_image_path}...") # Too verbose
                masks_data = mask_generator.generate(rgb_image_for_sam)

                if not masks_data:
                    print(f"Warning: No masks generated for {full_image_path}. Skipping.")
                    continue

                # --- Filter masks based on datasets.py logic ---
                valid_masks = [
                    m for m in masks_data
                    if MIN_PILL_AREA <= m['area'] <= MAX_PILL_AREA and m['predicted_iou'] >= MIN_IOU_SCORE
                ]
                # --- End Filter ---

                if not valid_masks:
                    print(f"Warning: No masks passed filtering for {full_image_path}. Skipping.")
                    continue

                # --- Adjust mask pool and determine target patch count ---
                selection_pool = []
                num_valid = len(valid_masks)
                target_patches_for_image = 0

                if num_valid >= MIN_MASKS_FOR_ADJUSTMENT:
                    # print(f"DEBUG: Found {num_valid} masks, adjusting pool to {TARGET_MASK_COUNT}.") # Optional Debug
                    target_patches_for_image = args.num_patches_per_image # Should be 100 based on bash script

                    # Calculate median area
                    areas = [m['area'] for m in valid_masks]
                    median_area = statistics.median(areas) if areas else 0

                    # Calculate difference from median and sort
                    masks_with_diff = [(abs(m['area'] - median_area), m) for m in valid_masks]
                    masks_with_diff.sort(key=lambda x: x[0]) # Sort by difference (closest first)
                    sorted_masks_by_closeness = [m[1] for m in masks_with_diff]

                    if num_valid > TARGET_MASK_COUNT:
                        # Too many: Keep the 100 closest to the median area
                        selection_pool = sorted_masks_by_closeness[:TARGET_MASK_COUNT]
                        # print(f"DEBUG: Reduced pool to {len(selection_pool)} by keeping closest to median {median_area}.") # Optional Debug
                    else: # MIN_MASKS_FOR_ADJUSTMENT <= num_valid <= TARGET_MASK_COUNT
                        # Need more: Duplicate masks closest to median until we have 100
                        num_to_add = TARGET_MASK_COUNT - num_valid
                        masks_to_duplicate = sorted_masks_by_closeness[:num_to_add]
                        duplicates = [copy.deepcopy(m) for m in masks_to_duplicate]
                        selection_pool = valid_masks + duplicates
                        # print(f"DEBUG: Increased pool to {len(selection_pool)} by duplicating {num_to_add} closest to median {median_area}.") # Optional Debug

                else: # Case 2: < 51 valid masks found -> Target 6 patches closest to median
                    target_patches_for_image = min(6, num_valid)
                    # Select the N masks closest to the median area
                    selection_pool = sorted_masks_by_closeness[:target_patches_for_image]
                    # print(f"DEBUG: Found only {num_valid} masks, selected {len(selection_pool)} closest to median {median_area}.")


                # --- Select final masks for saving ---
                if not selection_pool:
                     print(f"Warning: Selection pool empty for {full_image_path} after adjustment. Skipping.")
                     continue

                # Sort the final pool by area (largest first) to pick the top N
                selection_pool.sort(key=lambda x: x['area'], reverse=True)
                selected_masks = selection_pool[:target_patches_for_image]
                # --- End Selection ---


                base_filename = os.path.splitext(os.path.basename(relative_path))[0]
                patches_extracted_for_image = 0

                for i, mask_info in enumerate(selected_masks):
                    bbox = mask_info['bbox']  # Get bbox [x, y, w, h]

                    # Adjust bounding box (square, padded, clipped)
                    adjusted_bbox = _adjust_bbox(bbox, image.shape)
                    if adjusted_bbox is None:
                        print(f"Warning: Skipping mask {i} due to invalid adjusted bbox for {full_image_path}")
                        continue

                    # Extract patch from ORIGINAL HSI image
                    adj_x, adj_y, adj_w, adj_h = adjusted_bbox
                    patch_np = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]

                    if patch_np.size == 0:
                         print(f"Warning: Skipping mask {i} due to empty crop for {full_image_path}")
                         continue

                    # Resize patch to target size
                    resized_patch = resize_patch(patch_np, (args.patch_size, args.patch_size), device)
                    if resized_patch is None:
                        print(f"Warning: Skipping mask {i} due to resize failure for {full_image_path}")
                        continue

                    # Define output paths for the patch
                    patch_filename_npy = f"{base_filename}_patch_{patches_extracted_for_image}.npy"
                    patch_filename_png = f"{base_filename}_patch_{patches_extracted_for_image}.png"

                    output_patch_path_npy = os.path.join(output_patch_subdir, patch_filename_npy)
                    output_vis_path_png = os.path.join(output_vis_subdir, patch_filename_png)

                    # Save the patch as NPY
                    try:
                        np.save(output_patch_path_npy, resized_patch.astype(np.float32))
                    except Exception as e:
                        print(f"Error saving NPY {output_patch_path_npy}: {e}")
                        continue # Skip this patch if save fails

                    # Save visualization as PNG
                    save_visualization(
                        resized_patch,
                        output_vis_path_png,
                        title=f"{base_filename} - Patch {patches_extracted_for_image}\nArea: {mask_info['area']}, IoU: {mask_info['predicted_iou']:.2f}"
                    )

                    # Store the relative path and label for the new label file
                    # Path relative to the *output data directory*
                    rel_patch_path_npy = os.path.join("patches", rel_dir, patch_filename_npy)
                    all_patch_entries.append((rel_patch_path_npy, label))

                    patches_extracted_for_image += 1
                    patch_counter += 1

                # print(f"Extracted {patches_extracted_for_image} patches from {full_image_path}") # Too verbose

            except Exception as e:
                import traceback
                print(f"\n--- ERROR ---")
                print(f"Error processing line '{line}' corresponding to image '{full_image_path}':")
                print(f"Exception Type: {type(e)}")
                print(f"Exception Message: {e}")
                print("Traceback:")
                traceback.print_exc()
                print(f"-------------")
                continue # Continue to the next image

    except FileNotFoundError:
        print(f"Error: Label file not found at {args.label_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading label file {args.label_file}: {e}")
        exit(1)

    print(f"\nFinished processing. Generated {patch_counter} patches.")

    if not all_patch_entries:
        print("Error: No patches were generated. Check input data, SAM2 setup, and filtering parameters.")
        exit(1)

    # Save the new label file (list of relative patch paths and labels)
    print(f"Saving new label file to {new_label_file_path}")
    try:
        with open(new_label_file_path, 'w') as f:
            for rel_path, lbl in all_patch_entries:
                f.write(f"{rel_path} {lbl}\n")
    except Exception as e:
        print(f"Error writing new label file {new_label_file_path}: {e}")
        exit(1)

    # Optional: Save as .npy as well if needed by dataset loader later
    # patch_paths_array = np.array([entry[0] for entry in all_patch_entries], dtype=str)
    # labels_array = np.array([entry[1] for entry in all_patch_entries], dtype=int)
    # paths_save_path = os.path.join(args.output_dir, 'patch_paths.npy')
    # labels_save_path = os.path.join(args.output_dir, 'labels.npy')
    # print(f"Saving patch paths to {paths_save_path}")
    # np.save(paths_save_path, patch_paths_array)
    # print(f"Saving labels to {labels_save_path}")
    # np.save(labels_save_path, labels_array)

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess hyperspectral data using SAM2 for mask generation and patch extraction")
    # Essential arguments for this script
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing original hyperspectral image .npy files')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the text file containing relative original image paths and labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed patches, visualizations, and the new labels.txt')
    parser.add_argument('--sam2_checkpoint_path', type=str, required=True, help='Path to the SAM2 checkpoint file (.pt)')
    parser.add_argument('--patch_size', type=int, default=224, help='Target size (height and width) for the output patches')
    parser.add_argument('--num_patches_per_image', type=int, default=100, help='Target number of patches to generate per image if >= MIN_MASKS_FOR_ADJUSTMENT masks are found')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'], help='Device to use for SAM2 and resizing (cuda, cpu, or mps)')

    # Arguments kept for compatibility with bash script, but not directly used in this preprocessing logic
    parser.add_argument('--n_batch', type=int, default=25, help='(Not used in preprocessing)')
    parser.add_argument('--train_split_ratio', type=float, default=0.8, help='(Not used in preprocessing)')
    parser.add_argument('--encoder_type', type=str, default='resnet18', help='(Not used in preprocessing)')
    parser.add_argument('--num_channels', type=int, default=256, help='(Not used in preprocessing)')
    parser.add_argument('--n_epoch', type=int, default=50, help='(Not used in preprocessing)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='(Not used in preprocessing)')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='(Not used in preprocessing)')
    parser.add_argument('--learning_rate_period', type=int, default=10, help='(Not used in preprocessing)')
    parser.add_argument('--checkpoint_path', type=str, default='hyper_checkpoints/resnet/', help='(Not used in preprocessing)')


    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Add MIN/MAX constants to args namespace for easier access if needed, or just use globals
    args.MIN_PILL_AREA = MIN_PILL_AREA
    args.MAX_PILL_AREA = MAX_PILL_AREA
    args.MIN_IOU_SCORE = MIN_IOU_SCORE
    args.TARGET_MASK_COUNT = TARGET_MASK_COUNT
    args.MIN_MASKS_FOR_ADJUSTMENT = MIN_MASKS_FOR_ADJUSTMENT

    main(args)