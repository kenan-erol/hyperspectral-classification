import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2
import statistics

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

# ... (Keep _adjust_bbox, resize_patch, hsi_to_rgb_display functions as they are) ...
def _adjust_bbox(bbox, img_shape):
    """Make bbox square, enlarge slightly, and clip to image bounds."""
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * 1.1
    new_x = int(cx - side / 2)
    new_y = int(cy - side / 2)
    new_w = int(side)
    new_h = int(side)
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(img_w - new_x, new_w)
    new_h = min(img_h - new_y, new_h)
    if new_w <= 0 or new_h <= 0: return None
    return new_x, new_y, new_w, new_h

def resize_patch(patch_np, target_size, device):
    """Resize a numpy patch (H, W, C) to target_size using F.interpolate."""
    if patch_np is None or patch_np.size == 0: return None
    patch_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    resized_tensor = F.interpolate(
        patch_tensor, size=target_size, mode='bilinear', align_corners=False
    )
    resized_np = resized_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    return resized_np

def hsi_to_rgb_display(hsi_image):
    """Creates a displayable RGB image from HSI data using the mean across channels."""
    if hsi_image.ndim == 2: hsi_image = np.expand_dims(hsi_image, axis=-1)
    img_h, img_w, img_c = hsi_image.shape
    if img_c == 0: return np.zeros((img_h, img_w, 3), dtype=np.uint8)
    display_img = np.mean(hsi_image, axis=2)
    min_val, max_val = np.min(display_img), np.max(display_img)
    if max_val > min_val:
        display_img = ((display_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        display_img = np.zeros((img_h, img_w), dtype=np.uint8)
    return np.stack([display_img]*3, axis=-1)


def main(args):
    print("Starting preprocessing for hyperspectral classification...")
    device = torch.device(args.device)

    # --- Load SAM2 Model ---
    # ... (Keep SAM2 loading logic as before) ...
    print("Loading SAM2 model...")
    base_config_name = "configs"
    sam2_checkpoint = args.sam2_checkpoint_path
    model_cfg_rel_path = None
    ckpt_basename = os.path.basename(sam2_checkpoint)
    # Determine config based on checkpoint name
    if "sam2.1_hiera_tiny" in ckpt_basename: model_cfg_rel_path = "sam2.1/sam2.1_hiera_t.yaml"
    elif "sam2.1_hiera_small" in ckpt_basename: model_cfg_rel_path = "sam2.1/sam2.1_hiera_s.yaml"
    elif "sam2.1_hiera_base_plus" in ckpt_basename: model_cfg_rel_path = "sam2.1/sam2.1_hiera_b+.yaml"
    elif "sam2.1_hiera_large" in ckpt_basename: model_cfg_rel_path = "sam2.1/sam2.1_hiera_l.yaml"
    elif "sam2_hiera_tiny" in ckpt_basename: model_cfg_rel_path = "sam2/sam2_hiera_t.yaml"
    elif "sam2_hiera_small" in ckpt_basename: model_cfg_rel_path = "sam2/sam2_hiera_s.yaml"
    elif "sam2_hiera_base_plus" in ckpt_basename: model_cfg_rel_path = "sam2/sam2_hiera_b+.yaml"
    elif "sam2_hiera_large" in ckpt_basename: model_cfg_rel_path = "sam2/sam2_hiera_l.yaml"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    sam2_dir = os.path.join(project_root, 'sam2', 'sam2')

    if model_cfg_rel_path:
        model_cfg_path = os.path.join(sam2_dir, base_config_name, model_cfg_rel_path)
        model_cfg_name = os.path.join(base_config_name, model_cfg_rel_path).replace(".yaml", "")
        print(f"Using SAM2 config path: {model_cfg_path}")
        print(f"Using SAM2 config name for build: {model_cfg_name}")
    else:
        print(f"Warning: Could not determine specific config name for {ckpt_basename}. Using default.")
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
        mask_generator = SAM2AutomaticMaskGenerator(
				model=sam2,
				points_per_side=64,
				points_per_batch=64,
				pred_iou_thresh=0.9,
				stability_score_thresh=0.92, # >0.92 for less squarish masks
				stability_score_offset=0.5,
				box_nms_thresh=0.55,
				crop_n_layers=1,
				crop_nms_thresh = 0.7,
					crop_overlap_ratio = 512 / 1500,
					crop_n_points_downscale_factor = 2,
					# point_grids: Optional[List[np.ndarray]] = None,
					# min_mask_region_area = 15.0,
					output_mode = "binary_mask",
					multimask_output = True,
				min_mask_region_area=25.0,
				use_m2m=True,
			) # rn it is too edgy
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
    images_processed_count = 0 # <-- Initialize counter for processed images

    print("Reading label file and processing images...")
    try:
        with open(args.label_file, 'r') as f:
            lines = f.readlines()

        # Determine the number of lines (images) to process
        num_lines_to_process = len(lines)
        if args.max_images is not None and args.max_images > 0:
            num_lines_to_process = min(len(lines), args.max_images)
            print(f"Processing a maximum of {num_lines_to_process} images.")

        # Use tqdm for progress bar, iterating only up to num_lines_to_process
        for line_index in tqdm(range(num_lines_to_process), desc="Processing Images"):
            line = lines[line_index].strip() # Get the specific line
            if not line: continue

            # --- Check if max_images limit is reached (redundant with loop range, but safe) ---
            # if args.max_images is not None and args.max_images > 0 and images_processed_count >= args.max_images:
            #     print(f"Reached maximum image limit ({args.max_images}). Stopping.")
            #     break
            # --- End check ---

            try:
                relative_path, label_str = line.rsplit(' ', 1)
                label = int(label_str)
                full_image_path = os.path.join(args.data_dir, relative_path)

                if not os.path.exists(full_image_path):
                    continue

                rel_dir = os.path.dirname(relative_path)
                output_patch_subdir_check = os.path.join(output_patches_dir, rel_dir)
                # Check if the specific output directory for this image exists and has files
                if os.path.exists(output_patch_subdir_check) and len(os.listdir(output_patch_subdir_check)) > 0:
                    # print(f"Skipping already processed image: {relative_path}") # Optional: for confirmation
                    continue # Skip to the next image in the label file
                
                # Increment image counter *after* confirming the image exists
                images_processed_count += 1

                # Create directory structure mirroring input
                rel_dir = os.path.dirname(relative_path)
                output_patch_subdir = os.path.join(output_patches_dir, rel_dir)
                output_vis_subdir = os.path.join(output_vis_dir, rel_dir)
                os.makedirs(output_patch_subdir, exist_ok=True)
                os.makedirs(output_vis_subdir, exist_ok=True)

                # Load the full hyperspectral image
                image = np.load(full_image_path)
                if image.ndim == 2: image = np.expand_dims(image, axis=-1)
                if image.dtype == np.float64: image = image.astype(np.float32)
                img_h, img_w, img_c = image.shape

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
                masks_data = mask_generator.generate(rgb_image_for_sam)

                if not masks_data: continue

                # --- Filter masks based on datasets.py logic ---
                valid_masks = [
                    m for m in masks_data
                    if MIN_PILL_AREA <= m['area'] <= MAX_PILL_AREA and m['predicted_iou'] >= MIN_IOU_SCORE
                ]
                if not valid_masks: continue

                # --- Adjust mask pool and determine target patch count ---
                selection_pool = []
                num_valid = len(valid_masks)
                target_patches_for_image = 0
                areas = [m['area'] for m in valid_masks]
                median_area = statistics.median(areas) if areas else 0
                masks_with_diff = [(abs(m['area'] - median_area), m) for m in valid_masks]
                masks_with_diff.sort(key=lambda x: x[0])
                sorted_masks_by_closeness = [m[1] for m in masks_with_diff]

                if num_valid >= MIN_MASKS_FOR_ADJUSTMENT:
                    target_patches_for_image = args.num_patches_per_image
                    if num_valid > TARGET_MASK_COUNT:
                        selection_pool = sorted_masks_by_closeness[:TARGET_MASK_COUNT]
                    else:
                        num_to_add = TARGET_MASK_COUNT - num_valid
                        masks_to_duplicate = sorted_masks_by_closeness[:num_to_add]
                        selection_pool = valid_masks + masks_to_duplicate
                else:
                    target_patches_for_image = min(6, num_valid)
                    selection_pool = sorted_masks_by_closeness[:target_patches_for_image]
                if not selection_pool: continue

                selected_masks = selection_pool
                base_filename = os.path.splitext(os.path.basename(relative_path))[0]
                patches_extracted_for_image = 0

                for i, mask_info in enumerate(selected_masks):
                    bbox = mask_info['bbox']
                    adjusted_bbox = _adjust_bbox(bbox, image.shape)
                    if adjusted_bbox is None: continue

                    adj_x, adj_y, adj_w, adj_h = adjusted_bbox
                    patch_np = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]
                    if patch_np.size == 0: continue

                    resized_patch = resize_patch(patch_np, (args.patch_size, args.patch_size), device)
                    if resized_patch is None: continue

                    patch_filename_npy = f"{base_filename}_patch_{patches_extracted_for_image}.npy"
                    patch_filename_png = f"{base_filename}_patch_{patches_extracted_for_image}_vis.png"
                    output_patch_path_npy = os.path.join(output_patch_subdir, patch_filename_npy)
                    output_vis_path_png = os.path.join(output_vis_subdir, patch_filename_png)

                    try:
                        np.save(output_patch_path_npy, resized_patch.astype(np.float32))
                    except Exception as e:
                        print(f"Error saving NPY {output_patch_path_npy}: {e}")
                        continue

                    # --- Create and save the NEW visualization ---
                    try:
                        original_image_display = hsi_to_rgb_display(image)
                        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                        ax.imshow(original_image_display)
                        ax.set_title(f"{base_filename} - Patch {patches_extracted_for_image}\nArea: {mask_info['area']}, IoU: {mask_info['predicted_iou']:.2f}", fontsize=8)
                        ax.axis('off')
                        rect_x, rect_y, rect_w, rect_h = adjusted_bbox
                        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=1.5, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        plt.savefig(output_vis_path_png, bbox_inches='tight', pad_inches=0.1, dpi=150)
                        plt.close(fig)
                    except Exception as vis_e:
                        print(f"Error creating visualization {output_vis_path_png}: {vis_e}")
                        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
                    # --- End NEW visualization ---

                    rel_patch_path_npy = os.path.join("patches", rel_dir, patch_filename_npy)
                    all_patch_entries.append((rel_patch_path_npy, label))
                    patches_extracted_for_image += 1
                    patch_counter += 1

            except Exception as e:
                import traceback
                print(f"\n--- ERROR ---")
                print(f"Error processing line '{line}' corresponding to image '{full_image_path}':")
                print(f"Exception Type: {type(e)}")
                print(f"Exception Message: {e}")
                print("Traceback:")
                traceback.print_exc()
                print(f"-------------")
                continue

    except FileNotFoundError:
        print(f"Error: Label file not found at {args.label_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading label file {args.label_file}: {e}")
        exit(1)

    print(f"\nFinished processing {images_processed_count} images. Generated {patch_counter} patches.") # Updated message

    if not all_patch_entries:
        print("Error: No patches were generated. Check input data, SAM2 setup, and filtering parameters.")
        # Don't exit if max_images was set and might be 0 or very small
        if args.max_images is None or args.max_images > 0:
             exit(1)
        else:
             print("Note: No patches generated, but max_images was set to 0 or less.")


    # Save the new label file (only includes patches from processed images)
    print(f"Saving new label file to {new_label_file_path}")
    try:
        with open(new_label_file_path, 'w') as f:
            for rel_path, lbl in all_patch_entries:
                f.write(f"{rel_path} {lbl}\n")
    except Exception as e:
        print(f"Error writing new label file {new_label_file_path}: {e}")
        exit(1)

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess hyperspectral data using SAM2 for mask generation and patch extraction")
    # Essential arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing original hyperspectral image .npy files')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the text file containing relative original image paths and labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed patches, visualizations, and the new labels.txt')
    parser.add_argument('--sam2_checkpoint_path', type=str, required=True, help='Path to the SAM2 checkpoint file (.pt)')
    parser.add_argument('--patch_size', type=int, default=224, help='Target size (height and width) for the output patches')
    parser.add_argument('--num_patches_per_image', type=int, default=100, help='Target number of patches to generate per image if >= MIN_MASKS_FOR_ADJUSTMENT masks are found')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'], help='Device to use for SAM2 and resizing (cuda, cpu, or mps)')
    # --- New Argument ---
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process from the label file (default: process all)')

    # Arguments kept for compatibility with bash script, but not directly used
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
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)