# analyze_mask_areas.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import hydra
from omegaconf import OmegaConf

# Assuming sam2 package is installed or accessible
from sam2.build_sam import build_sam2, _load_checkpoint
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- Configuration (Adjust these paths/names) ---
SAM2_CHECKPOINT = "./sam2/checkpoints/sam2.1_hiera_base_plus.pt" # Or your checkpoint
MODEL_CFG_REL = "sam2.1/sam2.1_hiera_b+" # Relative config name used in training
CONFIG_DIR_ABS = os.path.abspath("./sam2/sam2/configs") # Absolute path to sam2 configs dir
DATA_DIR = "./data_processed/drop-4-npy/" # Path to your .npy files
LABEL_FILE = "./data_processed/drop-4-npy/labels.txt" # Path to your labels file
NUM_IMAGES_TO_ANALYZE = 20
OUTPUT_VIS_DIR = "./mask_analysis_vis"
# --- End Configuration ---

def hsi_to_rgb_display(hsi_image):
    # Helper to convert HSI to displayable RGB (same as in datasets.py)
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

def show_anns_with_area(anns, ax):
    # Modified show_anns to display area
    if not anns: return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    img_overlay = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img_overlay[:,:,3] = 0 # Transparent background
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]]) # Semi-transparent
        img_overlay[m] = color_mask
        # Add area text
        y, x = np.where(m)
        if len(x) > 0 and len(y) > 0:
             center_x, center_y = np.mean(x), np.mean(y)
             ax.text(center_x, center_y, str(ann['area']), color='white', fontsize=6, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=0.1, boxstyle='round,pad=0.1'))
    ax.imshow(img_overlay)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # --- Load SAM2 Model ---
    print("Loading SAM2 model...")
    cfg = None
    try:
        with hydra.initialize_config_dir(config_dir=CONFIG_DIR_ABS, version_base=None):
            full_config_name = f"configs/{MODEL_CFG_REL}"
            cfg = hydra.compose(config_name=full_config_name)
        print("Hydra config loaded.")
        sam2_model = hydra.utils.instantiate(cfg.model)
        _load_checkpoint(sam2_model, SAM2_CHECKPOINT)
        sam2_model.to(device)
        sam2_model.eval()
        mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
        print("SAM2 model and generator initialized.")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        exit(1)
    # --- End Load SAM2 Model ---

    # --- Load Image Paths ---
    image_paths = []
    try:
        with open(LABEL_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_paths.append(parts[0])
    except FileNotFoundError:
        print(f"Error: Label file not found at {LABEL_FILE}")
        exit(1)

    if not image_paths:
        print("Error: No image paths found in label file.")
        exit(1)

    # Select a subset
    image_paths = list(set(image_paths)) # Get unique image paths
    if len(image_paths) > NUM_IMAGES_TO_ANALYZE:
        import random
        image_paths = random.sample(image_paths, NUM_IMAGES_TO_ANALYZE)
    print(f"Analyzing {len(image_paths)} images...")
    # --- End Load Image Paths ---

    all_areas = []

    for i, rel_path in enumerate(image_paths):
        full_path = os.path.join(DATA_DIR, rel_path)
        print(f"Processing [{i+1}/{len(image_paths)}]: {full_path}")
        try:
            image_hsi = np.load(full_path)
            if image_hsi.ndim == 2: image_hsi = np.expand_dims(image_hsi, axis=-1)
            if image_hsi.dtype == np.float64: image_hsi = image_hsi.astype(np.float32)

            # Prepare for SAM
            image_rgb_display = hsi_to_rgb_display(image_hsi) # For visualization
            image_for_sam = np.mean(image_hsi, axis=2)
            min_val, max_val = np.min(image_for_sam), np.max(image_for_sam)
            if max_val > min_val:
                image_for_sam = ((image_for_sam - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                image_for_sam = np.zeros(image_for_sam.shape[:2], dtype=np.uint8)
            image_for_sam = np.stack([image_for_sam]*3, axis=-1)

            # Generate masks
            masks_data = mask_generator.generate(image_for_sam)

            if not masks_data:
                print("  No masks generated.")
                continue

            # Collect areas
            areas = [m['area'] for m in masks_data]
            all_areas.extend(areas)
            print(f"  Generated {len(masks_data)} masks. Areas: {sorted(areas)}")

            # Save visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb_display)
            show_anns_with_area(masks_data, plt.gca())
            plt.title(f"{os.path.basename(rel_path)}\n{len(masks_data)} masks generated")
            plt.axis('off')
            vis_filename = os.path.join(OUTPUT_VIS_DIR, f"{os.path.splitext(os.path.basename(rel_path))[0]}_masks.png")
            plt.savefig(vis_filename, bbox_inches='tight')
            plt.close()

        except FileNotFoundError:
            print(f"  Error: File not found: {full_path}")
        except Exception as e:
            print(f"  Error processing file {full_path}: {e}")
            import traceback
            traceback.print_exc()

    # --- Plot Histogram ---
    if all_areas:
        plt.figure(figsize=(12, 6))
        plt.hist(all_areas, bins=100, log=True) # Use log scale for y-axis if areas vary widely
        plt.title("Histogram of Mask Areas")
        plt.xlabel("Mask Area (pixels)")
        plt.ylabel("Frequency (log scale)")
        hist_filename = os.path.join(OUTPUT_VIS_DIR, "_area_histogram.png")
        plt.savefig(hist_filename)
        plt.close()
        print(f"\nHistogram saved to {hist_filename}")
        print(f"Min area found: {min(all_areas)}, Max area found: {max(all_areas)}")
    else:
        print("\nNo areas collected to generate histogram.")

    print("Analysis complete.")
