import os, argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from classification_model import ClassificationModel
from classification_cnn import train
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import torch.multiprocessing as mp
import hydra # Add hydra import
from omegaconf import OmegaConf, DictConfig # Add OmegaConf import

import matplotlib.pyplot as plt
import matplotlib.patches as patches # <-- Import patches

from datasets import HyperspectralPatchDataset, collate_fn_skip_none

#from dataset import HyperspectralDataset
#from log_utils import setup_logging, log_metrics
#import logging

# Define command-line arguments
parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--data_dir',
    type=str, required=True, help='Directory containing hyperspectral data subfolders')
parser.add_argument('--label_file',
    type=str, required=True, help='File containing relative image paths and labels')
parser.add_argument('--num_patches_per_image',
    type=int, default=5, help='Number of patches to sample per image')
parser.add_argument('--patch_size',
    type=int, default=224, help='Size of image patches')
parser.add_argument('--train_split_ratio',
    type=float, default=0.8, help='Proportion of unique images (pills) to use for training')

# Network settings
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type to build: vggnet11, resnet18')
parser.add_argument('--num_channels',
    type=int, required=True, help='Number of spectral bands in hyperspectral data')

# Training settings
parser.add_argument('--n_epoch',
    type=int, required=True, help='Number of passes through the full training dataset')
parser.add_argument('--learning_rate',
    type=float, required=True, help='Step size to update parameters')
parser.add_argument('--learning_rate_decay',
    type=float, required=True, help='Scaling factor to decrease learning rate at the end of each decay period')
parser.add_argument('--learning_rate_period',
    type=float, required=True, help='Number of epochs before reducing/decaying learning rate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path directory to save checkpoints and Tensorboard summaries')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

parser.add_argument('--sam2_checkpoint_path',
    type=str, default="./sam2/checkpoints/sam2.1_hiera_base_plus.pt", 
    help='Path to the SAM2 checkpoint file')

args = parser.parse_args()


if __name__ == '__main__':
    # --- Set multiprocessing start method ---
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
             print(f"Warning: Could not set multiprocessing start method: {e}")
    # -----------------------------------------
    
    os.makedirs(args.checkpoint_path, exist_ok=True)

    sam2_checkpoint = args.sam2_checkpoint_path # Use the argument for checkpoint path
    #model_cfg = "/configs/sam2.1/sam2.1_hiera_l.yaml" # Assuming this config path is correct

    base_config_name = "configs" # <--- Base directory name within the sam2 package
    
    model_cfg_name_rel = None
    sam2_checkpoint_basename = os.path.basename(args.sam2_checkpoint_path)

    if "sam2.1_hiera_tiny" in sam2_checkpoint_basename:
        model_cfg_name_rel = "sam2.1/sam2.1_hiera_t" # Relative path without .yaml
    elif "sam2.1_hiera_small" in sam2_checkpoint_basename:
        model_cfg_name_rel = "sam2.1/sam2.1_hiera_s"
    elif "sam2.1_hiera_base_plus" in sam2_checkpoint_basename:
        model_cfg_name_rel = "sam2.1/sam2.1_hiera_b+"
    elif "sam2.1_hiera_large" in sam2_checkpoint_basename:
        model_cfg_name_rel = "sam2.1/sam2.1_hiera_l"
    else:
        print(f"Warning: Could not determine specific config name for {args.sam2_checkpoint_path}. Using default.")
        model_cfg_name_rel = "sam2.1/sam2.1_hiera_l" # Defaulting to large

    print(f"Using SAM2 config name for Hydra: {model_cfg_name_rel}")

    # sam2 = build_sam2(model_cfg_name, sam2_checkpoint, device=args.device, apply_postprocessing=False)

    # mask_generator = SAM2AutomaticMaskGenerator(sam2)
    
    # --- Load Hydra Config in Main Process ---
    try:
        full_config_name = f"configs/{model_cfg_name_rel}"
        print(f"Attempting to compose with full config name: {full_config_name}")
        # --- End config name construction ---

        # Directly compose the config using the full name
        cfg = hydra.compose(config_name=full_config_name)
        print("Hydra config loaded successfully in main process.")

    except Exception as e:
        print(f"Error: Failed to load Hydra config '{full_config_name}': {e}") # Use full name in error
        if "Could not find config" in str(e) or "Cannot find primary config" in str(e):
             print(f"Suggestion: Hydra initialized by 'sam2', but couldn't find '{full_config_name}'.")
             print("Check the structure within the 'sam2/sam2/configs' directory and how 'initialize_config_module' registers paths.")
        elif "GlobalHydra is already initialized" in str(e):
             # This shouldn't happen now, but keep the check just in case
             print("Error: Still getting 'already initialized'. Check for other Hydra initializations.")
        else:
             print("An unexpected error occurred during Hydra composition.")
        exit(1)
    # --- End Hydra Config Loading ---

    # transform = transforms.Compose([
    #     # transforms.ToTensor(), # Convert numpy array to tensor
    #     # Add other transforms if needed, e.g., normalization
    #     transforms.Normalize(mean=[0.5]*args.num_channels, std=[0.5]*args.num_channels), # Normalize to [0, 1] range
    # ])
    
    transform_mean = [0.5] * args.num_channels if args.num_channels > 0 else None
    transform_std = [0.5] * args.num_channels if args.num_channels > 0 else None


    print("Loading image list and labels...")
    all_samples = []
    class_labels = set()
    # ... inside the if __name__ == '__main__': block ...
    try:
        with open(args.label_file, 'r') as f:
            for i, line in enumerate(f): # Add line number for better debugging
                line = line.strip() # Remove leading/trailing whitespace
                if not line: # Skip empty lines
                    continue
                try:
                    # --- More robust splitting ---
                    last_space_idx = line.rfind(' ')

                    # Check if a space was found
                    if last_space_idx == -1:
                        raise ValueError("No space found to separate path and label.")

                    # Extract path and label parts
                    relative_path = line[:last_space_idx].strip() # Strip path just in case
                    label_str = line[last_space_idx:].strip()     # Strip label part

                    # Check if parts are empty after stripping
                    if not relative_path or not label_str:
                        raise ValueError("Empty path or label after split.")
                    # --- End robust splitting ---

                    label = int(label_str) # Convert label part to integer

                    # # --- Add Debugging ---
                    # print(f"DEBUG: args.data_dir = {args.data_dir}")
                    # print(f"DEBUG: relative_path = {relative_path}")
                    full_path_check = os.path.join(args.data_dir, relative_path)
                    # print(f"DEBUG: Checking path: {full_path_check}")
                    # # --- End Debugging ---

                    if os.path.exists(full_path_check):
                        all_samples.append((relative_path, label))
                        class_labels.add(label)
                    else:
                        print(f"Warning: Image file not found {full_path_check}, skipping line {i+1}.")

                except ValueError as e:
                    # Handle lines that don't have the expected format
                    # Include the specific reason from the ValueError
                    print(f"Warning: Skipping malformed line {i+1} in {args.label_file}: '{line}' - Reason: {e}")

    except FileNotFoundError:
        print(f"Error: Label file not found at {args.label_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading label file {args.label_file}: {e}")
        exit(1)

    if not all_samples:
        raise ValueError("No valid samples found. Check label file and data directory.")

    num_classes = len(class_labels)
    print(f"Found {len(all_samples)} unique images belonging to {num_classes} classes.")

    image_paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_paths, _, train_labels, _ = train_test_split(
        image_paths, labels,
        train_size=args.train_split_ratio,
        random_state=42,
        stratify=labels
    )

    train_samples = list(zip(train_paths, train_labels))

    print(f"Splitting dataset: Train={len(train_samples)} unique images")

    print("Creating training dataset instance...")
    train_dataset = HyperspectralPatchDataset(
        args.data_dir,
        train_samples,
        sam2_checkpoint_path=args.sam2_checkpoint_path,
        sam2_model_config=cfg.model, # Pass the loaded config object (using original name)
        device=str(args.device), # Pass device string
        num_patches_per_image=args.num_patches_per_image,
        # --- Pass transform parameters ---
        transform_mean=transform_mean,
        transform_std=transform_std,
        num_channels=args.num_channels,
        # --- End transform parameters ---
        target_size=(args.patch_size, args.patch_size)
    )
    print(f"Training dataset size: {len(train_dataset)} patches")

    print("Creating DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.n_batch,
        shuffle=True,
        num_workers=4, # As specified in log
        pin_memory=True, # Good practice with CUDA
        drop_last=True, # As specified in log comparison
        collate_fn=collate_fn_skip_none # Keep this to handle None returns
    )
    print("DataLoader created.")

    print("Saving a few sample patches to files...")
    num_samples_to_save = 5 # Or get from args
    save_dir = os.path.join(args.checkpoint_path, "sample_patches_with_bbox") # Changed dir name
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving sample patches and visualizations to: {save_dir}")

    saved_count = 0
    vis_indices = random.sample(range(len(train_dataset)), min(num_samples_to_save * 5, len(train_dataset)))
    idx_iter = iter(vis_indices)

    # Helper function to create displayable RGB from HSI (using mean)
    def hsi_to_rgb_display(hsi_image):
        # ... (keep the helper function as defined before) ...
        if hsi_image.ndim == 2: hsi_image = np.expand_dims(hsi_image, axis=-1)
        img_h, img_w, img_c = hsi_image.shape
        if img_c == 0: return np.zeros((img_h, img_w, 3), dtype=np.uint8)

        display_img = np.mean(hsi_image, axis=2)
        min_val, max_val = np.min(display_img), np.max(display_img)
        if max_val > min_val:
            display_img = ((display_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            display_img = np.zeros((img_h, img_w), dtype=np.uint8)
        return np.stack([display_img]*3, axis=-1) # Stack to 3 channels


    while saved_count < num_samples_to_save:
        try:
            idx = next(idx_iter)
            relative_path, _ = train_dataset.samples_for_iteration[idx]
            full_image_path = os.path.join(args.data_dir, relative_path)
            base_filename = os.path.splitext(os.path.basename(relative_path))[0]

            # Get patch, label, and the bbox used for cropping
            patch_tensor, label, bbox = train_dataset[idx] # Unpack bbox

            # Skip if dataset __getitem__ failed
            if patch_tensor is None or bbox is None:
                 print(f"Skipping visualization for index {idx} due to error in __getitem__.")
                 continue

            # --- Save the patch .npy file ---
            # Move tensor to CPU for numpy conversion and saving
            patch_np_final = patch_tensor.cpu().numpy().transpose((1, 2, 0)) # CHW -> HWC
            patch_filename = f"{base_filename}_patch_{idx}_label{label}.npy"
            output_patch_path = os.path.join(save_dir, patch_filename)
            np.save(output_patch_path, patch_np_final)
            # --- End saving patch .npy ---

            # --- Create and save the visualization ---
            try:
                original_image_np = np.load(full_image_path)
                if original_image_np.ndim == 2: original_image_np = np.expand_dims(original_image_np, axis=-1)
                if original_image_np.dtype == np.float64: original_image_np = original_image_np.astype(np.float32)
                original_image_display = hsi_to_rgb_display(original_image_np)

                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(original_image_display)
                ax.set_title(f"Original: {os.path.basename(relative_path)}\nIndex: {idx}, Label: {label}")
                ax.axis('off')
                
                # Add the bounding box (use the potentially float values returned for accuracy)
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                vis_filename = f"{base_filename}_vis_{idx}_label{label}.png"
                output_vis_path = os.path.join(save_dir, vis_filename)
                plt.savefig(output_vis_path, bbox_inches='tight')
                plt.close(fig)

                saved_count += 1
            except FileNotFoundError:
                 print(f"Error: Original image file not found for visualization: {full_image_path}")
            except Exception as vis_e:
                 print(f"Error creating visualization for index {idx}: {vis_e}")
                 plt.close(fig) # Ensure figure is closed even on error
            # --- End visualization ---

        except StopIteration:
            print(f"Warning: Ran out of indices while trying to save {num_samples_to_save} samples.")
            break
        except Exception as loop_e:
            print(f"Error in visualization loop for index {idx}: {loop_e}")
            continue

    print(f"Finished saving {saved_count} sample patches and visualizations.")
    # --- End Save Sample Patches ---

    # Remove the plt.show() and related figure setup
    # plt.tight_layout() # Not needed
    # plt.show() # Remove this line
# --- End Visualization Code ---

    print(f"Initializing model with {args.num_channels} input channels and {num_classes} classes on {args.device}...")
    model = ClassificationModel(
        encoder_type=args.encoder_type,
        input_channels=args.num_channels,
        num_classes=num_classes,
        device=args.device
    )
    model.to(args.device)
    print("Model initialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    train(
        model=model,
        dataloader=train_dataloader,
        n_epoch=args.n_epoch,
        optimizer=optimizer,
        learning_rate_decay=args.learning_rate_decay,
        learning_rate_decay_period=args.learning_rate_period,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    print("Training finished.")
