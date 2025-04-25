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
from omegaconf import OmegaConf # Add OmegaConf import

import matplotlib.pyplot as plt

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

    if "sam2.1_hiera_tiny" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_t.yaml")
    elif "sam2.1_hiera_small" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_s.yaml")
    elif "sam2.1_hiera_base_plus" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_b+.yaml") # Correct name for Hydra
    elif "sam2.1_hiera_large" in os.path.basename(sam2_checkpoint):
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_l.yaml") # Correct name for Hydra
    # Add older sam2 checkpoints if needed
    elif "sam2_hiera_tiny" in os.path.basename(sam2_checkpoint):
         model_cfg_name = os.path.join(base_config_name, "sam2/sam2_hiera_t.yaml")
    elif "sam2_hiera_small" in os.path.basename(sam2_checkpoint):
         model_cfg_name = os.path.join(base_config_name, "sam2/sam2_hiera_s.yaml")
    elif "sam2_hiera_base_plus" in os.path.basename(sam2_checkpoint):
         model_cfg_name = os.path.join(base_config_name, "sam2/sam2_hiera_b+.yaml")
    elif "sam2_hiera_large" in os.path.basename(sam2_checkpoint):
         model_cfg_name = os.path.join(base_config_name, "sam2/sam2_hiera_l.yaml")
    else:
        print(f"Warning: Could not determine config name for checkpoint {sam2_checkpoint}. Using default large config.")
        model_cfg_name = os.path.join(base_config_name, "sam2.1/sam2.1_hiera_l.yaml") # Default if unsure

    # Remove the '.yaml' extension for Hydra's compose function
    model_cfg_name = model_cfg_name.replace(".yaml", "")

    print(f"Using SAM2 checkpoint: {sam2_checkpoint}")
    print(f"Using SAM2 config name for Hydra: {model_cfg_name}")

    # sam2 = build_sam2(model_cfg_name, sam2_checkpoint, device=args.device, apply_postprocessing=False)

    # mask_generator = SAM2AutomaticMaskGenerator(sam2)

    transform = transforms.Compose([
        # transforms.ToTensor(), # Convert numpy array to tensor
        # Add other transforms if needed, e.g., normalization
        transforms.Normalize(mean=[0.5]*args.num_channels, std=[0.5]*args.num_channels), # Normalize to [0, 1] range
    ])

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
        # Pass SAM2 config details instead of the model
        sam2_checkpoint_path=args.sam2_checkpoint_path, # Path to weights
        sam2_config_name=model_cfg_name,     # Name for Hydra (e.g., 'configs/sam2.1/sam2.1_hiera_b+')
        device=str(args.device),               # Device string ('cuda' or 'cpu')
        num_patches_per_image=args.num_patches_per_image,
        transform=transform,              # The corrected transform pipeline
        target_size=(args.patch_size, args.patch_size)
    )
    print(f"Training dataset size: {len(train_dataset)} patches")

    print("Creating DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.n_batch,
        shuffle=True,
        collate_fn=collate_fn_skip_none,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True
    )
    print("DataLoader created.")

    print("Saving a few sample patches to files...")
    num_samples_to_save = 5
    save_dir = os.path.join(args.checkpoint_path, "sample_patches") # Directory to save patches
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving patches to: {save_dir}")

    saved_count = 0
    # Try to get a few more indices in case some samples fail in __getitem__
    vis_indices = random.sample(range(len(train_dataset)), min(num_samples_to_save * 3, len(train_dataset)))
    idx_iter = iter(vis_indices)

    while saved_count < num_samples_to_save:
        try:
            idx = next(idx_iter)
            # Retrieve original path info for filename (optional but helpful)
            original_image_path, _ = train_dataset.samples_for_iteration[idx] # Access internal list
            base_filename = os.path.splitext(os.path.basename(original_image_path))[0]

            patch_tensor, label = train_dataset[idx]

            if patch_tensor is None: # Skip if __getitem__ returned None
                 continue

            # Convert tensor back to numpy (CHW -> HWC) for display/saving
            patch_np = patch_tensor.numpy().transpose((1, 2, 0))
    
            # Select bands to display (e.g., first 3, or specific RGB indices)
            # Option 1: First 3 channels
            # display_patch = patch_np[:, :, :3]
    
            # Option 2: Specific indices (ensure they exist)
            # Example: R=50, G=30, B=10 (adjust if your channel count is different)
            if patch_np.shape[2] >= 50: # Check if enough channels exist
                 display_patch = patch_np[:, :, [50, 30, 10]] # Example indices
            else: # Fallback: Use first 3 or mean
                 display_patch = patch_np[:, :, :min(3, patch_np.shape[2])]
                 if display_patch.shape[2] == 1: # If only one channel, make it grayscale RGB
                     display_patch = np.concatenate([display_patch]*3, axis=-1)
                 elif display_patch.shape[2] == 2: # Handle 2 channels if necessary
                     # Example: duplicate one channel or add a zero channel
                     display_patch = np.concatenate([display_patch, display_patch[:,:,:1]], axis=-1)
    
            # Normalize for display/saving (0-1 range is good for imsave)
            min_val = np.min(display_patch)
            max_val = np.max(display_patch)
            if max_val > min_val:
                 display_patch = (display_patch - min_val) / (max_val - min_val)
            else:
                 display_patch = np.zeros_like(display_patch) # Handle constant image case
            display_patch = np.clip(display_patch, 0, 1) # Ensure values are in [0, 1]

            # Construct save path
            save_filename = f"sample_{saved_count}_idx{idx}_label{label}_{base_filename}.png"
            save_path = os.path.join(save_dir, save_filename)

            # Save the image using matplotlib.pyplot.imsave
            plt.imsave(save_path, display_patch)

            saved_count += 1

        except StopIteration:
             print(f"Warning: Ran out of indices while trying to save {num_samples_to_save} samples.")
             break # Exit loop if we run out of indices
        except Exception as e:
             # Print error and original path for context
             try:
                 failed_path = train_dataset.samples_for_iteration[idx][0]
             except:
                 failed_path = "unknown"
             print(f"Error saving sample patch from index {idx} (orig path: {failed_path}): {e}")
             # Continue to try next index

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
