import os, argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset # Keep Dataset import
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import collections # Import collections for defaultdict

from classification_model import ClassificationModel
from classification_cnn import train

from pathlib import Path # Import Path

import matplotlib.pyplot as plt
# import matplotlib.patches as patches # No longer needed for bbox visualization here

# --- Import the NEW dataset and collate function ---
from datasets import PreprocessedPatchDataset, collate_fn_skip_none_preprocessed
# --- End Import ---

# Define command-line arguments
parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--data_dir',
    type=str, required=True, help='Directory containing PREPROCESSED patches and labels.txt') # Updated help text
parser.add_argument('--label_file',
    type=str, required=True, help='Path to the labels.txt file WITHIN the data_dir') # Updated help text
# --- Remove arguments no longer needed ---
# parser.add_argument('--num_patches_per_image', ...) # Determined by preproc
# parser.add_argument('--sam2_checkpoint_path', ...)
# --- End Remove ---
parser.add_argument('--patch_size',
    type=int, default=224, help='Expected size of image patches (used for verification/resizing)')
parser.add_argument('--train_split_ratio',
    type=float, default=0.8, help='Proportion of PATCHES to use for training') # Updated help text

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

args = parser.parse_args()

# --- Helper Function to Extract Common Path ---
def get_common_path_part(relative_path_str):
    """
    Extracts the common part (Drug/Group/Mxxx/patch_y.npy) from the full relative path.
    Returns the common part string or None if the path structure is unexpected.
    """
    p = Path(relative_path_str)
    parts = p.parts
    if len(parts) >= 5:
        if parts[0].lower() == 'real' and len(parts) == 5: # real/Drug/Group/Mxxx/patch.npy
            return str(Path(*parts[1:])) # Drug/Group/Mxxx/patch.npy
        elif parts[0].lower() == 'fake' and parts[1].lower() == 'patches_augmented' and len(parts) == 6: # fake/patches_augmented/Drug/Group/Mxxx/patch.npy
            return str(Path(*parts[2:])) # Drug/Group/Mxxx/patch.npy
    return None
# --- End Helper Function ---


if __name__ == '__main__':
    # --- Remove multiprocessing setup if not needed, or keep if useful for other parts ---
    # try:
    #     mp.set_start_method('spawn', force=True)
    #     print("Multiprocessing start method set to 'spawn'.")
    # except RuntimeError as e:
    #     if "context has already been set" not in str(e):
    #          print(f"Warning: Could not set multiprocessing start method: {e}")
    # -----------------------------------------

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Remove SAM2 / Hydra loading ---
    # sam2_checkpoint = args.sam2_checkpoint_path
    # ... (all code related to model_cfg_name_rel, hydra.compose, cfg) ...
    # --- End Remove ---

    # --- Keep Transform setup ---
    transform_mean = [0.5] * args.num_channels if args.num_channels > 0 else None
    transform_std = [0.5] * args.num_channels if args.num_channels > 0 else None
    # --- End Transform setup ---

    print("Loading patch list and labels from preprocessed data...")
    all_patch_samples = [] # List of (relative_patch_path, label)
    class_labels = set()
    full_label_file_path = os.path.join(args.label_file) # Construct full path

    try:
        with open(full_label_file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    # Split the line (assuming format: 'patches/Drug/Group/Mxxx/patch_y.npy label')
                    relative_path, label_str = line.rsplit(' ', 1)
                    label = int(label_str)

                    # Basic check if relative path seems valid (starts with 'patches/')
                    # if not relative_path.startswith('patches' + os.path.sep):
                    #      print(f"Warning: Line {i+1}: Relative path '{relative_path}' might be incorrect. Expected to start with 'patches/'.")

                    all_patch_samples.append((relative_path, label))
                    class_labels.add(label)

                except ValueError as e:
                    print(f"Warning: Skipping malformed line {i+1} in {full_label_file_path}: '{line}' - Reason: {e}")

    except FileNotFoundError:
        print(f"Error: Preprocessed label file not found at {full_label_file_path}")
        exit(1)
    except Exception as e:
        print(f"Error reading preprocessed label file {full_label_file_path}: {e}")
        exit(1)

    if not all_patch_samples:
        raise ValueError("No valid patch samples found. Check preprocessed label file and data directory.")

    num_classes = len(class_labels)
    print(f"Found {len(all_patch_samples)} total patches belonging to {num_classes} classes.")

    # # --- Split the PATCHES into training and test sets ---
    # patch_paths = [s[0] for s in all_patch_samples]
    # patch_labels = [s[1] for s in all_patch_samples]
    
    # --- START: New Splitting Logic ---
    print("Identifying real sources, fakes, and unused real patches...")
    real_source_map = {} # common_path -> (full_real_path, label)
    fake_map = {}        # common_path -> (full_fake_path, label)
    unused_real_list = [] # list of (full_real_path, label)
    temp_real_map = {}   # common_path -> (full_real_path, label) - temporary holder

    for rel_path, label in all_patch_samples:
        common_part = get_common_path_part(rel_path)
        if common_part:
            p = Path(rel_path)
            if p.parts[0].lower() == 'real':
                temp_real_map[common_part] = (rel_path, label)
            elif p.parts[0].lower() == 'fake':
                fake_map[common_part] = (rel_path, label)
        # else: # Optional: Log paths that don't match expected structure
        #     print(f"Warning: Path '{rel_path}' did not match expected real/ or fake/ structure.")

    # Populate real_source_map and unused_real_list
    for common_part, real_data in temp_real_map.items():
        if common_part in fake_map:
            real_source_map[common_part] = real_data
        else:
            unused_real_list.append(real_data)
            
    print(f"Identified {len(real_source_map)} real source patches.")
    print(f"Identified {len(fake_map)} fake patches.")
    print(f"Identified {len(unused_real_list)} unused real patches.")

    # --- Split Paired Data (Groups A & B) ---
    train_samples = []
    test_samples = []
    paired_keys = list(real_source_map.keys())

    if paired_keys:
        if args.train_split_ratio < 1.0 and args.train_split_ratio > 0.0:
            print(f"Splitting {len(paired_keys)} source/fake pairs ({args.train_split_ratio*100:.1f}% train)...")
            train_keys, test_keys = train_test_split(
                paired_keys,
                train_size=args.train_split_ratio,
                random_state=42 # Use a fixed random state for reproducibility
            )

            for key in train_keys:
                train_samples.append(real_source_map[key]) # Add real source
                if key in fake_map:
                    train_samples.append(fake_map[key]) # Add corresponding fake
                else:
                     print(f"Warning: Missing fake pair for train key {key}")


            for key in test_keys:
                test_samples.append(real_source_map[key]) # Add real source
                if key in fake_map:
                    test_samples.append(fake_map[key]) # Add corresponding fake
                else:
                    print(f"Warning: Missing fake pair for test key {key}")

            print(f"Added {len(train_keys)*2} paired samples to train set (approx).")
            print(f"Added {len(test_keys)*2} paired samples to test set (approx).")

        else: # Use all paired data for training if ratio is 1.0
            print(f"Using all {len(paired_keys)} source/fake pairs for training.")
            for key in paired_keys:
                train_samples.append(real_source_map[key])
                if key in fake_map:
                    train_samples.append(fake_map[key])
                else:
                    print(f"Warning: Missing fake pair for train key {key}")

    else:
        print("No source/fake pairs found to split.")
        
    # --- FIX: Add unused real patches to the training set ---
    print(f"Adding {len(unused_real_list)} unused real patches to the training set.")
    train_samples.extend(unused_real_list)
    # --- END FIX ---

    # --- Final Shuffle and Summary ---
    random.seed(42) # Re-seed before final shuffle if desired
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    print("\n--- Final Split Summary ---")
    print(f"Total training samples: {len(train_samples)}")
    print(f"Total testing samples: {len(test_samples)}")

    # --- Save the TEST set paths and labels ---
    if test_samples:
        test_set_file_path = os.path.join(args.checkpoint_path, 'test_samples.txt')
        print(f"Saving test set file list ({len(test_samples)} samples) to: {test_set_file_path}")
        try:
            with open(test_set_file_path, 'w') as f_test:
                for path, label in test_samples:
                    f_test.write(f"{path} {label}\n") # Use original relative path
        except IOError as e:
            print(f"Error saving test set file list: {e}")
    else:
        print("Warning: No samples allocated to the test set.")
    # --- END: New Splitting Logic ---


    print("Creating training dataset instance from preprocessed patches...")
    # --- Instantiate the NEW dataset ---
    train_dataset = PreprocessedPatchDataset(
        data_dir=args.data_dir, # Base directory (e.g., './data_processed_patch/')
        samples=train_samples,  # List of (relative_path, label) for training
        num_channels=args.num_channels,
        transform_mean=transform_mean,
        transform_std=transform_std,
        target_size=(args.patch_size, args.patch_size),
        save_visualization_path='hyper_checkpoints/resnet/transform_viz', # Optional, can be None
        num_visualizations_to_save=5
    )
    # --- End Instantiate ---
    print(f"Training dataset size: {len(train_dataset)} patches")


    print("Creating DataLoader...")
    # --- Use the NEW collate_fn and potentially more workers ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.n_batch,
        shuffle=True,
        num_workers=4, # Increase workers (adjust based on your system)
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_skip_none_preprocessed # Use the new collate function
    )
    # --- End DataLoader Update ---
    print("DataLoader created.")


    # --- Modify Sample Saving Section ---
    print("Saving a few sample patches to files...")
    num_samples_to_save = 5
    save_dir = os.path.join(args.checkpoint_path, "sample_preprocessed_patches") # New dir name
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving sample patches visualizations to: {save_dir}")

    # Clear existing files
    print(f"Clearing existing files in {save_dir}...")
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
        except Exception as e: print(f'Failed to delete {file_path}. Reason: {e}')

    saved_count = 0
    if len(train_dataset) > 0:
        vis_indices = random.sample(range(len(train_dataset)), min(num_samples_to_save * 2, len(train_dataset)))
        idx_iter = iter(vis_indices)
    else:
        print("Warning: Training dataset is empty, cannot save samples.")
        idx_iter = iter([])

    # Keep hsi_to_rgb_display helper function
    def hsi_to_rgb_display(hsi_image):
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

    while saved_count < num_samples_to_save:
        try:
            idx = next(idx_iter)
            # Get patch tensor and label directly from the dataset
            patch_tensor, label = train_dataset[idx]

            if patch_tensor is None:
                print(f"Skipping visualization for index {idx} due to error in __getitem__.")
                continue

            # Get the relative path for filename purposes
            relative_path, _ = train_dataset.samples[idx] # Access samples directly
            base_filename = os.path.splitext(os.path.basename(relative_path))[0]

            # --- Save the patch .npy file (optional, as it's already saved) ---
            # patch_np_final = patch_tensor.cpu().numpy().transpose((1, 2, 0)) # CHW -> HWC
            # patch_filename = f"{base_filename}_sample_{idx}_label{label}.npy"
            # output_patch_path = os.path.join(save_dir, patch_filename)
            # np.save(output_patch_path, patch_np_final)
            # --- End saving patch .npy ---

            # --- Create and save visualization OF THE PATCH ---
            try:
                # Convert patch tensor (potentially normalized) back to displayable format
                # Note: If normalized, this visualization might look different than the original patch vis
                patch_np_vis = patch_tensor.cpu().numpy().transpose((1, 2, 0)) # CHW -> HWC
                patch_display = hsi_to_rgb_display(patch_np_vis) # Use helper on the patch data

                fig, ax = plt.subplots(1, 1, figsize=(6, 6)) # Smaller figure
                ax.imshow(patch_display)
                ax.set_title(f"Patch: {os.path.basename(relative_path)}\nIndex: {idx}, Label: {label}")
                ax.axis('off')

                vis_filename = f"{base_filename}_vis_{idx}_label{label}.png"
                output_vis_path = os.path.join(save_dir, vis_filename)
                plt.savefig(output_vis_path, bbox_inches='tight')
                plt.close(fig)

                saved_count += 1
            except Exception as vis_e:
                print(f"Error creating visualization for index {idx}: {vis_e}")
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
            # --- End visualization ---

        except StopIteration:
            print(f"Warning: Ran out of indices while trying to save {num_samples_to_save} samples.")
            break
        except Exception as loop_e:
            print(f"Error in visualization loop for index {idx}: {loop_e}")
            continue

    print(f"Finished saving {saved_count} sample patch visualizations.")
    # --- End Modify Sample Saving ---


    # --- Keep Model Initialization and Training Call ---
    print(f"Initializing model with {args.num_channels} input channels and {num_classes} classes on {args.device}...")
    model = ClassificationModel(
        encoder_type=args.encoder_type,
        input_channels=args.num_channels,
        num_classes=num_classes,
        device=args.device
    )
    model.to(args.device)
    print("Model initialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    print("Optimizer initialized.")

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
    # --- End Keep ---
