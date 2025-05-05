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

import matplotlib.pyplot as plt
from datasets import PreprocessedPatchDataset, collate_fn_skip_none_preprocessed

parser = argparse.ArgumentParser()

parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--data_dir',
    type=str, required=True, help='Directory containing PREPROCESSED patches and labels.txt')
parser.add_argument('--label_file',
    type=str, required=True, help='Path to the labels.txt file WITHIN the data_dir')
parser.add_argument('--patch_size',
    type=int, default=224, help='Expected size of image patches (used for verification/resizing)')
parser.add_argument('--train_split_ratio',
    type=float, default=0.8, help='Proportion of PATCHES to use for training')

parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type to build: vggnet11, resnet18')
parser.add_argument('--num_channels',
    type=int, required=True, help='Number of spectral bands in hyperspectral data')

parser.add_argument('--n_epoch',
    type=int, required=True, help='Number of passes through the full training dataset')
parser.add_argument('--learning_rate',
    type=float, required=True, help='Step size to update parameters')
parser.add_argument('--learning_rate_decay',
    type=float, required=True, help='Scaling factor to decrease learning rate at the end of each decay period')
parser.add_argument('--learning_rate_period',
    type=float, required=True, help='Number of epochs before reducing/decaying learning rate')

parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path directory to save checkpoints and Tensorboard summaries')

parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()


if __name__ == '__main__':

    os.makedirs(args.checkpoint_path, exist_ok=True)
    transform_mean = [0.5] * args.num_channels if args.num_channels > 0 else None
    transform_std = [0.5] * args.num_channels if args.num_channels > 0 else None

    print("Loading patch list and labels from preprocessed data...")
    all_patch_samples = []
    class_labels = set()
    full_label_file_path = os.path.join(args.label_file)

    try:
        with open(full_label_file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    relative_path, label_str = line.rsplit(' ', 1)
                    label = int(label_str)

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
    patch_paths = [s[0] for s in all_patch_samples]
    patch_labels = [s[1] for s in all_patch_samples]

    if args.train_split_ratio < 1.0 and args.train_split_ratio > 0.0:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            patch_paths, patch_labels,
            train_size=args.train_split_ratio,
            random_state=42,
            stratify=patch_labels
        )
        train_samples = list(zip(train_paths, train_labels))
        test_samples = list(zip(test_paths, test_labels))
        print(f"Split data: {len(train_samples)} training patches, {len(test_samples)} test patches.")

        # --- SAVE THE TEST SET LIST ---
        test_set_file_path = os.path.join(args.checkpoint_path, 'test_samples.txt')
        print(f"Saving test set file list to: {test_set_file_path}")
        try:
            with open(test_set_file_path, 'w') as f_test:
                for path, label in test_samples:
                    f_test.write(f"{path} {label}\n")
        except IOError as e:
            print(f"Error saving test set file list: {e}")
        # --- END SAVE TEST SET ---
    else:
        train_samples = all_patch_samples
        print(f"Using all {len(train_samples)} patches for training (train_split_ratio >= 1.0).")
    # --- End Patch Splitting ---


    print("Creating training dataset instance from preprocessed patches...")
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.n_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_skip_none_preprocessed
    )
    # --- End DataLoader ---
    print("DataLoader created.")
    
    print("Saving a few sample patches to files...")
    num_samples_to_save = 5
    save_dir = os.path.join(args.checkpoint_path, "sample_preprocessed_patches")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving sample patches visualizations to: {save_dir}")
    
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
            patch_tensor, label = train_dataset[idx]

            if patch_tensor is None:
                print(f"Skipping visualization for index {idx} due to error in __getitem__.")
                continue

            relative_path, _ = train_dataset.samples[idx]
            base_filename = os.path.splitext(os.path.basename(relative_path))[0]

            # --- Create and save visualization OF THE PATCH ---
            try:
                patch_np_vis = patch_tensor.cpu().numpy().transpose((1, 2, 0)) # CHW -> HWC
                patch_display = hsi_to_rgb_display(patch_np_vis)

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
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