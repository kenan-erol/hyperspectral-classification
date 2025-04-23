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


class HyperspectralPatchDataset(Dataset):
    def __init__(self, data_dir, samples_list, sam2_model, num_patches_per_image=5, transform=None, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.num_patches_per_image = num_patches_per_image
        self.sam2 = sam2_model  # Use the passed SAM2 model

        # Store the original image paths and labels provided
        self.original_samples = samples_list

        # Create multiple entries for patch sampling
        self.samples_for_iteration = []
        for image_path, label in self.original_samples:
            for _ in range(self.num_patches_per_image):
                self.samples_for_iteration.append((image_path, label))

    def __len__(self):
        return len(self.samples_for_iteration)

    def _adjust_bbox(self, bbox, img_shape):
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

        return new_x, new_y, new_w, new_h

    def __getitem__(self, idx):
        original_image_path, label = self.samples_for_iteration[idx]
        full_path = os.path.join(self.data_dir, original_image_path)

        try:
            image = np.load(full_path)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)

            img_h, img_w = image.shape[:2]

            masks_bboxes = self.sam2.generate_masks(image)

            if not masks_bboxes:
                print(f"Warning: No masks found for {full_path}. Using random crop.")
                rand_x = random.randint(0, max(0, img_w - self.target_size[1]))
                rand_y = random.randint(0, max(0, img_h - self.target_size[0]))
                patch = image[rand_y:rand_y+min(self.target_size[0], img_h-rand_y),
                              rand_x:rand_x+min(self.target_size[1], img_w-rand_x), :]
            else:
                selected_bbox = random.choice(masks_bboxes)
                adj_x, adj_y, adj_w, adj_h = self._adjust_bbox(selected_bbox, image.shape)
                patch = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]

            if patch.size == 0:
                print(f"Warning: Empty patch generated for {full_path}. Using random crop.")
                rand_x = random.randint(0, max(0, img_w - self.target_size[1]))
                rand_y = random.randint(0, max(0, img_h - self.target_size[0]))
                patch = image[rand_y:rand_y+min(self.target_size[0], img_h-rand_y),
                              rand_x:rand_x+min(self.target_size[1], img_w-rand_x), :]

            if self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = torch.from_numpy(patch.transpose((2, 0, 1))).float()

            if patch_tensor.shape[1:] != self.target_size:
                patch_tensor = F.interpolate(
                    patch_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            return patch_tensor, label

        except Exception as e:
            print(f"Error processing {full_path} at index {idx}: {e}")
            return None


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
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

    sam2 = build_sam2(model_cfg_name, sam2_checkpoint, device=args.device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    transform = transforms.Compose([
        transforms.ToTensor(), # Convert numpy array to tensor
        # Add other transforms if needed, e.g., normalization
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
                    full_path_check = os.path.join(args.data_dir, relative_path)

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
        sam2_model=mask_generator,  # Pass the initialized model
        num_patches_per_image=args.num_patches_per_image,
        transform=transform,
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