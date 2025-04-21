import os, argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F

from classification_model import ClassificationModel
from classification_cnn import train
from sam2 import SAM2  # Import SAM2 for mask generation

# Define command-line arguments
parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--data_dir',
    type=str, required=True, help='Directory containing hyperspectral data')
parser.add_argument('--label_file',
    type=str, required=True, help='File containing image paths and labels')
parser.add_argument('--num_patches_per_image',
    type=int, default=5, help='Number of patches to sample per image')
parser.add_argument('--patch_size',
    type=int, default=224, help='Size of image patches')
parser.add_argument('--train_split_ratio',
    type=float, default=0.8, help='Proportion of data to use for training')

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


class HyperspectralPatchDataset(Dataset):
    def __init__(self, data_dir, label_file, num_patches_per_image=5, transform=None, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        self.sam2 = SAM2()  # Initialize SAM2

        # Load original image paths and labels
        original_samples = []
        with open(label_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                full_path = os.path.join(data_dir, image_path)
                if os.path.exists(full_path):
                    original_samples.append((full_path, int(label)))
                else:
                    print(f"Warning: Image file not found {full_path}")

        # Create multiple entries for each image to sample patches
        for image_path, label in original_samples:
            for _ in range(num_patches_per_image):
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

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
        image_path, label = self.samples[idx]

        try:
            # Load the hyperspectral image (H, W, C) or (C, H, W)
            # Assuming np.load results in (H, W, C) for consistency
            image = np.load(image_path)  # Shape: (H, W, C) or similar
            if image.ndim == 2:  # Handle grayscale case if necessary
                image = np.expand_dims(image, axis=-1)

            img_h, img_w = image.shape[:2]

            # Generate masks using SAM2 (returns list of bounding boxes [x, y, w, h])
            masks_bboxes = self.sam2.generate_masks(image)  # Adapt based on actual SAM2 output

            if not masks_bboxes:
                # Handle case with no masks: use random crop as fallback
                print(f"Warning: No masks found for {image_path}. Using random crop.")
                rand_x = random.randint(0, max(0, img_w - self.target_size[1]))
                rand_y = random.randint(0, max(0, img_h - self.target_size[0]))
                patch = image[rand_y:rand_y+min(self.target_size[0], img_h-rand_y), 
                              rand_x:rand_x+min(self.target_size[1], img_w-rand_x), :]
            else:
                # Select one random bounding box
                selected_bbox = random.choice(masks_bboxes)

                # Adjust bounding box (square, enlarge, clip)
                adj_x, adj_y, adj_w, adj_h = self._adjust_bbox(selected_bbox, image.shape)

                # Crop the patch
                patch = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]

            # Ensure patch is not empty after adjustments/cropping
            if patch.size == 0:
                print(f"Warning: Empty patch generated for {image_path}. Using random crop.")
                rand_x = random.randint(0, max(0, img_w - self.target_size[1]))
                rand_y = random.randint(0, max(0, img_h - self.target_size[0]))
                patch = image[rand_y:rand_y+min(self.target_size[0], img_h-rand_y), 
                              rand_x:rand_x+min(self.target_size[1], img_w-rand_x), :]

            # Apply transformations (including resize)
            if self.transform:
                patch_tensor = self.transform(patch)
            else:
                # Basic conversion if no transform provided
                patch_tensor = torch.from_numpy(patch.transpose((2, 0, 1))).float()  # C, H, W

            # Ensure patch is resized to target size
            if patch_tensor.shape[1:] != self.target_size:
                patch_tensor = F.interpolate(
                    patch_tensor.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)

            return patch_tensor, label

        except Exception as e:
            print(f"Error processing {image_path} at index {idx}: {e}")
            # Return a dummy item that will be filtered out by the collate function
            return None


def collate_fn_skip_none(batch):
    """Collate function that filters out None items."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])  # Handle empty batch case
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # Create output directories
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Transformations ---
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts (H, W, C) to (C, H, W) and scales to [0, 1]
        # Add data augmentation transforms if needed
    ])

    # --- Dataset and Splitting ---
    print("Loading dataset...")
    full_dataset = HyperspectralPatchDataset(
        args.data_dir,
        args.label_file,
        num_patches_per_image=args.num_patches_per_image,
        transform=transform,
        target_size=(args.patch_size, args.patch_size)
    )
    print(f"Full dataset size: {len(full_dataset)} patches")

    if not full_dataset.samples:
        raise ValueError("Dataset is empty. Check data paths and loading logic.")

    # Split dataset into training and testing sets
    total_size = len(full_dataset)
    train_size = int(total_size * args.train_split_ratio)
    test_size = total_size - train_size

    print(f"Splitting dataset: Train={train_size}, Test={test_size}")
    # Ensure sizes are valid before splitting
    if train_size <= 0 or test_size <= 0:
        raise ValueError(f"Invalid train/test split sizes: Train={train_size}, Test={test_size}. Need more data.")

    train_dataset, _ = random_split(full_dataset, [train_size, test_size])

    # --- DataLoader ---
    print("Creating DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.n_batch,
        shuffle=True,
        collate_fn=collate_fn_skip_none,
        num_workers=4,  # Adjust based on system capabilities
        pin_memory=True if args.device == 'cuda' else False
    )
    print("DataLoader created.")

    # --- Model ---
    print(f"Initializing model with {args.num_channels} input channels on {args.device}...")
    
    # Get number of classes from dataset
    class_labels = set()
    for _, label in full_dataset.samples:
        class_labels.add(label)
    num_classes = len(class_labels)
    
    # Initialize model
    model = ClassificationModel(
        encoder_type=args.encoder_type,
        input_channels=args.num_channels,
        num_classes=num_classes,
        device=args.device
    )
    model.to(args.device)
    print("Model initialized.")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Train ---
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