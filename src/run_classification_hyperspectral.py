import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F # For padding if needed

# Assuming these modules exist and have the expected interfaces
from classification_model import ClassificationModel
from classification_cnn import train, evaluate
from sam2 import SAM2 # Assuming SAM2 has a 'generate_masks' method returning bboxes

# --- Configuration ---
# TODO: Determine the actual number of channels in your hyperspectral data
NUM_HYPERSPECTRAL_CHANNELS = 100 # Example: Replace with actual number
# TODO: Determine the desired input size for the CNN model
PATCH_SIZE = 224

class HyperspectralPatchDataset(Dataset):
    def __init__(self, data_dir, label_file, num_patches_per_image=5, transform=None, target_size=(PATCH_SIZE, PATCH_SIZE)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        self.sam2 = SAM2() # Initialize SAM2

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
            image = np.load(image_path) # Shape: (H, W, C) or similar
            if image.ndim == 2: # Handle grayscale case if necessary
                 image = np.expand_dims(image, axis=-1)

            # Ensure image has channel dimension if needed by SAM2 or transforms
            # Example: If image is (H, W), make it (H, W, 1)
            # if image.ndim == 2:
            #     image = image[..., None]

            img_h, img_w = image.shape[:2]

            # Generate masks using SAM2 (returns list of bounding boxes [x, y, w, h])
            # This might be computationally expensive here
            masks_bboxes = self.sam2.generate_masks(image) # Adapt based on actual SAM2 output

            if not masks_bboxes:
                # Handle case with no masks: maybe return a random crop?
                # For now, let's try getting another item (might be inefficient)
                print(f"Warning: No masks found for {image_path}. Trying random crop.")
                # Fallback: random crop (adjust size as needed)
                rand_x = random.randint(0, img_w - self.target_size[1])
                rand_y = random.randint(0, img_h - self.target_size[0])
                patch = image[rand_y:rand_y+self.target_size[0], rand_x:rand_x+self.target_size[1], :]
            else:
                # Select one random bounding box
                selected_bbox = random.choice(masks_bboxes)

                # Adjust bounding box (square, enlarge, clip)
                adj_x, adj_y, adj_w, adj_h = self._adjust_bbox(selected_bbox, image.shape)

                # Crop the patch
                patch = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]

            # Ensure patch is not empty after adjustments/cropping
            if patch.size == 0:
                 print(f"Warning: Empty patch generated for {image_path}. Trying random crop.")
                 rand_x = random.randint(0, img_w - self.target_size[1])
                 rand_y = random.randint(0, img_h - self.target_size[0])
                 patch = image[rand_y:rand_y+self.target_size[0], rand_x:rand_x+self.target_size[1], :]


            # Apply transformations (including resize)
            if self.transform:
                # Transform might expect (C, H, W) or (H, W, C).
                # ToTensor typically converts (H, W, C) numpy to (C, H, W) tensor.
                patch_tensor = self.transform(patch)
            else:
                # Basic conversion if no transform provided (adjust as needed)
                patch_tensor = torch.from_numpy(patch.transpose((2, 0, 1))).float() # C, H, W

            # Ensure patch is resized to target size (if not done in transforms)
            # This might be needed if cropping results in variable sizes
            # Using interpolate for general resizing of tensors
            if patch_tensor.shape[1:] != self.target_size:
                 patch_tensor = F.interpolate(patch_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)


            return patch_tensor, label

        except Exception as e:
            print(f"Error processing {image_path} at index {idx}: {e}")
            # Return a dummy item or None, requires careful handling in collate_fn
            # Returning None is simpler if collate_fn filters Nones
            return None

def collate_fn_skip_none(batch):
    """Collate function that filters out None items."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([]) # Handle empty batch case
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # --- Configuration ---
    data_dir = './data/drop-4/' # Adjust path as needed
    label_file = './data/labels.txt' # Adjust path as needed
    checkpoint_path = './checkpoints/hyperspectral/'
    output_path = './outputs/hyperspectral/'
    n_batch = 25 # Batch size 20-30
    n_epoch = 30
    learning_rate = 0.001
    learning_rate_decay = 0.5
    learning_rate_period = 10
    num_patches_per_image = 5
    train_split_ratio = 0.8 # 80% for training, 20% for testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # --- Transformations ---
    # IMPORTANT:
    # 1. Resize is crucial for batching if patches have variable sizes after cropping.
    # 2. ToTensor converts numpy HWC to tensor CHW and scales to [0, 1].
    # 3. Normalization needs per-channel mean/std for your specific dataset.
    #    Calculate these values beforehand or omit normalization initially.
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts (H, W, C) np.uint8 -> (C, H, W) torch.float32 [0,1]
        # transforms.Resize(PATCH_SIZE, antialias=True), # Resize added here if not handled later
        # Add other augmentations if needed (e.g., RandomHorizontalFlip)
        # transforms.Normalize(mean=[...], std=[...]) # Add correct normalization values
    ])

    # --- Dataset and Splitting ---
    print("Loading dataset...")
    full_dataset = HyperspectralPatchDataset(
        data_dir,
        label_file,
        num_patches_per_image=num_patches_per_image,
        transform=transform,
        target_size=(PATCH_SIZE, PATCH_SIZE)
    )
    print(f"Full dataset size: {len(full_dataset)} patches")

    if not full_dataset:
        raise ValueError("Dataset is empty. Check data paths and loading logic.")

    # Split dataset into training and testing sets
    total_size = len(full_dataset)
    train_size = int(total_size * train_split_ratio)
    test_size = total_size - train_size

    print(f"Splitting dataset: Train={train_size}, Test={test_size}")
    # Ensure sizes are valid before splitting
    if train_size <= 0 or test_size <= 0:
         raise ValueError(f"Invalid train/test split sizes: Train={train_size}, Test={test_size}. Need more data.")

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # --- DataLoaders ---
    print("Creating DataLoaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=n_batch,
        shuffle=True,
        collate_fn=collate_fn_skip_none, # Use collate function that handles None
        num_workers=4, # Adjust based on system capabilities
        pin_memory=True if device == 'cuda' else False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=n_batch,
        shuffle=False, # No need to shuffle test set
        collate_fn=collate_fn_skip_none,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    print("DataLoaders created.")

    # --- Model ---
    print(f"Initializing model with {NUM_HYPERSPECTRAL_CHANNELS} input channels on {device}...")
    # TODO: Ensure ClassificationModel and its encoder handle NUM_HYPERSPECTRAL_CHANNELS
    # You might need to modify the first layer of the encoder (e.g., ResNet)
    model = ClassificationModel(
        encoder_type='resnet18', # Or another suitable encoder
        input_channels=NUM_HYPERSPECTRAL_CHANNELS,
        num_classes=len(set(full_dataset.dataset.samples[i][1] for i in full_dataset.indices)), # Get num classes from dataset
        device=device
    )
    model.to(device)
    print("Model initialized.")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Train ---
    print("Starting training...")
    train(model, train_dataloader, n_epoch, optimizer, learning_rate_decay, learning_rate_period, checkpoint_path, device)
    print("Training finished.")

    # --- Evaluate ---
    print("Starting evaluation on the test set...")
    # Load the best checkpoint for evaluation
    best_checkpoint_path = os.path.join(checkpoint_path, 'best_model.pth') # Assuming train saves 'best_model.pth'
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print(f"Loaded best model from {best_checkpoint_path}")
    else:
        print("Warning: No best model checkpoint found. Evaluating the final model.")

    # Assuming evaluate function returns metrics like accuracy
    # You might need to adapt the evaluate function to return metrics dict
    # Also, pass class names if available for detailed reports
    # class_names = [str(i) for i in range(model.num_classes)] # Example class names
    evaluation_results = evaluate(model, test_dataloader, output_path, device) # Removed dataset.classes assuming evaluate doesn't need it directly
    print("Evaluation finished.")
    print(f"Evaluation Results: {evaluation_results}") # Print the results returned by evaluate
