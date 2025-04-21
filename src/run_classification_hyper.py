import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset # Removed random_split as we use sklearn now
from torchvision import transforms
import torch.nn.functional as F # For padding if needed
from sklearn.model_selection import train_test_split # Import train_test_split

# Assuming these modules exist and have the expected interfaces
from classification_model import ClassificationModel
from classification_cnn import evaluate # Removed train
from sam2 import SAM2 # Assuming SAM2 has a 'generate_masks' method returning bboxes
import argparse # Add argparse for command-line arguments

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Evaluate Hyperspectral Classification Model")
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing hyperspectral data subfolders')
parser.add_argument('--label_file', type=str, required=True, help='File containing relative image paths and labels')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
parser.add_argument('--output_path', type=str, required=True, help='Directory to save evaluation outputs (e.g., confusion matrix plot)')
parser.add_argument('--encoder_type', type=str, required=True, help='Encoder type used during training (e.g., resnet18)')
parser.add_argument('--num_channels', type=int, required=True, help='Number of spectral bands (must match training)')
parser.add_argument('--n_batch', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--num_patches_per_image', type=int, default=5, help='Number of patches per image for evaluation (should ideally match training)')
parser.add_argument('--patch_size', type=int, default=224, help='Patch size used during training')
parser.add_argument('--train_split_ratio', type=float, default=0.8, help='Train split ratio used during training (to define the test set)')
parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
args = parser.parse_args()


# --- Dataset Class (Identical to train script) ---
class HyperspectralPatchDataset(Dataset):
    # Modified to accept a list of samples directly
    def __init__(self, data_dir, samples_list, num_patches_per_image=5, transform=None, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.num_patches_per_image = num_patches_per_image
        self.sam2 = SAM2()  # Initialize SAM2

        # Store the original image paths and labels provided
        self.original_samples = samples_list

        # Create multiple entries for patch sampling
        self.samples_for_iteration = []
        for image_path, label in self.original_samples:
            for _ in range(self.num_patches_per_image):
                # Store the *original* path, patch extraction happens in __getitem__
                self.samples_for_iteration.append((image_path, label))

    def __len__(self):
        # Length is based on the number of patches we want to generate
        return len(self.samples_for_iteration)

    # ... existing _adjust_bbox method ...
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
        # Get the original image path and label for this iteration
        original_image_path, label = self.samples_for_iteration[idx]
        full_path = os.path.join(self.data_dir, original_image_path) # Construct full path

        try:
            # Load the hyperspectral image (H, W, C) or (C, H, W)
            # Assuming np.load results in (H, W, C) for consistency
            image = np.load(full_path)  # Shape: (H, W, C) or similar
            if image.ndim == 2:  # Handle grayscale case if necessary
                image = np.expand_dims(image, axis=-1)

            # ... rest of the __getitem__ logic remains the same ...
            # ... (mask generation, bbox selection, cropping, transform, resize) ...
            img_h, img_w = image.shape[:2]

            # Generate masks using SAM2 (returns list of bounding boxes [x, y, w, h])
            masks_bboxes = self.sam2.generate_masks(image)  # Adapt based on actual SAM2 output

            if not masks_bboxes:
                # Handle case with no masks: use random crop as fallback
                print(f"Warning: No masks found for {full_path}. Using random crop.")
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
                print(f"Warning: Empty patch generated for {full_path}. Using random crop.")
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
            print(f"Error processing {full_path} at index {idx}: {e}")
            # Return a dummy item that will be filtered out by the collate function
            return None

# --- Collate Function (Identical to train script) ---
def collate_fn_skip_none(batch):
    """Collate function that filters out None items."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([]) # Handle empty batch case
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # --- Configuration ---
    data_dir = args.data_dir
    label_file = args.label_file
    checkpoint_load_path = args.checkpoint_path # Path to the specific .pth file
    output_path = args.output_path
    n_batch = args.n_batch
    num_patches_per_image = args.num_patches_per_image
    patch_size = args.patch_size
    train_split_ratio = args.train_split_ratio
    device = args.device
    num_channels = args.num_channels
    encoder_type = args.encoder_type

    os.makedirs(output_path, exist_ok=True)

    # --- Transformations ---
    # Use the same basic transform as training, without augmentation
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts (H, W, C) np.uint8 -> (C, H, W) torch.float32 [0,1]
        # transforms.Resize(patch_size, antialias=True), # Ensure resize if needed
        # transforms.Normalize(mean=[...], std=[...]) # Use same normalization as training if applied
    ])

    # --- Load All Samples and Split Based on Unique Images ---
    print("Loading image list and labels...")
    all_samples = []
    class_labels = set()
    class_map = {} # To map integer labels back to names if needed
    label_counter = 0
    try:
        with open(label_file, 'r') as f:
            for line in f:
                relative_path, label_str = line.strip().split()
                label = int(label_str)
                # Check if the actual file exists relative to data_dir
                full_path_check = os.path.join(data_dir, relative_path)
                if os.path.exists(full_path_check):
                    all_samples.append((relative_path, label))
                    class_labels.add(label)
                    # Optional: try to infer class name from path
                    try:
                        # Example: Assumes path like 'DrugName_Date/Group/M*/file.npy'
                        drug_name = relative_path.split(os.sep)[0].split('_')[0]
                        if label not in class_map:
                            class_map[label] = drug_name
                    except:
                        if label not in class_map: # Fallback to numeric label
                           class_map[label] = str(label)

                else:
                    print(f"Warning: Image file not found {full_path_check}, skipping.")
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading label file {label_file}: {e}")
        exit(1)


    if not all_samples:
        raise ValueError("No valid samples found. Check label file and data directory.")

    num_classes = len(class_labels)
    # Create sorted class names list for evaluation function
    class_names = [class_map[i] for i in sorted(class_labels)]
    print(f"Found {len(all_samples)} unique images belonging to {num_classes} classes: {class_names}")


    # Split the list of unique (image_path, label) tuples
    image_paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    # Use train_test_split to get the *test* set consistent with training split
    _, test_paths, _, test_labels = train_test_split(
        image_paths, labels,
        train_size=train_split_ratio, # Specify train size to get the remainder as test
        random_state=42,  # Use the same random state as training
        stratify=labels # Ensure class distribution is similar
    )

    test_samples = list(zip(test_paths, test_labels))

    print(f"Using Test set with {len(test_samples)} unique images")

    # --- Create Test Dataset ---
    print("Creating test dataset instance...")
    test_dataset = HyperspectralPatchDataset(
        data_dir,
        test_samples, # Pass the list of test samples
        num_patches_per_image=num_patches_per_image,
        transform=transform,
        target_size=(patch_size, patch_size)
    )
    print(f"Test dataset size: {len(test_dataset)} patches")


    # --- DataLoaders ---
    print("Creating Test DataLoader...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=n_batch,
        shuffle=False, # No need to shuffle test set
        collate_fn=collate_fn_skip_none,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    print("Test DataLoader created.")

    # --- Model ---
    print(f"Initializing model with {num_channels} input channels and {num_classes} classes on {device}...")
    model = ClassificationModel(
        encoder_type=encoder_type,
        input_channels=num_channels,
        num_classes=num_classes, # Use dynamically determined number of classes
        device=device
    )
    model.to(device)
    print("Model initialized.")

    # --- Load Checkpoint ---
    print(f"Loading model checkpoint from {checkpoint_load_path}...")
    try:
        # Use restore_model which handles DataParallel wrappers if necessary
        step, _ = model.restore_model(checkpoint_load_path)
        print(f"Loaded model weights from step {step}.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_load_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    # --- Evaluate ---
    print("Starting evaluation on the test set...")
    model.eval() # Set model to evaluation mode

    # Pass class names list to evaluate function
    evaluation_results = evaluate(model, test_dataloader, class_names, output_path, device)
    print("Evaluation finished.")
    # Assuming evaluate prints results internally or returns them
    # print(f"Evaluation Results: {evaluation_results}") # Uncomment if evaluate returns results
