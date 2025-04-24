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
            # Load the full hyperspectral image (assuming HWC format after potential expansion)
            image_data = np.load(full_path)
            if image_data.ndim == 2:
                image_data = np.expand_dims(image_data, axis=-1) # Ensure 3 dims (H, W, C)

            img_h, img_w, img_c = image_data.shape

            # --- Convert hyperspectral to RGB for SAM2 ---
            # Option 1: Select specific bands (e.g., Red, Green, Blue indices if known)
            # Example: Assuming R=50, G=30, B=10
            # if img_c >= 50: # Ensure enough channels
            #     rgb_image = image_data[:, :, [50, 30, 10]].astype(np.uint8)
            # else: # Fallback if not enough channels
            #     print(f"Warning: Not enough channels in {full_path} for RGB conversion. Using grayscale.")
            #     rgb_image = np.mean(image_data, axis=2).astype(np.uint8)
            #     rgb_image = np.stack([rgb_image]*3, axis=-1) # Convert grayscale to 3-channel RGB

            # Option 2: Use mean across channels (simple grayscale representation)
            rgb_image = np.mean(image_data, axis=2)
            # Normalize to 0-255 for SAM2 if needed (check SAM2 input requirements)
            rgb_image = ((rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image)) * 255).astype(np.uint8)
            rgb_image = np.stack([rgb_image]*3, axis=-1) # Convert grayscale to 3-channel RGB
            # --- End RGB Conversion ---


            # --- Fix SAM2 call and result processing ---
            # Use generate() method which expects an RGB image (HWC, uint8)
            mask_results = self.sam2.generate(rgb_image) # Use generate()

            if not mask_results:
                print(f"Warning: No masks found for {full_path}. Using random crop.")
                rand_x = random.randint(0, max(0, img_w - self.target_size[1]))
                rand_y = random.randint(0, max(0, img_h - self.target_size[0]))
                # Crop from the original hyperspectral data
                patch = image_data[rand_y:rand_y+min(self.target_size[0], img_h-rand_y),
                              rand_x:rand_x+min(self.target_size[1], img_w-rand_x), :]
            else:
                # Select a random mask result (each is a dictionary)
                selected_mask_info = random.choice(mask_results)
                # Get the bounding box ('bbox' is in XYWH format)
                bbox_xywh = selected_mask_info['bbox']
                # Adjust the bounding box
                adj_x, adj_y, adj_w, adj_h = self._adjust_bbox(bbox_xywh, image_data.shape)
                # Crop from the original hyperspectral data
                patch = image_data[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :]
            # --- End Fix ---

            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0: # More robust empty check
                print(f"Warning: Empty patch generated for {full_path} after cropping. Using random crop.")
                rand_x = random.randint(0, max(0, img_w - self.target_size[1]))
                rand_y = random.randint(0, max(0, img_h - self.target_size[0]))
                patch = image_data[rand_y:rand_y+min(self.target_size[0], img_h-rand_y),
                              rand_x:rand_x+min(self.target_size[1], img_w-rand_x), :]
                # Handle case where even random crop might be empty if target_size > image size
                if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                     print(f"Error: Could not generate valid patch for {full_path}. Skipping.")
                     return None # Skip this sample

            # --- Transform and Resize ---
            # Ensure patch is C, H, W for PyTorch transforms/interpolation
            patch = patch.transpose((2, 0, 1)) # HWC to CHW
            patch_tensor = torch.from_numpy(patch).float()

            # Resize if necessary (using interpolate requires CHW format)
            if patch_tensor.shape[1:] != self.target_size:
                # Add batch dimension for interpolate, then remove
                patch_tensor = F.interpolate(
                    patch_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear', # Consider 'nearest' if bilinear interpolation is problematic for spectral data
                    align_corners=False
                ).squeeze(0)

            # Apply other transforms if they expect CHW tensor
            if self.transform:
                patch_tensor = self.transform(patch_tensor)
            # --- End Transform and Resize ---


            return patch_tensor, label

        except Exception as e:
            print(f"Error processing {full_path} at index {idx}: {e}")
            # Optionally re-raise or return None depending on desired behavior
            # raise e # Re-raise to stop execution
            return None # Skip this sample if errors occur


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)