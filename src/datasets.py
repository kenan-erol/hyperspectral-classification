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
from sam2.build_sam import build_sam2, _load_checkpoint
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import hydra # Add hydra import
import hydra.utils
from omegaconf import OmegaConf, DictConfig # Add OmegaConf import

MIN_PILL_AREA = 25
MAX_PILL_AREA = 35*35
MIN_IOU_SCORE = 0.5

class HyperspectralPatchDataset(Dataset):
    def __init__(self,
                 data_dir,
                 samples,
                 sam2_checkpoint_path,
                 sam2_model_config: DictConfig, # Keep original name based on previous discussion
                 device,
                 num_patches_per_image=5,
                 # --- Accept transform parameters ---
                 transform_mean=None, # e.g., [0.5, 0.5, ...]
                 transform_std=None,  # e.g., [0.5, 0.5, ...]
                 num_channels=0,      # Needed for mean/std list creation
                 # --- End transform parameters ---
                 target_size=(224, 224)):
        self.data_dir = data_dir
        self.original_samples = samples
        self.samples_for_iteration = self._expand_samples(samples, num_patches_per_image)
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_model_config = sam2_model_config # Store the config object
        self.device = device # Store device string (e.g., 'cuda')
        self._worker_sam2_model = None
        self.num_patches_per_image = num_patches_per_image
        # Store transform parameters
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.num_channels = num_channels
        self.target_size = target_size
        print(f"Dataset initialized. Total samples for iteration: {len(self.samples_for_iteration)}")



    def _expand_samples(self, samples, num_patches_per_image):
        """Repeats each sample (image_path, label) num_patches_per_image times."""
        expanded_samples = []
        for sample in samples:
            expanded_samples.extend([sample] * num_patches_per_image)
        return expanded_samples
    
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

    def _initialize_worker_sam2(self):
        """Initializes SAM2 model within the worker process."""
        worker_pid = os.getpid() # Get worker PID for logging
        print(f"Initializing SAM2 in worker {worker_pid}...")
        try:
            # --- Use the specific model config ---
            # No need to check for 'model' key here, as we assume sam2_model_config IS the model part
            print(f"Instantiating model defined by: {self.sam2_model_config.get('_target_', 'N/A')}")
            sam2_model = hydra.utils.instantiate(self.sam2_model_config) # Use the passed model config directly
            # --- End config change ---

            if self.sam2_checkpoint_path:
                print(f"Loading checkpoint into model: {self.sam2_checkpoint_path}")
                _load_checkpoint(sam2_model, self.sam2_checkpoint_path)
            else:
                print("Warning: No SAM2 checkpoint path provided for worker initialization.")

            sam2_model.to(self.device)
            sam2_model.eval()
            print("Model instantiated and checkpoint loaded.")

            self._worker_sam2_model = SAM2AutomaticMaskGenerator(sam2_model)
            print(f"SAM2 initialized successfully in worker {worker_pid} on device {self.device}.")

        except Exception as e:
             print(f"!!! ERROR initializing SAM2 in worker {worker_pid}: {e}")
             import traceback
             traceback.print_exc()
             raise e



    def __getitem__(self, idx):
        print(f"Processing index {idx} in worker {os.getpid()}...")
        # Ensure worker-specific model is initialized
        if self._worker_sam2_model is None:
            try:
                self._initialize_worker_sam2()
            except Exception as init_e:
                print(f"Worker {os.getpid()} failed SAM2 init in __getitem__ for index {idx}: {init_e}")
                return None, None, None # Return None for all expected values

        # --- Reconstruct Transform ---
        transform = None
        if self.transform_mean is not None and self.transform_std is not None and self.num_channels > 0:
             mean = self.transform_mean if len(self.transform_mean) == self.num_channels else [self.transform_mean[0]] * self.num_channels
             std = self.transform_std if len(self.transform_std) == self.num_channels else [self.transform_std[0]] * self.num_channels
             transform = transforms.Compose([
                 transforms.Normalize(mean=mean, std=std),
             ])

        relative_path, label = self.samples_for_iteration[idx]
        full_image_path = os.path.join(self.data_dir, relative_path)

        try:
            image = np.load(full_image_path)
            print(f"Loaded image {full_image_path} in worker {os.getpid()}.")
            if image.ndim == 2: image = np.expand_dims(image, axis=-1)
            if image.dtype == np.float64: image = image.astype(np.float32)
            img_h, img_w, img_c = image.shape

            rgb_image_for_sam = np.mean(image, axis=2)
            min_val, max_val = np.min(rgb_image_for_sam), np.max(rgb_image_for_sam)
            if max_val > min_val:
                rgb_image_for_sam = ((rgb_image_for_sam - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                rgb_image_for_sam = np.zeros((img_h, img_w), dtype=np.uint8)
            rgb_image_for_sam = np.stack([rgb_image_for_sam]*3, axis=-1)

            masks_data = self._worker_sam2_model.generate(rgb_image_for_sam)
            
            print(f"Generated masks for {full_image_path} in worker {os.getpid()}.")

            if not masks_data:
                # print(f"Warning: No masks found for {full_image_path} in worker {os.getpid()}.") # Less verbose
                return None, None, None

            # --- Filter and Select Mask ---
            # 1. Filter by minimum area
            valid_masks = [m for m in masks_data if m['area'] >= MIN_PILL_AREA]

            # Optional: Add more filters (e.g., max area, score threshold)
            valid_masks = [m for m in valid_masks if m['area'] <= MAX_PILL_AREA]
            valid_masks = [m for m in valid_masks if m['predicted_iou'] >= MIN_IOU_SCORE]
            
            print(f"Filtered masks for {full_image_path} in worker {os.getpid()}. Found {len(valid_masks)} valid masks.")

            if not valid_masks:
                # print(f"Warning: No masks passed filtering for {full_image_path} in worker {os.getpid()}.") # Less verbose
                return None, None, None

            # 2. Select the largest remaining mask
            mask_info = max(valid_masks, key=lambda x: x['area'])
            # --- End Filter and Select ---

            bbox = mask_info['bbox'] # [x, y, w, h]

            # --- Convert bbox coords to INT and Crop ---
            # (Keep the existing cropping logic using integer bbox)
            x, y, w, h = map(int, bbox)
            adj_x = max(0, x)
            adj_y = max(0, y)
            adj_w = min(w, img_w - adj_x)
            adj_h = min(h, img_h - adj_y)

            if adj_w <= 0 or adj_h <= 0:
                # print(f"Warning: Invalid bbox dimensions after clipping for {full_image_path}.") # Less verbose
                return None, None, None

            patch_np = image[adj_y : adj_y + adj_h, adj_x : adj_x + adj_w, :]

            if patch_np.size == 0:
                # print(f"Warning: Empty patch after cropping for {full_image_path}.") # Less verbose
                return None, None, None
            # --- End Cropping ---

            # --- Data Type Conversion, Resizing, Transform, CPU move ---
            patch_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float() # CHW
            tensor_device = torch.device(self.device)
            patch_tensor = patch_tensor.to(tensor_device)
            
            if patch_tensor.shape[1:] != self.target_size:
                patch_tensor = F.interpolate(
                    patch_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            if transform:
                 try:
                     patch_tensor = transform(patch_tensor)
                 except TypeError as te:
                     print(f"Transform TypeError in worker {os.getpid()} for index {idx}: {te}. Input tensor shape: {patch_tensor.shape}, dtype: {patch_tensor.dtype}, device: {patch_tensor.device}")
                     raise te # Re-raise

            patch_tensor = patch_tensor.cpu()
            # --- End ---
            
            print(f"Processed patch for {full_image_path} in worker {os.getpid()}.")

            # Return tensor, label, and the original chosen bbox (can still be float)
            return patch_tensor, label, mask_info['bbox'] # Return original bbox for visualization

        except Exception as e:
            print(f"Error processing {full_image_path} at index {idx} in worker {os.getpid()}: {e}")
            # import traceback # Uncomment for debugging
            # traceback.print_exc() # Uncomment for debugging
            return None, None, None # Return None for all


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)