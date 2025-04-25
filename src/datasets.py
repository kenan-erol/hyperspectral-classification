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

class HyperspectralPatchDataset(Dataset):
    def __init__(self,
                 data_dir,
                 samples,
                 # Remove sam2_model parameter
                 sam2_checkpoint_path, # Add path
                 sam2_config_name: DictConfig,     # Add config name
                 device,               # Add device string
                 num_patches_per_image=5,
                 transform=None,
                 target_size=(224, 224)):
        self.data_dir = data_dir
        self.samples_for_iteration = self._expand_samples(samples, num_patches_per_image)
        # Store config details, not the model instance
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_config_name = sam2_config_name
        self.device = device # Store device string
        # --- Worker-specific model placeholder ---
        self._worker_sam2_model = None
        # -----------------------------------------
        self.num_patches_per_image = num_patches_per_image
        self.transform = transform
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
            # --- Use the passed config object ---
            # Check if the necessary 'model' key exists in the config object
            if 'model' not in self.sam2_config_name: # Check for 'model' key
                 # Update error message for clarity
                 raise KeyError("Key 'model' not found in the provided sam2_config_name object.")

            # # Build SAM2 model using the relevant part of the config (.model)
            # # Pass device string here
            # sam2 = build_sam2(self.sam2_config_name.model, checkpoint_path=self.sam2_checkpoint_path, device=self.device) # Use .model attribute
            # # --- build_sam2 already moves model to device and sets eval mode ---

            # # Create the generator
            # self._worker_sam2_model = SAM2AutomaticMaskGenerator(sam2)
            # print(f"SAM2 initialized successfully in worker {worker_pid} on device {self.device}.")
        # --- Directly instantiate the model using hydra.utils ---
            print(f"Instantiating model defined by: {self.sam2_config_name.model.get('_target_', 'N/A')}")
            # Instantiate the base model structure from the config part
            sam2_model = hydra.utils.instantiate(self.sam2_config_name.model)

            # Load the checkpoint using the internal helper function
            if self.sam2_checkpoint_path:
                print(f"Loading checkpoint into model: {self.sam2_checkpoint_path}")
                _load_checkpoint(sam2_model, self.sam2_checkpoint_path)
            else:
                print("Warning: No SAM2 checkpoint path provided for worker initialization.")

            # Move model to device and set to eval mode
            sam2_model.to(self.device)
            sam2_model.eval()
            print("Model instantiated and checkpoint loaded.")
            # --- End direct instantiation ---

            # Create the generator using the instantiated model
            self._worker_sam2_model = SAM2AutomaticMaskGenerator(sam2_model)
            print(f"SAM2 initialized successfully in worker {worker_pid} on device {self.device}.")

        except Exception as e:
             print(f"!!! ERROR initializing SAM2 in worker {worker_pid}: {e}")
             import traceback # Uncomment for debugging
             traceback.print_exc() # Uncomment for debugging
             raise e # Re-raise to make the failure clear in the main process



    def __getitem__(self, idx):
        # --- Initialize model if needed for this worker ---
        if self._worker_sam2_model is None:
            self._initialize_worker_sam2()
        # -------------------------------------------------

        if idx >= len(self.samples_for_iteration):
             # This case should ideally not be hit if __len__ is correct
             print(f"Warning: Index {idx} out of bounds for dataset size {len(self.samples_for_iteration)}")
             return None

        relative_path, label = self.samples_for_iteration[idx]
        full_image_path = os.path.join(self.data_dir, relative_path)

        try:
            # Load the full hyperspectral image
            image = np.load(full_image_path)
            # ... (handle ndim, dtype as before) ...
            img_h, img_w, img_c = image.shape

            # --- Convert hyperspectral to RGB for SAM2 ---
            # (Using mean method as example, ensure this matches preproc)

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
            # rgb_image = np.mean(image_data, axis=2)
            # # Normalize to 0-255 for SAM2 if needed (check SAM2 input requirements)
            # rgb_image = ((rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image)) * 255).astype(np.uint8)
            # rgb_image = np.stack([rgb_image]*3, axis=-1) # Convert grayscale to 3-channel RGB
            # --- End RGB Conversion ---
            
            rgb_image_for_sam = np.mean(image, axis=2)
            min_val, max_val = np.min(rgb_image_for_sam), np.max(rgb_image_for_sam)
            if max_val > min_val:
                rgb_image_for_sam = ((rgb_image_for_sam - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                rgb_image_for_sam = np.zeros((img_h, img_w), dtype=np.uint8)
            rgb_image_for_sam = np.stack([rgb_image_for_sam]*3, axis=-1)

            # Generate masks using the worker-specific model
            masks_data = self._worker_sam2_model.generate(rgb_image_for_sam)

            if not masks_data: return None # No masks found

            # Select a mask (e.g., randomly, largest) - using random for variety
            mask_info = random.choice(masks_data)
            bbox = mask_info['bbox'] # [x, y, w, h]

            # Adjust bbox and crop from ORIGINAL hyperspectral image
            adj_x = max(0, bbox[0])
            adj_y = max(0, bbox[1])
            adj_w = min(bbox[2], img_w - adj_x)
            adj_h = min(bbox[3], img_h - adj_y)

            if adj_w <= 0 or adj_h <= 0: return None # Invalid bbox

            patch_np = image[adj_y:adj_y+adj_h, adj_x:adj_x+adj_w, :] # HWC NumPy

            if patch_np.size == 0: return None

            # --- Data Type Conversion and Resizing ---
            # Convert to Tensor (CHW float) BEFORE resizing
            patch_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float()

            # Resize Tensor
            if patch_tensor.shape[1:] != self.target_size:
                # Ensure tensor is on the correct device for F.interpolate if using GPU in worker
                patch_tensor = patch_tensor.to(self.device)
                patch_tensor = F.interpolate(
                    patch_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                # Move back to CPU if transforms expect CPU tensor
                # patch_tensor = patch_tensor.cpu() # Optional: depends on transform

            # --- Apply Transforms (Assuming they expect a Tensor CHW) ---
            if self.transform:
                 # Ensure tensor is on CPU if transforms expect CPU tensor
                 # patch_tensor = patch_tensor.cpu()
                 try:
                     patch_tensor = self.transform(patch_tensor)
                 except TypeError as te:
                     # Catch the specific error and provide more context
                     print(f"Transform TypeError in worker {os.getpid()} for index {idx}: {te}. Input tensor shape: {patch_tensor.shape}, dtype: {patch_tensor.dtype}, device: {patch_tensor.device}")
                     raise te # Re-raise after printing

			# Ensure final tensor is on the correct device for collation
            patch_tensor = patch_tensor.to(self.device)

            return patch_tensor, label

        except Exception as e:
            print(f"Error processing {full_image_path} at index {idx} in worker {os.getpid()}: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for full traceback in worker
            return None


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)