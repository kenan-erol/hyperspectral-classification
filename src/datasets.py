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

from log_utils import hsi_to_rgb_display
import matplotlib.pyplot as plt

import statistics
import copy

plt.use('Agg') # disable interactivity for train

MIN_PILL_AREA = 50
MAX_PILL_AREA = 170
MIN_IOU_SCORE = 0.5

TARGET_MASK_COUNT = 100
MIN_MASKS_FOR_ADJUSTMENT = 51 # More than 50

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

            self._worker_sam2_model = SAM2AutomaticMaskGenerator(
				model=sam2_model,
				points_per_side=64,
				points_per_batch=64,
				pred_iou_thresh=0.9,
				stability_score_thresh=0.92, # >0.92 for less squarish masks
				stability_score_offset=0.5,
				box_nms_thresh=0.55,
				crop_n_layers=1,
				crop_nms_thresh = 0.7,
					crop_overlap_ratio = 512 / 1500,
					crop_n_points_downscale_factor = 2,
					# point_grids: Optional[List[np.ndarray]] = None,
					# min_mask_region_area = 15.0,
					output_mode = "binary_mask",
					multimask_output = True,
				min_mask_region_area=25.0,
				use_m2m=True,
			) # rn it is too edgy
            print(f"SAM2 initialized successfully in worker {worker_pid} on device {self.device}.")

        except Exception as e:
             print(f"!!! ERROR initializing SAM2 in worker {worker_pid}: {e}")
             import traceback
             traceback.print_exc()
             raise e



    def __getitem__(self, idx):
        # print(f"Processing index {idx} in worker {os.getpid()}...")
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
            # print(f"Loaded image {full_image_path} in worker {os.getpid()}.")
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
            
            # print(f"Generated masks for {full_image_path} in worker {os.getpid()}.")

            if not masks_data:
                # print(f"Warning: No masks found for {full_image_path} in worker {os.getpid()}.") # Less verbose
                return None, None, None

            # --- Filter and Select Mask ---
            valid_masks = [
                m for m in masks_data
                if MIN_PILL_AREA <= m['area'] <= MAX_PILL_AREA and m['predicted_iou'] >= MIN_IOU_SCORE
            ]

            if not valid_masks:
                # print(f"Warning: No masks passed initial filtering for {full_image_path}.") # Less verbose
                return None, None, None
            
            print(f"Filtered masks for {full_image_path} in worker {os.getpid()}. Found {len(valid_masks)} valid masks before selecting.")

            selection_pool = valid_masks # Default pool

            if len(valid_masks) >= MIN_MASKS_FOR_ADJUSTMENT:
                # Calculate median area
                areas = [m['area'] for m in valid_masks]
                # Ensure areas list is not empty before calculating median
                if not areas:
                     median_area = 0 # Or handle as an error/warning
                else:
                     median_area = statistics.median(areas)

                # Calculate difference from median and sort
                # Store tuples (difference, mask_dict)
                masks_with_diff = [
                    (abs(m['area'] - median_area), m) for m in valid_masks
                ]
                # Sort by difference (ascending - closest first)
                masks_with_diff.sort(key=lambda x: x[0])
                sorted_masks_by_closeness = [m[1] for m in masks_with_diff] # Get back list of dicts

                if len(valid_masks) > TARGET_MASK_COUNT:
                    # Too many masks: Keep the 100 closest to the median area
                    selection_pool = sorted_masks_by_closeness[:TARGET_MASK_COUNT]
                    # print(f"DEBUG: Reduced pool from {len(valid_masks)} to {len(selection_pool)} by keeping closest to median area {median_area}.") # Optional Debug

                else: # 50 < len(valid_masks) <= 100
                    # Too few masks (but > 50): Duplicate masks closest to median until we have 100
                    num_to_add = TARGET_MASK_COUNT - len(valid_masks)
                    # Get the masks to duplicate (those closest to median)
                    masks_to_duplicate = sorted_masks_by_closeness[:num_to_add]
                    # Create deep copies to avoid modifying original list elements if needed later
                    duplicates = [copy.deepcopy(m) for m in masks_to_duplicate]
                    selection_pool = valid_masks + duplicates # Combine original list with duplicates
                    # print(f"DEBUG: Increased pool from {len(valid_masks)} to {len(selection_pool)} by duplicating {num_to_add} masks closest to median area {median_area}.") # Optional Debug


            # 3. Select the largest mask from the final pool
            if not selection_pool: # Safety check in case pool somehow becomes empty
                 # print(f"Warning: Selection pool became empty for {full_image_path}.") # Less verbose
                 return None, None, None

            # 2. Select the largest remaining mask
            mask_info = max(valid_masks, key=lambda x: x['area'])
            
            print(f"Filtered masks for {full_image_path} in worker {os.getpid()}. Found {len(valid_masks)} valid masks after selecting.")
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
            
            # print(f"Processed patch for {full_image_path} in worker {os.getpid()}.")

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

class PreprocessedPatchDataset(Dataset):
    """
    Dataset to load pre-generated hyperspectral patches (.npy files).
    Assumes a label file where each line is: relative/path/to/patch.npy label
    The relative path should be relative to the data_dir provided.
    """
    def __init__(self,
                 data_dir,      # Base directory where patches are saved (e.g., './data_processed_patch/')
                 samples,       # List of (relative_patch_path, label) tuples
                 num_channels=0, # Needed for transform
                 save_visualization_path=None, # e.g., 'hyper_checkpoints/resnet/visualizations/'
                 num_visualizations_to_save=0,   # How many initial samples to visualize
                 transform_mean=None,
                 transform_std=None,
                 target_size=(224, 224)): # Ensure patches are resized if needed
        self.data_dir = data_dir
        self.samples = samples # Use the provided list directly
        self.num_channels = num_channels
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.target_size = target_size
        self.transform = self._create_transform()
        
        # --- Store Visualization Params ---
        self.save_visualization_path = save_visualization_path
        self.num_visualizations_to_save = num_visualizations_to_save
        if self.save_visualization_path:
            os.makedirs(self.save_visualization_path, exist_ok=True) # Create dir if needed
        # --- End Store Visualization Params ---
        
        if not self.samples:
             print("Warning: PreprocessedPatchDataset initialized with zero samples.")
        else:
             print(f"PreprocessedPatchDataset initialized. Using {len(self.samples)} provided samples.")
             # Optional: Check if the first sample file exists
             first_rel_path, _ = self.samples[0]
             first_full_path = os.path.join(self.data_dir, first_rel_path)
             if not os.path.exists(first_full_path):
                 print(f"Warning: First sample file not found at {first_full_path}. Check data_dir and relative paths.")


    def _create_transform(self):
        """Creates the normalization transform if parameters are provided."""
        if self.transform_mean is not None and self.transform_std is not None and self.num_channels > 0:
            mean = self.transform_mean if len(self.transform_mean) == self.num_channels else [self.transform_mean[0]] * self.num_channels
            std = self.transform_std if len(self.transform_std) == self.num_channels else [self.transform_std[0]] * self.num_channels
            return transforms.Compose([
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                RandomIntensityScale(min_factor=0.8, max_factor=1.2)
            ])
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
             print(f"Error: Index {idx} out of bounds for dataset size {len(self.samples)}")
             return None, None # Or raise IndexError

        relative_path, label = self.samples[idx]
        full_patch_path = os.path.join(self.data_dir, relative_path)

        try:
            # Load the .npy patch file
            patch_np = np.load(full_patch_path) # Should be HWC, float32

            # Ensure correct format (HWC -> CHW tensor)
            if patch_np.ndim == 2: # Handle potential grayscale if needed
                patch_np = np.expand_dims(patch_np, axis=-1)
            # patch_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float() # CHW
            
            original_tensor = torch.from_numpy(patch_np.transpose((2, 0, 1))).float() # CHW

            # --- Prepare Original for Visualization (BEFORE transform) ---
            original_rgb_for_viz = None
            save_this_sample = self.save_visualization_path and idx < self.num_visualizations_to_save and hsi_to_rgb_display is not None
            if save_this_sample:
                try:
                    # Resize original tensor if needed for consistent viz size
                    if original_tensor.shape[1:] != self.target_size:
                        original_tensor_resized_viz = F.interpolate(
                            original_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
                        ).squeeze(0)
                    else:
                        original_tensor_resized_viz = original_tensor
                    # Convert resized original tensor back to HWC numpy for display
                    original_np_hwc_viz = np.transpose(original_tensor_resized_viz.numpy(), (1, 2, 0))
                    original_rgb_for_viz = hsi_to_rgb_display(original_np_hwc_viz)
                except Exception as viz_e:
                    print(f"Warning: Failed to create original RGB for visualization (idx {idx}): {viz_e}")
                    save_this_sample = False # Don't save if original fails
            # --- End Prepare Original ---

            # Resize tensor before applying transforms if needed
            if original_tensor.shape[1:] != self.target_size:
                 patch_tensor = F.interpolate(
                     original_tensor.unsqueeze(0),
                     size=self.target_size,
                     mode='bilinear',
                     align_corners=False
                 ).squeeze(0)
            else:
                 patch_tensor = original_tensor # Use original if already correct size

            # Apply normalization and augmentation transforms
            transformed_tensor = patch_tensor # Start with the (potentially resized) tensor
            if self.transform:
                transformed_tensor = self.transform(transformed_tensor.clone()) # Use clone if transforms modify inplace

            # --- Save Visualization (if requested and original viz worked) ---
            if save_this_sample:
                try:
                    # Convert transformed tensor back to HWC numpy for display
                    transformed_np_hwc_viz = np.transpose(transformed_tensor.numpy(), (1, 2, 0))
                    transformed_rgb_for_viz = hsi_to_rgb_display(transformed_np_hwc_viz)

                    # Plot side-by-side
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(original_rgb_for_viz)
                    axes[0].set_title(f"Original (Resized) - Idx {idx}")
                    axes[0].axis('off')

                    axes[1].imshow(transformed_rgb_for_viz)
                    axes[1].set_title(f"After Transforms - Idx {idx}")
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    # Construct filename
                    base_filename = os.path.splitext(os.path.basename(relative_path))[0]
                    save_filename = f"viz_{idx}_{base_filename}.png"
                    full_save_path = os.path.join(self.save_visualization_path, save_filename)
                    plt.savefig(full_save_path)
                    plt.close(fig) # Close figure to free memory
                    # print(f"Saved visualization for index {idx} to {full_save_path}") # Optional log
                except Exception as viz_e:
                    print(f"Warning: Failed to save visualization for index {idx}: {viz_e}")
                    if 'fig' in locals() and plt.fignum_exists(fig.number): # Attempt to close figure on error
                         plt.close(fig)
            # --- End Save Visualization ---

            # Return the transformed tensor for training/evaluation
            return transformed_tensor, label

        except FileNotFoundError:
            print(f"Error: Patch file not found: {full_patch_path} for index {idx}")
            return None, None # Return None if file not found
        except Exception as e:
            print(f"Error loading or processing patch {full_patch_path} for index {idx}: {e}")
            # import traceback # Uncomment for debugging
            # traceback.print_exc() # Uncomment for debugging
            return None, None # Return None on other errors

def collate_fn_skip_none_preprocessed(batch):
    """Collate function that filters out items where the patch tensor is None."""
    # Filter out samples where the first element (the tensor) is None
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        # Return empty tensors or raise an error if the whole batch failed
        # Returning empty tensors might be safer for the training loop
        return torch.tensor([]), torch.tensor([]) # Match expected output structure (tensors, labels)
    # Use default collate for the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)

class RandomIntensityScale:
    def __init__(self, min_factor=0.8, max_factor=1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, tensor):
        factor = random.uniform(self.min_factor, self.max_factor)
        return tensor * factor