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