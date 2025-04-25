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

from datasets import HyperspectralPatchDataset, collate_fn_skip_none

# train (#train_classification_hyper.py:203-213): Instantiates HyperspectralPatchDataset passing sam2_checkpoint_path, sam2_config_name, and device. This allows the dataset's __getitem__ (#datasets.py:95-199) to initialize and use SAM2 within worker processes (#datasets.py:79-93) to generate patches on-the-fly.
# run (#run_classification_hyper.py:119-127): Instantiates the same HyperspectralPatchDataset class but does not pass the SAM2-related arguments (sam2_checkpoint_path, sam2_config_name).
# Sensibility: This reveals a major difference in assumptions and is highly suspicious.
# The train script assumes it needs to generate patches dynamically using SAM2 during data loading.
# The run script, by omitting the SAM2 arguments, implicitly assumes that the HyperspectralPatchDataset can function without them. However, looking at the __getitem__ implementation (#datasets.py:95-199), it requires self._worker_sam2_model (#datasets.py:143-144) which is only initialized if the SAM2 config details are provided to __init__ (#datasets.py:18-41).
# Conclusion: run_classification_hyper.py likely assumes it's operating on pre-generated patches (like those created by preproc_a2s2k.py), but it's using a Dataset class designed for on-the-fly generation. This mismatch will likely cause __getitem__ to fail consistently in the run script because self._worker_sam2_model will never be initialized.
# DataLoader Settings:

# train (#train_classification_hyper.py:218-225): shuffle=True, drop_last=True.
# run (#run_classification_hyper.py:132-138): shuffle=False. The excerpt cuts off, but drop_last should ideally be False for evaluation. Both use num_workers=4 and collate_fn_skip_none.
# Sensibility: Shuffling only training data makes sense. drop_last=True for training is acceptable, but drop_last=False is generally preferred for testing to evaluate all samples. Using collate_fn_skip_none in run might hide errors if __getitem__ fails due to the SAM2 issue mentioned above. num_workers=4 is acceptable but might be overkill for inference.
# Model:

# train: Initializes a new ClassificationModel.
# run: (Assumed) Loads model state from a checkpoint file specified via arguments.
# Sensibility: Expected difference between training and inference. make sure run can actually load the model state correctly.

# train: Defines a transform pipeline (currently includes Normalize).
# run: Also uses a transform variable (definition not shown in excerpt).
# Sensibility: Makes sense, but it is absolutely critical that the normalization (and any other non-augmentation transforms) applied during testing (run) are identical to those used during training (train). If the means/stds in transforms.Normalize differ, or if it's missing in one script, the evaluation results will be invalid.
# Suspicious Lines / Potential Issues:

# HyperspectralPatchDataset Usage in run_classification_hyper.py (#run_classification_hyper.py:119-127): This is the biggest issue. Using this dataset class without providing the SAM2 arguments it needs for its core __getitem__ logic (#datasets.py:95-199) is incorrect if it's meant to work like the training script. If run is meant for pre-generated patches, it should use a different, simpler Dataset class that just loads .npy files.
# Transform Consistency: The transform definition in run_classification_hyper.py (not shown) must match the one in train_classification_hyper.py (#train_classification_hyper.py:123-128), especially the Normalize parameters. Verify this.
# collate_fn_skip_none in run_classification_hyper.py (#run_classification_hyper.py:136): If the dataset issue (Point 1) causes __getitem__ to always return None, this collate function will result in empty batches being fed to the model during evaluation, likely causing a crash similar to the conv2d error seen during training attempts, or producing zero accuracy.
# drop_last in Test DataLoader (#run_classification_hyper.py:138): Ensure drop_last=False (or is omitted, as False is the default) for the test DataLoader to evaluate the entire test set.
# transforms.Normalize in train_classification_hyper.py (#train_classification_hyper.py:127): The current normalization mean=[0.5]*args.num_channels, std=[0.5]*args.num_channels assumes the input tensor is scaled to [0, 1] and shifts it to [-1, 1]. Is this the correct normalization strategy? Often, normalization uses the actual dataset mean/std per channel. Also, this transform expects a Tensor, which is now correct after previous fixes, but the TypeError seen before suggests this might have been problematic. Ensure the tensor entering this transform has the shape (num_channels, H, W).
# In summary, the most critical issue is the inconsistent use and assumptions of the HyperspectralPatchDataset between the two scripts regarding SAM2 patch generation. This needs to be reconciled.

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
