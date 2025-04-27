import os, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from classification_model import ClassificationModel
# --- Import the NEW dataset and collate function ---
from datasets import PreprocessedPatchDataset, collate_fn_skip_none_preprocessed
# --- End Import ---

# --- Remove SAM2/Hydra imports if they were present ---
# ...
# --- End Remove ---


# Define command-line arguments
parser = argparse.ArgumentParser()
# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--data_dir',
    type=str, required=True, help='Directory containing PREPROCESSED patches and labels.txt') # Updated help
parser.add_argument('--label_file',
    type=str, required=True, help='Path to the labels.txt file WITHIN the data_dir') # Updated help
# --- Remove arguments no longer needed ---
# parser.add_argument('--num_patches_per_image', ...)
# --- End Remove ---
parser.add_argument('--patch_size',
    type=int, default=224, help='Expected size of image patches')
parser.add_argument('--train_split_ratio', # Keep this to identify the TEST split
    type=float, default=0.8, help='Proportion of PATCHES used for training (to determine test set)')

# Network settings
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type used during training: vggnet11, resnet18')
parser.add_argument('--num_channels',
    type=int, required=True, help='Number of spectral bands in hyperspectral data')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path directory containing the trained model checkpoint (e.g., model_best.pth)')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()

def run(model, dataloader, device):
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad(): # Disable gradient calculations for inference
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # --- CHANGE THIS LINE ---
            # outputs = model(inputs) # Incorrect: ClassificationModel is not callable
            outputs = model.forward(inputs) # Correct: Explicitly call the forward method
            # --- END CHANGE ---
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


if __name__ == '__main__':
    # --- Setup Device ---
    if args.device == 'gpu' or args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    # --- End Device Setup ---

    # --- Load Preprocessed Labels ---
    print("Loading patch list and labels from preprocessed data...")
    all_patch_samples = []
    class_labels = set()
    full_label_file_path = os.path.join(args.label_file)

    try:
        with open(full_label_file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    relative_path, label_str = line.rsplit(' ', 1)
                    label = int(label_str)
                    all_patch_samples.append((relative_path, label))
                    class_labels.add(label)
                except ValueError as e:
                    print(f"Warning: Skipping malformed line {i+1} in {full_label_file_path}: '{line}' - Reason: {e}")
    except FileNotFoundError:
        print(f"Error: Preprocessed label file not found at {full_label_file_path}")
        exit(1)
    except Exception as e:
        print(f"Error reading preprocessed label file {full_label_file_path}: {e}")
        exit(1)

    if not all_patch_samples:
        raise ValueError("No valid patch samples found.")

    num_classes = len(class_labels)
    print(f"Found {len(all_patch_samples)} total patches belonging to {num_classes} classes.")
    # --- End Load Labels ---


    # --- Determine Test Split ---
    patch_paths = [s[0] for s in all_patch_samples]
    patch_labels = [s[1] for s in all_patch_samples]

    if args.train_split_ratio < 1.0:
         # Use train_test_split to get the *test* portion consistently
         _, test_paths, _, test_labels = train_test_split(
             patch_paths, patch_labels,
             train_size=args.train_split_ratio, # Specify train size
             random_state=42, # Use same random state as in training script
             stratify=patch_labels # Stratify based on patch labels
         )
         test_samples = list(zip(test_paths, test_labels))
         print(f"Using Test set with {len(test_samples)} patches")
    else:
         # If train ratio was 1.0, there's no separate test set from this split
         print("Warning: train_split_ratio is 1.0 or greater. No test set generated from this split.")
         print("Evaluation will run on the full dataset provided.")
         test_samples = all_patch_samples # Evaluate on all data as 'test'

    if not test_samples:
         print("Error: No samples available for the test set.")
         exit(1)
    # --- End Test Split ---


    # --- Create Test Dataset ---
    # Define transform (only normalization needed usually for evaluation)
    transform_mean = [0.5] * args.num_channels if args.num_channels > 0 else None
    transform_std = [0.5] * args.num_channels if args.num_channels > 0 else None

    print("Creating test dataset instance...")
    # --- Instantiate the NEW dataset ---
    test_dataset = PreprocessedPatchDataset(
        data_dir=args.data_dir,
        samples=test_samples, # Pass the list of test samples
        num_channels=args.num_channels,
        transform_mean=transform_mean,
        transform_std=transform_std,
        target_size=(args.patch_size, args.patch_size),
        save_visualization_path='hyper_checkpoints/resnet/transform_viz', # Optional, can be None
        num_visualizations_to_save=5
    )
    # --- End Instantiate ---
    print(f"Test dataset size: {len(test_dataset)} patches")


    # --- DataLoaders ---
    print("Creating Test DataLoader...")
    # --- Use the NEW collate_fn and potentially more workers ---
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.n_batch,
        shuffle=False, # No need to shuffle test set
        num_workers=4, # Increase workers
        pin_memory=True,
        collate_fn=collate_fn_skip_none_preprocessed # Use the new collate function
    )
    # --- End DataLoader Update ---
    print("Test DataLoader created.")


    # --- Load Model ---
    print(f"Loading model with {args.num_channels} input channels and {num_classes} classes...")
    model = ClassificationModel(
        encoder_type=args.encoder_type,
        input_channels=args.num_channels,
        num_classes=num_classes,
        device=device # Pass the determined device
    )

    # Load checkpoint
    model_checkpoint_file = os.path.join(args.checkpoint_path) # Or specific checkpoint name
    if not os.path.exists(model_checkpoint_file):
        print(f"Error: Model checkpoint not found at {model_checkpoint_file}")
        exit(1)

    try:
        print(f"Loading checkpoint dictionary from {model_checkpoint_file}")
        # Load the entire checkpoint dictionary, mapping tensors to the correct device
        checkpoint = torch.load(model_checkpoint_file, map_location=device)

        # Check if the expected keys exist
        if 'encoder_state_dict' not in checkpoint or 'decoder_state_dict' not in checkpoint:
             raise KeyError("Checkpoint dictionary missing 'encoder_state_dict' or 'decoder_state_dict'")

        print("Loading state dict into model encoder and decoder...")
        # Load the state dicts into the respective components
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # Ensure the model components are on the correct device (redundant if initialized correctly, but safe)
        model.to(device)
        print("Model loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {model_checkpoint_file}")
        exit(1)
    except KeyError as e:
        print(f"Error: Checkpoint file structure incorrect. {e}")
        exit(1)
    except Exception as e:
        # Catch other potential errors like corrupted file, mismatched keys within encoder/decoder
        print(f"Error loading model checkpoint: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        exit(1)


    # --- Run Evaluation ---
    print("Starting evaluation...")
    from tqdm import tqdm # Make sure tqdm is imported
    predictions, true_labels = run(model, test_dataloader, device)
    print("Evaluation finished.")
    # --- End Evaluation ---


    # --- Calculate and Print Metrics ---
    if not predictions or not true_labels:
        print("Error: No predictions or labels were generated during evaluation.")
        exit(1)

    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, target_names=[str(i) for i in sorted(list(class_labels))], zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    # --- End Metrics ---


    # --- Save Confusion Matrix Plot ---
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(list(class_labels)), yticklabels=sorted(list(class_labels)))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # --- CHANGE HERE ---
        # Get the directory part of the checkpoint path
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        # Ensure the directory exists (optional but good practice)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Join the directory with the filename
        plot_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        # --- END CHANGE ---

        plt.savefig(plot_path)
        print(f"Confusion matrix plot saved to {plot_path}")
        plt.close()
    except Exception as plot_e:
        print(f"Error saving confusion matrix plot: {plot_e}")
    # --- End Plot ---