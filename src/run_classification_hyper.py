import os, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm

from classification_model import ClassificationModel
# --- Import the NEW dataset and collate function ---
from datasets import PreprocessedPatchDataset, collate_fn_skip_none_preprocessed
# --- End Import ---


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
# parser.add_argument('--train_split_ratio', # Keep this to identify the TEST split
#     type=float, default=0.8, help='Proportion of PATCHES used for training (to determine test set)')

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

    # --- Load Class Labels from Original Label File ---
    # We still need the original label file to know the number of classes
    print(f"Reading original label file {args.label_file} to determine number of classes...")
    class_labels_set = set()
    try:
        with open(args.label_file, 'r') as f:
             for line_num, line in enumerate(f, 1): # Add line number for better warnings
                line = line.strip()
                if not line: continue
                parts = line.rsplit(maxsplit=1) # Correctly splits path from label
                if len(parts) == 2:
                    path_part, label_part = parts # Use clearer variable names
                    try:
                        # --- FIXED: Added conversion and adding to set ---
                        label_int = int(label_part)
                        class_labels_set.add(label_int)
                        # --- END FIXED ---
                    except ValueError:
                         # --- FIXED: Added specific warning ---
                         print(f"Warning [Line {line_num}]: Invalid label '{label_part}' in original label file: {line}")
                else:
                     # --- FIXED: Added specific warning ---
                     print(f"Warning [Line {line_num}]: Split error (expected 2 parts, got {len(parts)}) in original label file: {line}")
    except FileNotFoundError:
        print(f"Error: Original label file not found at {args.label_file}. Cannot determine number of classes.")
        sys.exit(1) # Use sys.exit
    except Exception as e:
        print(f"Error reading original label file {args.label_file}: {e}")
        sys.exit(1) # Use sys.exit

    if not class_labels_set:
        # --- MODIFIED: Changed to sys.exit ---
        print("Error: No valid labels found in the original label file.")
        sys.exit(1)
        # --- END MODIFIED ---

    num_classes = len(class_labels_set)
    class_names = [str(i) for i in sorted(list(class_labels_set))] # For reports
    print(f"Determined {num_classes} classes from label file.")
    # --- End Load Class Labels ---


    # --- Determine Test Split by Loading from File ---
    # Construct path to the test samples file based on the checkpoint file's directory
    # --- MODIFIED: Check if checkpoint_path is a file or dir ---
    if os.path.isfile(args.checkpoint_path):
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
    else:
        # Assume it's a directory path if it's not a file
        checkpoint_dir = args.checkpoint_path
        print(f"Warning: --checkpoint_path '{args.checkpoint_path}' is not a file. Assuming it's the directory containing test_samples.txt.")
    # --- END MODIFIED ---
    test_set_file_path = os.path.join(checkpoint_dir, 'test_samples.txt')

    print(f"Loading test set samples from: {test_set_file_path}")
    test_samples = []
    try:
        with open(test_set_file_path, 'r') as f_test:
            for line in f_test:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) == 2:
                    rel_path, label_str = parts
                    try:
                        label = int(label_str)
                        test_samples.append((rel_path, label))
                    except ValueError:
                        print(f"Warning: Skipping line with non-integer label in test set file: {line}")
                else:
                    print(f"Warning: Skipping malformed line in test set file: {line}")
    except FileNotFoundError:
        print(f"Error: Test set file '{test_set_file_path}' not found.")
        print("Make sure the training script saved it in the same directory as the checkpoint.")
        # --- Fallback (Optional, but less ideal) ---
        # print("Attempting fallback: Re-calculating split using random_state=42.")
        # print("WARNING: This might not be the exact test set used during training if data changed.")
        # try:
        #     # Reload all samples from the main label file again for fallback split
        #     all_patch_samples_fallback = []
        #     with open(args.label_file, 'r') as f_fallback:
        #         # ... (similar loading logic as in train.py) ...
        #     patch_paths_fallback = [s[0] for s in all_patch_samples_fallback]
        #     patch_labels_fallback = [s[1] for s in all_patch_samples_fallback]
        #     # Use a dummy train_split_ratio if needed, assuming 0.8 was used in training
        #     train_split_ratio_fallback = 0.8
        #     _, test_paths, _, test_labels = train_test_split(
        #         patch_paths_fallback, patch_labels_fallback,
        #         train_size=train_split_ratio_fallback,
        #         random_state=42, # MUST match the one used in training
        #         stratify=patch_labels_fallback
        #     )
        #     test_samples = list(zip(test_paths, test_labels))
        #     print(f"Fallback split generated {len(test_samples)} test samples.")
        # except Exception as fallback_e:
        #     print(f"Fallback split failed: {fallback_e}")
        #     exit(1) # Exit if file not found and fallback fails
        # --- End Fallback ---
        exit(1) # Exit if test set file is mandatory
    except Exception as e:
        print(f"Error reading test set file {test_set_file_path}: {e}")
        exit(1)

    if not test_samples:
         print("Error: No samples loaded for the test set.")
         exit(1)
    print(f"Loaded {len(test_samples)} samples for the test set.")
    # --- End Load Test Split ---
    
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
    # from tqdm import tqdm # Make sure tqdm is imported
    predictions, true_labels = run(model, test_dataloader, device)
    print("Evaluation finished.")
    # --- End Evaluation ---


    # --- Calculate and Print Metrics ---
    if not predictions or not true_labels:
        print("Error: Evaluation produced no predictions or labels.")
        exit(1)

    accuracy = accuracy_score(true_labels, predictions)
    # Ensure labels for confusion matrix and report are within the expected range
    labels_present = sorted(list(set(true_labels) | set(predictions)))
    conf_matrix = confusion_matrix(true_labels, predictions, labels=labels_present)
    class_report = classification_report(true_labels, predictions, labels=labels_present, target_names=[class_names[i] for i in labels_present], zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    # Format confusion matrix for better readability
    print("Labels:", labels_present)
    print(conf_matrix)
    # --- End Metrics ---


    # --- Save Confusion Matrix Plot ---
    try:
        fig, ax = plt.subplots(figsize=(max(6, num_classes // 2), max(5, num_classes // 2)))
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        tick_marks = np.arange(len(labels_present))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([class_names[i] for i in labels_present], rotation=45, ha='left')
        ax.set_yticklabels([class_names[i] for i in labels_present])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')

        # Add text annotations
        for i in range(len(labels_present)):
            for j in range(len(labels_present)):
                 ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')


        # Save the plot in the same directory as the checkpoint
        plot_save_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        plt.savefig(plot_save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix plot saved to: {plot_save_path}")
    except Exception as plot_e:
        print(f"Error saving confusion matrix plot: {plot_e}")
    # --- End Plot ---