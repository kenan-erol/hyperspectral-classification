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
from datasets import PreprocessedPatchDataset, collate_fn_skip_none_preprocessed

parser = argparse.ArgumentParser()
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--data_dir',
    type=str, required=True, help='Directory containing PREPROCESSED patches and labels.txt')
parser.add_argument('--label_file',
    type=str, required=True, help='Path to the labels.txt file WITHIN the data_dir')
parser.add_argument('--patch_size',
    type=int, default=224, help='Expected size of image patches')
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type used during training: vggnet11, resnet18')
parser.add_argument('--num_channels',
    type=int, required=True, help='Number of spectral bands in hyperspectral data')
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path directory containing the trained model checkpoint (e.g., model_best.pth)')
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()

def run(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


if __name__ == '__main__':
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

    # --- Load Class Labels from Original Label File ---
    # We still need the original label file to know the number of classes
    print(f"Reading original label file {args.label_file} to determine number of classes...")
    class_labels_set = set()
    try:
        with open(args.label_file, 'r') as f:
             for line_num, line in enumerate(f, 1): # line number for better warnings
                line = line.strip()
                if not line: continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) == 2:
                    path_part, label_part = parts
                    try:
                        label_int = int(label_part)
                        class_labels_set.add(label_int)
                    except ValueError:
                         print(f"Warning [Line {line_num}]: Invalid label '{label_part}' in original label file: {line}")
                else:
                     print(f"Warning [Line {line_num}]: Split error (expected 2 parts, got {len(parts)}) in original label file: {line}")
    except FileNotFoundError:
        print(f"Error: Original label file not found at {args.label_file}. Cannot determine number of classes.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading original label file {args.label_file}: {e}")
        sys.exit(1)

    if not class_labels_set:
        print("Error: No valid labels found in the original label file.")
        sys.exit(1)

    num_classes = len(class_labels_set)
    class_names = [str(i) for i in sorted(list(class_labels_set))]
    print(f"Determined {num_classes} classes from label file.")
    # --- End Load Class Labels ---


    # --- Determine Test Split by Loading from File ---
    if os.path.isfile(args.checkpoint_path):
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
    else:
        checkpoint_dir = args.checkpoint_path
        print(f"Warning: --checkpoint_path '{args.checkpoint_path}' is not a file. Assuming it's the directory containing test_samples.txt.")
    test_set_file_path = os.path.join(checkpoint_dir, 'test_samples.txt')

    print(f"Loading test set samples from: {test_set_file_path}")
    test_samples = []
    try:
        with open(test_set_file_path, 'r') as f_test:
            for line in f_test:
                line = line.strip()
                if not line: continue
                parts = line.rsplit(maxsplit=1)
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
        exit(1)
    except Exception as e:
        print(f"Error reading test set file {test_set_file_path}: {e}")
        exit(1)

    if not test_samples:
         print("Error: No samples loaded for the test set.")
         exit(1)
    print(f"Loaded {len(test_samples)} samples for the test set.")
    # --- End Load Test Split ---
    
    # --- Create Test Dataset ---
    transform_mean = [0.5] * args.num_channels if args.num_channels > 0 else None
    transform_std = [0.5] * args.num_channels if args.num_channels > 0 else None

    print("Creating test dataset instance...")
    test_dataset = PreprocessedPatchDataset(
        data_dir=args.data_dir,
        samples=test_samples,
        num_channels=args.num_channels,
        transform_mean=transform_mean,
        transform_std=transform_std,
        target_size=(args.patch_size, args.patch_size),
        save_visualization_path='hyper_checkpoints/resnet/transform_viz',
        num_visualizations_to_save=5
    )
    # --- End Instantiate ---
    print(f"Test dataset size: {len(test_dataset)} patches")


    # --- DataLoaders ---
    print("Creating Test DataLoader...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.n_batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_skip_none_preprocessed
    )
    # --- End DataLoader ---
    print("Test DataLoader created.")


    # --- Load Model ---
    print(f"Loading model with {args.num_channels} input channels and {num_classes} classes...")
    model = ClassificationModel(
        encoder_type=args.encoder_type,
        input_channels=args.num_channels,
        num_classes=num_classes,
        device=device
    )

    # Load checkpoint
    model_checkpoint_file = args.checkpoint_path
    if not os.path.isfile(model_checkpoint_file):
        print(f"Error: Checkpoint file not found at '{model_checkpoint_file}'")
        sys.exit(1)

    try:
        print(f"Restoring model weights from: {model_checkpoint_file}")
        checkpoint = torch.load(model_checkpoint_file, map_location=device)
        
        if 'encoder_state_dict' not in checkpoint or 'decoder_state_dict' not in checkpoint:
             raise KeyError("Checkpoint dictionary missing 'encoder_state_dict' or 'decoder_state_dict'")

        print("Loading state dict into model encoder and decoder...")
        
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        model.to(device)
        print("Model loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {model_checkpoint_file}")
        exit(1)
    except KeyError as e:
        print(f"Error: Checkpoint file structure incorrect. {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


    # --- Run Evaluation ---
    print("Starting evaluation...")
    predictions, true_labels = run(model, test_dataloader, device)
    print("Evaluation finished.")
    # --- End Evaluation ---


    # --- Calculate and Print Metrics ---
    if not predictions or not true_labels:
        print("Error: Evaluation produced no predictions or labels.")
        exit(1)

    accuracy = accuracy_score(true_labels, predictions)
    labels_present = sorted(list(set(true_labels) | set(predictions)))
    conf_matrix = confusion_matrix(true_labels, predictions, labels=labels_present)
    class_report = classification_report(true_labels, predictions, labels=labels_present, target_names=[class_names[i] for i in labels_present], zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
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
        
        for i in range(len(labels_present)):
            for j in range(len(labels_present)):
                 ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

        plot_save_path = os.path.join(checkpoint_dir, 'confusion_matrix.png')
        plt.savefig(plot_save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix plot saved to: {plot_save_path}")
    except Exception as plot_e:
        print(f"Error saving confusion matrix plot: {plot_e}")
    # --- End Plot ---