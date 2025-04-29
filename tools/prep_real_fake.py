# tools/prepare_real_fake_labels.py

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Define the labels for the two classes
REAL_LABEL = 1
FAKE_LABEL = 0

def is_valid_patch_file(file_path_obj: Path):
    """
    Check if the file is a valid patch file (measurement_patch_*.npy).
    """
    filename = file_path_obj.name.lower()
    return filename.startswith("measurement_patch_") and filename.endswith(".npy")

def create_real_fake_labels(data_dir, output_file):
    """
    Walk through the data directory containing 'real' and 'fake' subdirs
    and create a labels.txt file for binary classification.
    Format: <relative_path_to_patch.npy> <label> (0 for fake, 1 for real)
    """
    data_dir_path = Path(data_dir).resolve()
    output_file_path = Path(output_file).resolve() # Resolve output path as well

    # Ensure the output directory exists
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_file_path.parent}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning base directory: {data_dir_path}")
    print(f"Output file will be: {output_file_path}")

    output_lines = []
    real_files_count = 0
    fake_files_count = 0
    total_scanned = 0

    # Define the subdirectories and their corresponding labels
    subdirs_to_process = [
        ('real', REAL_LABEL),
        ('fake', FAKE_LABEL)
    ]

    for subdir_name, label in subdirs_to_process:
        current_subdir_path = data_dir_path / subdir_name
        print(f"\nScanning subdirectory: {current_subdir_path} for class {label} ({subdir_name})...")

        if not current_subdir_path.is_dir():
            print(f"Warning: Subdirectory '{current_subdir_path}' not found. Skipping.")
            continue

        # Use rglob to recursively find all .npy files
        # Convert to list first to show progress with tqdm correctly for rglob
        try:
            npy_files = list(current_subdir_path.rglob('*.npy'))
            if not npy_files:
                 print(f"No .npy files found in {current_subdir_path}.")
                 continue # Skip if no npy files found

            print(f"Found {len(npy_files)} potential .npy files. Validating...")
            file_iterator = tqdm(npy_files, desc=f"Processing {subdir_name}", unit="file")

            for file_path_obj in file_iterator:
                total_scanned += 1
                if is_valid_patch_file(file_path_obj):
                    try:
                        # Get path relative to the main data_dir
                        rel_path = file_path_obj.relative_to(data_dir_path)
                        # Use as_posix() for consistent forward slashes
                        output_lines.append(f"{rel_path.as_posix()} {label}")

                        if label == REAL_LABEL:
                            real_files_count += 1
                        else:
                            fake_files_count += 1
                    except ValueError:
                        print(f"Warning: Could not make path relative for {file_path_obj} relative to {data_dir_path}. Skipping.")
                    except Exception as e:
                         print(f"Error processing file {file_path_obj}: {e}")
        except Exception as e:
             print(f"Error scanning directory {current_subdir_path}: {e}")


    print(f"\nScanned {total_scanned} potential files.")

    if not output_lines:
        print("Error: No valid patch files found in 'real' or 'fake' subdirectories.", file=sys.stderr)
        sys.exit(1)

    # Sort lines for consistency (optional, but good practice)
    output_lines.sort()

    # Write labels file
    print(f"\nWriting {len(output_lines)} entries to labels file: {output_file_path}")
    try:
        with open(output_file_path, 'w') as f:
            f.write('\n'.join(output_lines))
            f.write('\n') # Add trailing newline
        print("Labels file created successfully.")
    except IOError as e:
        print(f"Error writing to output file {output_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\n--- Summary ---")
    print(f"Included {real_files_count} files for class {REAL_LABEL} (real).")
    print(f"Included {fake_files_count} files for class {FAKE_LABEL} (fake).")
    print(f"Total files included: {len(output_lines)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labels file for real vs. fake hyperspectral patch classification.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Base data directory containing 'real' and 'fake' subdirectories with patch files.")
    parser.add_argument("--output", type=str, default="labels_real_fake.txt",
                        help="Output path for the labels file (e.g., ./labels_real_fake.txt)")

    args = parser.parse_args()

    # Basic check if data_dir exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    create_real_fake_labels(args.data_dir, args.output)