import os
import sys
from pathlib import Path
from tqdm import tqdm # Import tqdm for progress bars

# --- Configuration ---
# File containing the ground truth: full/path/patch.npy<separator>label
# Use the remapped file as the source of truth for labels now
LABELS_TRUTH_FILE = 'labels_remap.txt' # *** CHANGED: Use the output from the previous step ***
# The root directory where your patch folders are located (e.g., 'patches/' inside data_processed_patch)
# Assumes the script is run from 'data_processed_patch' directory
SEARCH_ROOT_DIR = 'patches' # *** CHANGED: Search within the 'patches' subdirectory ***
# The output file we will create
OUTPUT_FILE = 'labels_final_corrected.txt' # *** CHANGED: New output filename ***
# What separates the path and label in LABELS_TRUTH_FILE?
SEPARATOR = ' '
# Set of valid file extensions to look for (lowercase)
TARGET_EXTENSIONS = {'.npy'} # *** CHANGED: Look for .npy files ***
# --- End Configuration ---

def main():
    # --- Get Absolute Paths ---
    # Get absolute path for the labels file based on the script's location or CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes script is in tools/
    # Construct absolute path relative to project root if labels file is relative
    if not os.path.isabs(LABELS_TRUTH_FILE):
         abs_labels_truth_file = os.path.join(project_root, LABELS_TRUTH_FILE)
    else:
         abs_labels_truth_file = LABELS_TRUTH_FILE

    # Construct absolute path for search directory relative to project root
    if not os.path.isabs(SEARCH_ROOT_DIR):
        abs_search_root_dir = os.path.join(project_root, SEARCH_ROOT_DIR)
    else:
        abs_search_root_dir = SEARCH_ROOT_DIR

    # Construct absolute path for output file relative to project root
    if not os.path.isabs(OUTPUT_FILE):
        abs_output_file = os.path.join(project_root, OUTPUT_FILE)
    else:
        abs_output_file = OUTPUT_FILE

    print(f"Reading ground truth labels from: {abs_labels_truth_file}")
    print(f"Searching for patch files under: {abs_search_root_dir}")
    print(f"Will write output to: {abs_output_file}")

    # 1. Read labels_remap.txt into a dictionary: {filename_only: (full_path_from_file, label)}
    label_map = {}
    expected_filenames = set()
    try:
        with open(abs_labels_truth_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines or comments
                    continue
                parts = line.split(SEPARATOR, 1) # Split only on the first separator
                if len(parts) == 2:
                    full_path_str, label_str = parts
                    full_path_str = full_path_str.strip()
                    label_str = label_str.strip()

                    if not full_path_str or not label_str:
                         print(f"Warning: Empty path or label on line {i+1} in {abs_labels_truth_file}. Skipping.", file=sys.stderr)
                         continue

                    # Extract just the filename (e.g., measurement_patch_99.npy)
                    filename_only = os.path.basename(full_path_str)

                    # Validate label is integer
                    try:
                        label_int = int(label_str)
                    except ValueError:
                        print(f"Warning: Invalid label '{label_str}' on line {i+1} in {abs_labels_truth_file}. Skipping.", file=sys.stderr)
                        continue

                    if filename_only in label_map:
                        # This indicates a potential problem - same filename listed twice
                        print(f"Warning: Duplicate filename '{filename_only}' detected in {abs_labels_truth_file} (line {i+1}). Check source file. Using latest entry.", file=sys.stderr)
                    label_map[filename_only] = (full_path_str, label_int)
                    expected_filenames.add(filename_only)
                else:
                    print(f"Warning: Could not parse line {i+1} in {abs_labels_truth_file}: '{line}'. Expected format 'full/path/file.npy{SEPARATOR}label'. Skipping.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Cannot find the label file: {abs_labels_truth_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {abs_labels_truth_file}: {e}", file=sys.stderr)
        sys.exit(1)

    if not label_map:
        print(f"Error: No valid label entries found in {abs_labels_truth_file}.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully read {len(label_map)} unique label entries from {os.path.basename(abs_labels_truth_file)}.")

    # 2. Walk the actual directory tree and find files
    output_lines = []
    files_found_in_fs = set()
    print(f"Searching filesystem...")

    # Use tqdm for the directory walk
    walk_iter = os.walk(abs_search_root_dir)
    # Estimate total files for tqdm (optional, can be slow for huge dirs)
    # total_files_estimate = sum(len(files) for _, _, files in os.walk(abs_search_root_dir))

    # Wrap os.walk with tqdm
    pbar = tqdm(walk_iter, desc="Scanning directories", unit=" dir")
    for dirpath, _, filenames in pbar:
        # Update description with current directory being scanned
        pbar.set_postfix_str(os.path.basename(dirpath), refresh=True)
        for filename in filenames:
            # Check extension
            _, ext = os.path.splitext(filename)
            if ext.lower() in TARGET_EXTENSIONS:
                # Check if this filename is one we expect from the labels file
                if filename in label_map:
                    # Get the label associated with this filename
                    _, correct_label = label_map[filename] # We only need the label here

                    # Construct the relative path from the *project root*
                    # This assumes SEARCH_ROOT_DIR is relative to project root
                    full_actual_path = os.path.join(dirpath, filename)
                    relative_path_from_project_root = os.path.relpath(full_actual_path, project_root)
                    # Ensure forward slashes for consistency
                    output_path_str = Path(relative_path_from_project_root).as_posix()

                    output_lines.append(f"{output_path_str}{SEPARATOR}{correct_label}")
                    files_found_in_fs.add(filename) # Mark this filename as found

    pbar.close() # Close the progress bar

    print(f"Search complete. Found {len(files_found_in_fs)} matching '.npy' files listed in {os.path.basename(abs_labels_truth_file)}.")

    # 3. Report discrepancies
    files_in_labels_not_found_in_fs = expected_filenames - files_found_in_fs
    if files_in_labels_not_found_in_fs:
        print(f"\nWarning: {len(files_in_labels_not_found_in_fs)} files listed in {os.path.basename(abs_labels_truth_file)} were NOT found in the directory tree ({abs_search_root_dir}):", file=sys.stderr)
        # Print only a few examples if the list is long
        count = 0
        for missing_file in sorted(list(files_in_labels_not_found_in_fs)):
            print(f"  - {missing_file}", file=sys.stderr)
            count += 1
            if count >= 10:
                print(f"  ... and {len(files_in_labels_not_found_in_fs) - count} more.", file=sys.stderr)
                break
    else:
         # Check if the counts match exactly
         if len(label_map) == len(files_found_in_fs):
              print(f"\nOK: All {len(label_map)} files listed in {os.path.basename(abs_labels_truth_file)} were found in the filesystem.")
         else:
              # This case shouldn't happen if no files were missing, but good to check
              print(f"\nWarning: Mismatch in counts. Labels file had {len(label_map)} unique entries, but found {len(files_found_in_fs)} matching files in FS.", file=sys.stderr)


    # 4. Write the output file
    if not output_lines:
        print("\nError: No matching files were found to write to the output file.", file=sys.stderr)
        sys.exit(1)

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(abs_output_file), exist_ok=True)
        # Sort output lines for consistency
        output_lines.sort()
        with open(abs_output_file, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"\nSuccessfully created output file: {abs_output_file} with {len(output_lines)} entries.")
    except Exception as e:
        print(f"\nError writing output file {abs_output_file}: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")

if __name__ == "__main__":
    main()