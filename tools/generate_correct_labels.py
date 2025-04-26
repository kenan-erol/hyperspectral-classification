import os
import sys

# --- Configuration ---
# File containing the ground truth: filename<separator>label
LABELS_TRUTH_FILE = 'labels.txt'
# The root directory where your 'train', 'test', 'valid' folders are located.
# Assumes the script is run from the directory containing train/, test/, valid/
# Or change '.' to the actual path if needed, e.g., '/path/to/dataset'
SEARCH_ROOT_DIR = '.'
# The output file we will create
OUTPUT_FILE = 'correct_full_paths_labels.txt'
# What separates the filename and label in LABELS_TRUTH_FILE? (e.g., ' ', '\t', ',')
# Adjust if your separator is different (looks like space based on typical formats)
SEPARATOR = ' '
# Set of valid image extensions to look for (lowercase)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
# --- End Configuration ---

def find_file_path(filename_to_find, root_dir):
    """Searches for a specific filename within the root directory and subdirectories."""
    for dirpath, _, filenames in os.walk(root_dir):
        for current_filename in filenames:
            if current_filename == filename_to_find:
                # Found it! Return the full path.
                return os.path.join(dirpath, current_filename)
    # If the loop finishes without finding the file
    return None

def main():
    print(f"Reading ground truth labels from: {LABELS_TRUTH_FILE}")
    
    # 1. Read labels.txt into a dictionary: {filename: label}
    label_map = {}
    try:
        with open(LABELS_TRUTH_FILE, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines or comments
                    continue
                parts = line.split(SEPARATOR, 1) # Split only on the first separator
                if len(parts) == 2:
                    filename, label = parts
                    if not filename:
                         print(f"Warning: Empty filename on line {i+1} in {LABELS_TRUTH_FILE}. Skipping.", file=sys.stderr)
                         continue
                    if not label:
                         print(f"Warning: Empty label for filename '{filename}' on line {i+1} in {LABELS_TRUTH_FILE}. Skipping.", file=sys.stderr)
                         continue
                    if filename in label_map:
                        print(f"Warning: Duplicate filename '{filename}' detected in {LABELS_TRUTH_FILE} (line {i+1}). Using the latest label '{label}'.", file=sys.stderr)
                    label_map[filename.strip()] = label.strip()
                else:
                    print(f"Warning: Could not parse line {i+1} in {LABELS_TRUTH_FILE}: '{line}'. Expected format 'filename{SEPARATOR}label'. Skipping.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Cannot find the label file: {LABELS_TRUTH_FILE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {LABELS_TRUTH_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    if not label_map:
        print(f"Error: No valid label entries found in {LABELS_TRUTH_FILE}.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Successfully read {len(label_map)} label entries.")
    print(f"Searching for image files under: {os.path.abspath(SEARCH_ROOT_DIR)}")

    # 2. Walk the directory tree and match files
    found_files_count = 0
    output_lines = []
    files_in_labels_not_found = set(label_map.keys()) # Keep track of files we need to find

    # Use os.walk to find all files efficiently
    all_image_files_found = {} # Store {filename: full_path} for quick lookup
    for dirpath, _, filenames in os.walk(SEARCH_ROOT_DIR):
        for filename in filenames:
            # Check if it's an image file we care about
            _, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTENSIONS:
                 # Check if this filename is one we are looking for from labels.txt
                 if filename in label_map:
                     full_path = os.path.join(dirpath, filename)
                     # Use os.path.normpath to clean up path (e.g., './train/...' -> 'train/...')
                     normalized_path = os.path.normpath(full_path)
                     
                     # Get the correct label from our map
                     correct_label = label_map[filename]
                     
                     output_lines.append(f"{normalized_path}{SEPARATOR}{correct_label}")
                     found_files_count += 1
                     
                     # Mark this file as found
                     if filename in files_in_labels_not_found:
                         files_in_labels_not_found.remove(filename)
                     
                     # Optional: Check for duplicates found in filesystem
                     if filename in all_image_files_found:
                         print(f"Warning: Found duplicate filename '{filename}' in filesystem.", file=sys.stderr)
                         print(f"  - Already found at: {all_image_files_found[filename]}", file=sys.stderr)
                         print(f"  - Now found at:    {normalized_path}", file=sys.stderr)
                         print(f"  - Using the latest one found for the output file.", file=sys.stderr)
                     all_image_files_found[filename] = normalized_path


    print(f"Search complete. Matched {found_files_count} files listed in {LABELS_TRUTH_FILE}.")

    # 3. Report any files from labels.txt that were not found
    if files_in_labels_not_found:
        print(f"\nWarning: {len(files_in_labels_not_found)} files listed in {LABELS_TRUTH_FILE} were NOT found in the directory tree:", file=sys.stderr)
        for missing_file in sorted(list(files_in_labels_not_found)):
            print(f"  - {missing_file}", file=sys.stderr)
    else:
        print(f"All {len(label_map)} files listed in {LABELS_TRUTH_FILE} were found.")

    # 4. Write the output file
    if not output_lines:
        print("\nError: No matching files were found to write to the output file.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(OUTPUT_FILE, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"\nSuccessfully created output file: {OUTPUT_FILE} with {len(output_lines)} entries.")
    except Exception as e:
        print(f"\nError writing output file {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")

if __name__ == "__main__":
    main()