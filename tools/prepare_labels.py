#!/usr/bin/env python3
import os
import argparse
import re
from collections import defaultdict
from pathlib import Path # Use pathlib for easier path manipulation

# --- Drug Name Standardization (Keep as is) ---
def standardize_drug_name(raw_name):
    """Standardize drug name variations."""
    drug_name_mapping = {
        'zopiklon': ['Zopiklon', 'zopiklon'],
        'tramadol': ['Tramadol', 'tramadol', 'Tradolan', 'tradolan'],
        'ecstasy': ['Ecstasy', 'ecstasy'],
        'clonazepam': ['Klonazepam', 'klonazepam', 'Clonazepam', 'clonazepam'],
        'oxycodone': ['Oxycodone', 'oxycodone', 'oxykodon', 'Oxykodon', 'OxyContin', 'oxycontin'],
        'bromazolam': ['Bromazolam', 'bromazolam']
    }
    raw_name_lower = raw_name.lower()
    for standard_name, variations in drug_name_mapping.items():
        for variation in variations:
            # Check if the variation is part of the raw name (e.g., "Bromazolam 2025-01-20")
            if variation.lower() in raw_name_lower:
                return standard_name
    # If no match, return the cleaned raw name (lowercase, no date)
    cleaned_name = re.sub(r'\s+\d{4}-\d{2}-\d{2}$', '', raw_name).strip().lower()
    return cleaned_name if cleaned_name else None

# --- Modified: Extract drug name from patch directory structure ---
def extract_drug_name_from_patch_path(relative_path_obj: Path):
    """
    Extracts and standardizes the drug name from the relative path of a patch file.
    Assumes structure like: DrugName YYYY-MM-DD/Group/Mxxxx/measurement_patch_y.npy
    """
    try:
        # The first part of the relative path should be the drug directory name
        if not relative_path_obj.parts:
            return None
        drug_dir_name = relative_path_obj.parts[0]

        # Skip parts that indicate empty trays or non-drug folders explicitly
        if any(skip_term in drug_dir_name.lower() for skip_term in ['empty tray', 'drop-', 'group']):
             return None

        # Standardize the extracted name
        standardized_name = standardize_drug_name(drug_dir_name)
        return standardized_name
    except IndexError:
        return None
    except Exception as e:
        print(f"Error extracting drug name from {relative_path_obj}: {e}")
        return None

# --- Modified: Validation for patch files ---
def is_valid_patch_file(file_path_obj: Path):
    """
    Check if the file is a valid patch file (measurement_patch_*.npy).
    Basic check, can be expanded (e.g., check size > 0).
    """
    filename = file_path_obj.name.lower()
    return filename.startswith("measurement_patch_") and filename.endswith(".npy")

# --- Modified: Main function to handle patch directories ---
def create_labels_file(data_dir, output_file):
    """
    Walk through the data directory (expecting patch structure) and create a labels.txt file.
    Format: <relative_path_to_patch.npy> <integer_label>
    """
    data_dir_path = Path(data_dir).resolve()
    output_file_path = Path(output_file)

    # Ensure the output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning directory: {data_dir_path}")

    # Find all potential patch files
    patch_files_relative = []
    total_files_scanned = 0
    for root, _, files in os.walk(data_dir_path):
        root_path = Path(root)
        for file in files:
            total_files_scanned += 1
            file_path_obj = root_path / file
            # Check if it's a valid patch file based on name
            if is_valid_patch_file(file_path_obj):
                try:
                    # Get path relative to the input data_dir
                    rel_path = file_path_obj.relative_to(data_dir_path)
                    patch_files_relative.append(rel_path)
                except ValueError:
                    print(f"Warning: Could not make path relative for {file_path_obj} relative to {data_dir_path}")

    print(f"Scanned {total_files_scanned} files. Found {len(patch_files_relative)} potential patch files.")

    # Group by drug class
    class_to_files = defaultdict(list)
    skipped_files_count = 0

    for rel_path_obj in patch_files_relative:
        drug_name = extract_drug_name_from_patch_path(rel_path_obj)
        if drug_name:
            class_to_files[drug_name].append(rel_path_obj)
        else:
            # print(f"Skipping file with unrecognized drug type or structure: {rel_path_obj}") # Optional debug
            skipped_files_count += 1

    # Assign integer labels to each class
    class_to_label = {drug_name: i for i, drug_name in enumerate(sorted(class_to_files.keys()))}

    # Create mapping for the output file
    output_lines = []
    included_files_count = 0
    for drug_name, files in class_to_files.items():
        label = class_to_label[drug_name]
        for rel_path_obj in files:
            # Use as_posix() for consistent path separators in the output file
            output_lines.append(f"{rel_path_obj.as_posix()} {label}")
            included_files_count += 1

    # Sort lines for consistency (optional, but good practice)
    output_lines.sort()

    # Write labels file
    try:
        with open(output_file_path, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\nCreated labels file at {output_file_path}")
    except IOError as e:
        print(f"Error writing to output file {output_file_path}: {e}")
        return

    # Print summary
    print(f"Included {included_files_count} files across {len(class_to_files)} classes.")
    print(f"Skipped {skipped_files_count} files (unrecognized drug type, structure, or relative path error).")

    if class_to_label:
        print("\nClass mapping:")
        for drug_name, label in class_to_label.items():
            print(f"  Class {label}: {drug_name} ({len(class_to_files[drug_name])} files)")
    else:
        print("\nWarning: No classes were identified.")

    return class_to_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labels file for hyperspectral classification from patch data.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory containing patch files (e.g., ./data_processed_patch/patches)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the labels file (e.g., ./labels_patches.txt)")

    args = parser.parse_args()

    create_labels_file(args.data_dir, args.output)