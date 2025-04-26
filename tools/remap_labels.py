import os
import re
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_incorrect_path(incorrect_path_str):
    """
    Parses the potentially incorrect path string from the original labels.txt
    to extract key identifying components: Drug, Group, Mxxx, Filename.
    Assumes the first directory after 'patches/' is the Drug name, and
    the structure '.../Group/Mxxx/filename.npy' exists towards the end.
    """
    try:
        p = Path(incorrect_path_str)
        parts = [part for part in p.parts if part] # Filter empty parts

        if len(parts) < 5 or parts[0].lower() != 'patches':
            return None # Not a valid structure

        drug_name = parts[1] # Assume first dir after 'patches' is the drug name
        filename = parts[-1]
        mxxx_id = parts[-2]
        group_dir = parts[-3]

        # Basic validation
        if group_dir.lower() != 'group' or not mxxx_id.startswith('M') or not filename.endswith('.npy'):
             # Try going back one level if repetition exists like Drug/Drug/Group...
             if len(parts) > 5 and parts[-4].lower() == 'group':
                 filename = parts[-1]
                 mxxx_id = parts[-2]
                 group_dir = parts[-4] # Use the earlier 'Group'
                 # Drug name remains parts[1]
             else:
                 # print(f"Warning: Unexpected structure near end: {parts[-3:]}")
                 return None # Structure doesn't match expected end

        return (drug_name, group_dir, mxxx_id, filename)

    except Exception as e:
        # print(f"Error parsing incorrect path '{incorrect_path_str}': {e}")
        return None

def parse_correct_path(correct_relative_path_str):
    """
    Parses the correct relative path string (e.g., 'Drug/Group/Mxxx/filename.npy')
    """
    try:
        p = Path(correct_relative_path_str)
        parts = [part for part in p.parts if part] # Filter empty parts

        if len(parts) != 4: # Expecting Drug/Group/Mxxx/filename.npy
             # print(f"Warning: Correct path has unexpected number of parts: {parts}")
             return None

        drug_name = parts[0]
        group_dir = parts[1]
        mxxx_id = parts[2]
        filename = parts[3]

        # Basic validation
        if group_dir.lower() != 'group' or not mxxx_id.startswith('M') or not filename.endswith('.npy'):
             # print(f"Warning: Correct path structure mismatch: {parts}")
             return None

        return (drug_name, group_dir, mxxx_id, filename)

    except Exception as e:
        # print(f"Error parsing correct path '{correct_relative_path_str}': {e}")
        return None


def main(original_labels_path, patches_base_dir, output_labels_path):
    """
    Generates a corrected labels file based on actual file structure
    and labels from the original (potentially incorrect) labels file.
    """
    print("Step 1: Creating mapping from original labels file...")
    label_mapping = {} # Key: (Drug, Group, Mxxx, Filename), Value: label
    skipped_original = 0
    try:
        with open(original_labels_path, 'r') as f_orig:
            for line in tqdm(f_orig, desc="Parsing original labels"):
                line = line.strip()
                if not line: continue

                parts = line.split()
                if len(parts) < 2: continue

                label_str = parts[-1]
                incorrect_path_str = " ".join(parts[:-1])

                key_parts = parse_incorrect_path(incorrect_path_str)

                if key_parts:
                    try:
                        label = int(label_str)
                        # Use lowercase for key components for case-insensitive matching later if needed,
                        # but store original case drug name if needed for output consistency.
                        # For simplicity, let's use original case for now.
                        label_mapping[key_parts] = label
                    except ValueError:
                        skipped_original += 1
                else:
                    skipped_original += 1

    except FileNotFoundError:
        print(f"Error: Original labels file not found at '{original_labels_path}'")
        exit(1)
    except Exception as e:
        print(f"Error reading original labels file: {e}")
        exit(1)

    print(f"Created mapping for {len(label_mapping)} entries. Skipped {skipped_original} lines from original file.")
    if not label_mapping:
        print("Error: No valid entries could be mapped from the original labels file.")
        exit(1)

    print(f"\nStep 2: Walking directory '{patches_base_dir}' to find actual patch files...")
    corrected_lines = []
    found_files = 0
    matched_files = 0
    unmatched_files = 0

    # Ensure the base directory exists
    if not os.path.isdir(patches_base_dir):
         print(f"Error: Patches base directory '{patches_base_dir}' not found.")
         exit(1)

    for root, _, files in os.walk(patches_base_dir):
        for filename in files:
            if filename.endswith('.npy') and filename.startswith('measurement_patch_'):
                found_files += 1
                full_path = os.path.join(root, filename)
                # Get path relative to the *parent* of patches_base_dir if it includes 'patches' itself
                # Or relative to patches_base_dir if it's the direct container like 'patches/'
                relative_to_walk_start = os.path.relpath(full_path, patches_base_dir)

                # Parse this correct relative path (e.g., Drug/Group/Mxxx/filename.npy)
                correct_key_parts = parse_correct_path(relative_to_walk_start)

                if correct_key_parts:
                    # Look up the label using the parsed parts
                    label = label_mapping.get(correct_key_parts)
                    if label is not None:
                        # Construct the final path string starting with 'patches/'
                        final_path_str = Path('patches') / relative_to_walk_start
                        corrected_lines.append(f"{final_path_str.as_posix()} {label}") # Use as_posix for forward slashes
                        matched_files += 1
                    else:
                        # print(f"Warning: No label found in mapping for key: {correct_key_parts} (File: {relative_to_walk_start})")
                        unmatched_files += 1
                else:
                     # print(f"Warning: Could not parse correct path structure for: {relative_to_walk_start}")
                     unmatched_files += 1 # Count files that couldn't be parsed correctly

    print(f"Found {found_files} '.npy' files.")
    print(f"Successfully matched {matched_files} files to labels.")
    if unmatched_files > 0:
        print(f"Warning: Could not find labels or parse path for {unmatched_files} files.")

    if not corrected_lines:
        print("Error: No corrected label lines were generated. Check paths and mapping.")
        exit(1)

    print(f"\nStep 3: Writing corrected labels to '{output_labels_path}'...")
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
        with open(output_labels_path, 'w') as f_out:
            # Sort lines for consistency (optional, but helpful)
            corrected_lines.sort()
            for line in corrected_lines:
                f_out.write(line + "\n")
        print("Corrected labels file written successfully.")
    except Exception as e:
        print(f"Error writing output file: {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap labels from an old labels file to a corrected file structure.")
    parser.add_argument('--original_labels', type=str, default='./dump/labels.txt',
                        help="Path to the original labels.txt file (with potentially incorrect paths).")
    parser.add_argument('--patches_dir', type=str, default='./data_processed_patch/patches',
                        help="Path to the base directory containing the *correctly* structured patch folders (e.g., './data_processed_patch/patches').")
    parser.add_argument('--output_labels', type=str, default='./labels_corrected.txt',
                        help="Path to save the new, corrected labels file.")

    args = parser.parse_args()

    main(args.original_labels, args.patches_dir, args.output_labels)