import os
import re
from pathlib import Path
from tqdm import tqdm
import argparse

# --- Start Modification: Renamed and improved parsing function ---
def parse_path_for_mapping(path_str):
    """
    Parses a path string (potentially correct or incorrect) from labels.txt
    to extract key identifying components: Drug, Group, Mxxx, Filename.
    Returns a tuple (Drug, Group, Mxxx, Filename) or None if parsing fails.
    """
    try:
        p = Path(path_str)
        parts = [part for part in p.parts if part] # Filter empty parts

        if not parts or parts[0].lower() != 'patches':
            # print(f"Debug parse_path_for_mapping: Path doesn't start with patches: {parts}")
            return None # Must start with patches

        # Find filename, Mxxx, Group by working backwards
        filename = None
        mxxx_id = None
        group_dir = None
        filename_idx = -1

        # Iterate backwards from the end to find the standard structure
        for i in range(len(parts) - 1, 0, -1):
            part = parts[i]
            # Check for filename pattern
            if part.endswith('.npy') and part.startswith('measurement_patch_'):
                filename = part
                filename_idx = i
                # Check preceding parts for Mxxx and Group
                if i > 1 and parts[i-1].startswith('M'):
                    mxxx_id = parts[i-1]
                    if i > 2 and parts[i-2].lower() == 'group':
                        group_dir = parts[i-2]
                        break # Found the expected end structure: Group/Mxxx/Filename.npy

        if not (filename and mxxx_id and group_dir):
             # print(f"Debug parse_path_for_mapping: Could not find Group/Mxxx/filename structure at the end: {parts}")
             return None # Didn't find the expected structure at the end

        # Assume Drug name is the component immediately after 'patches'
        # This works for both 'patches/Drug/Group/...' and 'patches/Drug/Drug/Group/...'
        if len(parts) > 1:
            drug_name = parts[1]
            return (drug_name, group_dir, mxxx_id, filename)
        else:
            # print(f"Debug parse_path_for_mapping: Path too short after finding structure: {parts}")
            return None # Path is just 'patches/'? Should not happen if structure found

    except Exception as e:
        # print(f"Debug parse_path_for_mapping: Exception parsing '{path_str}': {e}")
        return None
# --- End Modification ---


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
                path_str_from_file = " ".join(parts[:-1]) # Use the new function name

                # --- Start Modification: Use the new parsing function ---
                key_parts = parse_path_for_mapping(path_str_from_file)
                # --- End Modification ---

                if key_parts:
                    try:
                        label = int(label_str)
                        # Check for duplicate keys which might indicate issues in original labels.txt
                        # if key_parts in label_mapping and label_mapping[key_parts] != label:
                        #      print(f"Warning: Duplicate key {key_parts} with different labels ({label_mapping[key_parts]} vs {label}) found in original file. Overwriting.")
                        label_mapping[key_parts] = label
                    except ValueError:
                        # print(f"Warning: Could not convert label '{label_str}' to int for key {key_parts}")
                        skipped_original += 1
                else:
                    # print(f"Warning: Could not parse key parts from line: {line}")
                    skipped_original += 1

    except FileNotFoundError:
        print(f"Error: Original labels file not found at '{original_labels_path}'")
        exit(1)
    except Exception as e:
        print(f"Error reading original labels file: {e}")
        exit(1)

    print(f"Created mapping for {len(label_mapping)} unique entries. Skipped {skipped_original} lines from original file.")
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

    # Use tqdm for walking directory as well
    file_list = []
    for root, _, files in os.walk(patches_base_dir):
        for filename in files:
            if filename.endswith('.npy') and filename.startswith('measurement_patch_'):
                 file_list.append(os.path.join(root, filename))

    print(f"Found {len(file_list)} potential '.npy' files. Matching them to labels...")

    for full_path in tqdm(file_list, desc="Matching files"):
        found_files += 1 # Count every file found by walk
        relative_to_walk_start = os.path.relpath(full_path, patches_base_dir)

        # Parse this correct relative path (e.g., Drug/Group/Mxxx/filename.npy)
        correct_key_parts = parse_correct_path(relative_to_walk_start)

        if correct_key_parts:
            # Look up the label using the parsed parts from the *correct* path
            label = label_mapping.get(correct_key_parts)
            if label is not None:
                # Construct the final path string starting with 'patches/'
                final_path_str = Path('patches') / relative_to_walk_start
                corrected_lines.append(f"{final_path_str.as_posix()} {label}") # Use as_posix for forward slashes
                matched_files += 1
            else:
                # This file exists but its key wasn't in the mapping from original labels.txt
                # print(f"Warning: No label found in mapping for key: {correct_key_parts} (File: {relative_to_walk_start})")
                unmatched_files += 1
        else:
             # The path structure of the found file itself is unexpected
             # print(f"Warning: Could not parse correct path structure for found file: {relative_to_walk_start}")
             unmatched_files += 1 # Count files that couldn't be parsed correctly

    print(f"\nTotal files processed: {found_files}")
    print(f"Successfully matched {matched_files} files to labels.")
    if unmatched_files > 0:
        # This warning is now more meaningful - it means files exist on disk
        # that either weren't in the original labels file at all, or couldn't be parsed.
        print(f"Warning: Could not find labels or parse path for {unmatched_files} files found in the directory structure.")

    if not corrected_lines:
        print("Error: No corrected label lines were generated. Check paths and mapping.")
        exit(1)

    print(f"\nStep 3: Writing corrected labels to '{output_labels_path}'...")
    try:
        # Resolve to an absolute path before creating dirs or opening
        absolute_output_path = os.path.abspath(output_labels_path)
        # print(f"Resolved absolute output path: {absolute_output_path}") # Keep for debugging if needed

        # Ensure output directory exists using the absolute path
        output_dir = os.path.dirname(absolute_output_path)
        if output_dir and not os.path.exists(output_dir):
            # print(f"Creating output directory: {output_dir}") # Keep for debugging if needed
            os.makedirs(output_dir)

        # Write using the absolute path
        with open(absolute_output_path, 'w') as f_out:
            # Sort lines for consistency (optional, but helpful)
            corrected_lines.sort()
            for line in corrected_lines:
                f_out.write(line + "\n")
        print("Corrected labels file written successfully.")
    except Exception as e:
        print(f"Error writing output file: {e}")
        # import traceback # Uncomment for more detailed debugging if needed
        # traceback.print_exc() # Uncomment for more detailed debugging if needed
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
