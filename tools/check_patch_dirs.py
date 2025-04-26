import os
import sys
from pathlib import Path
import argparse

def get_expected_top_dirs(labels_file_path):
    """
    Reads the labels file and extracts the unique top-level directory names
    expected under the 'patches/' base directory.
    Handles paths like 'patches/Drug Name/MaybeRepeat/Group/...'
    Returns a set of expected directory names (original case).
    """
    expected_dirs = set()
    line_num = 0
    try:
        with open(labels_file_path, 'r') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line {line_num} in labels file: '{line}'", file=sys.stderr)
                    continue

                path_str = " ".join(parts[:-1]) # Handle spaces in path

                try:
                    # Use pathlib for robust path parsing
                    p = Path(path_str)
                    # Get path components: ['patches', 'Drug Name', 'MaybeRepeat', ...]
                    components = [comp for comp in p.parts if comp] # Filter empty parts

                    if len(components) > 1 and components[0].lower() == 'patches':
                        # The first directory *after* 'patches' is the expected one
                        expected_dirs.add(components[1])
                    else:
                         print(f"Warning: Skipping line {line_num}: Path '{path_str}' does not start with 'patches/' or is too short.", file=sys.stderr)

                except Exception as path_e:
                     print(f"Warning: Error parsing path on line {line_num} ('{path_str}'): {path_e}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Labels file not found at '{labels_file_path}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading labels file '{labels_file_path}': {e}", file=sys.stderr)
        return None

    if not expected_dirs:
         print(f"Warning: No valid 'patches/...' paths found in '{labels_file_path}'.", file=sys.stderr)

    return expected_dirs

def get_actual_top_dirs(patches_base_dir):
    """
    Lists the directories directly under the given base directory.
    Returns a set of actual directory names.
    """
    actual_dirs = set()
    try:
        if not os.path.isdir(patches_base_dir):
            print(f"Error: Patches base directory '{patches_base_dir}' not found or is not a directory.", file=sys.stderr)
            return None

        for item in os.listdir(patches_base_dir):
            if os.path.isdir(os.path.join(patches_base_dir, item)):
                actual_dirs.add(item)

    except Exception as e:
        print(f"Error listing directories in '{patches_base_dir}': {e}", file=sys.stderr)
        return None

    return actual_dirs

def main(labels_file_path, patches_base_dir):
    print(f"Checking directory structure in '{patches_base_dir}' against '{labels_file_path}'...")

    expected_dirs = get_expected_top_dirs(labels_file_path)
    if expected_dirs is None:
        sys.exit(1)

    actual_dirs = get_actual_top_dirs(patches_base_dir)
    if actual_dirs is None:
        sys.exit(1)

    # Compare case-insensitively
    expected_dirs_lower = {d.lower() for d in expected_dirs}
    actual_dirs_lower = {d.lower() for d in actual_dirs}

    unexpected_dirs_lower = actual_dirs_lower - expected_dirs_lower
    missing_expected_dirs_lower = expected_dirs_lower - actual_dirs_lower

    found_mismatch = False

    if unexpected_dirs_lower:
        found_mismatch = True
        print("\n[MISMATCH FOUND] The following directories exist but do not match expected names derived from labels.txt:")
        # Map back to original case for reporting
        unexpected_original_case = {d for d in actual_dirs if d.lower() in unexpected_dirs_lower}
        for dir_name in sorted(list(unexpected_original_case)):
            print(f"  - '{dir_name}'")
    else:
        print("\n[OK] All existing directories seem to correspond to names found in labels.txt.")

    if missing_expected_dirs_lower:
         # This is less critical but good to know
         print("\n[INFO] The following expected directory names from labels.txt were NOT found:")
         missing_original_case = {d for d in expected_dirs if d.lower() in missing_expected_dirs_lower}
         for dir_name in sorted(list(missing_original_case)):
             print(f"  - '{dir_name}'")


    if found_mismatch:
        print("\nExpected top-level directory names (derived from labels.txt):")
        for dir_name in sorted(list(expected_dirs)):
             print(f"  - '{dir_name}'")
        print("\nPlease ensure the directory names under 'patches/' exactly match one of the expected names (case might matter depending on subsequent scripts).")
    else:
        print("\nDirectory structure appears consistent with labels.txt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check patch directory names against a labels file.")
    parser.add_argument('--labels_file', type=str, default='./dump/labels.txt',
                        help="Path to the labels.txt file.")
    parser.add_argument('--patches_dir', type=str, default='./patches',
                        help="Path to the base 'patches' directory to check.")
    args = parser.parse_args()

    # Adjust patches_dir if it doesn't exist but ./data_processed_patch/patches does
    if not os.path.exists(args.patches_dir) and os.path.exists('./data_processed_patch/patches'):
        print(f"Info: '{args.patches_dir}' not found, using './data_processed_patch/patches' instead.")
        args.patches_dir = './data_processed_patch/patches'

    main(args.labels_file, args.patches_dir)