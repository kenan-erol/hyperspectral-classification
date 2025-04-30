import os
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

def create_symlinks(source_base_dir, target_base_dir):
    """
    Creates symbolic links in target_base_dir for all .npy files found
    recursively in source_base_dir, mirroring the directory structure.

    Args:
        source_base_dir (str): The base directory containing the original .npy files.
        target_base_dir (str): The base directory where symbolic links should be created.
    """
    source_path = Path(source_base_dir).resolve()
    target_path = Path(target_base_dir).resolve()

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Ensure the target base directory exists
    try:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured target base directory exists: {target_path}")
    except OSError as e:
        print(f"Error creating target base directory {target_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning source directory: {source_path}")
    print(f"Creating links under target directory: {target_path}")

    # Find all .npy files recursively
    try:
        npy_files = list(source_path.rglob('*.npy'))
        if not npy_files:
            print("Warning: No .npy files found in the source directory.")
            return
    except Exception as e:
        print(f"Error scanning source directory {source_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(npy_files)} .npy files to link.")
    created_links = 0
    skipped_links = 0
    errors = 0

    for original_file_path in tqdm(npy_files, desc="Creating symlinks", unit="file"):
        try:
            # Get the path relative to the source base directory
            relative_path = original_file_path.relative_to(source_path)

            # Construct the full path for the symbolic link
            link_path = target_path / relative_path

            # Create parent directories for the link if they don't exist
            link_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if the link path already exists
            if link_path.exists() or link_path.is_symlink():
                 # Optionally, remove existing link/file before creating new one
                 # print(f"Warning: Target path {link_path} already exists. Removing.")
                 # link_path.unlink(missing_ok=True) # Use missing_ok=True for safety
                 print(f"Warning: Target path {link_path} already exists. Skipping.")
                 skipped_links += 1
                 continue


            # Create the symbolic link (original_file_path is already absolute due to .resolve())
            os.symlink(original_file_path, link_path)
            created_links += 1

        except OSError as e:
            print(f"\nError creating symlink for {original_file_path} -> {link_path}: {e}", file=sys.stderr)
            errors += 1
        except Exception as e:
            print(f"\nUnexpected error processing {original_file_path}: {e}", file=sys.stderr)
            errors += 1

    print("\n--- Symlink Creation Summary ---")
    print(f"Successfully created: {created_links} links")
    print(f"Skipped (already exist): {skipped_links} links")
    print(f"Errors encountered: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symbolic links for .npy files from a source to a target directory, mirroring structure.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help="The source base directory containing the original .npy files (e.g., './data_real_fake')")
    parser.add_argument('--target_dir', type=str, required=True,
                        help="The target base directory where symbolic links will be created (e.g., './data_processed_patch/patches')")

    args = parser.parse_args()

    create_symlinks(args.source_dir, args.target_dir)