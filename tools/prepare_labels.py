#!/usr/bin/env python3
import os
import argparse
import re
from collections import defaultdict

def extract_drug_name(path):
    """Extract drug name from folder path and standardize similar drug names"""
    # Try to get the first part of the path that contains a drug name
    path_parts = os.path.normpath(path).split(os.sep)
    
    # Define standardized name mappings for similar drug variations
    # Format: 'standardized_name': ['variation1', 'variation2', ...]
    drug_name_mapping = {
        'zopiklon': ['Zopiklon', 'zopiklon'],
        'tramadol': ['Tramadol', 'tramadol', 'Tradolan', 'tradolan'],
        'ecstasy': ['Ecstasy', 'ecstasy'],
        'clonazepam': ['Klonazepam', 'klonazepam', 'Clonazepam', 'clonazepam'],
        'oxycodone': ['Oxycodone', 'oxycodone', 'oxykodon', 'Oxykodon', 'OxyContin', 'oxycontin'],
        'bromazolam': ['Bromazolam', 'bromazolam']
    }
    
    # Create a flat list of all drug name variations for initial matching
    all_drug_variations = []
    for variations in drug_name_mapping.values():
        all_drug_variations.extend(variations)
    
    # Skip empty trays
    if any('empty tray' in part.lower() for part in path_parts):
        return None
    
    # First check each path part for an exact match with any drug variation
    for part in path_parts:
        for drug_variation in all_drug_variations:
            if drug_variation in part:
                # Map to standardized name
                for standard_name, variations in drug_name_mapping.items():
                    if drug_variation in variations:
                        return standard_name
    
    # If no exact match, try to extract drug name from the first part that contains a date pattern
    date_pattern = re.compile(r'.*(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})')
    for part in path_parts:
        match = date_pattern.match(part)
        if match:
            # Extract the part before the date
            drug_part = part.split(match.group(1))[0].strip()
            # Skip if this is an empty string or likely not a drug name
            if drug_part and not any(x in drug_part.lower() for x in ['empty', 'tray']):
                # Try to match the extracted drug part with known variations
                drug_part_lower = drug_part.lower()
                for standard_name, variations in drug_name_mapping.items():
                    for variation in variations:
                        if variation.lower() in drug_part_lower:
                            return standard_name
                # If no match to known variations, return as is
                return drug_part_lower
    
    # If all else fails, use the parent directory name if it's not "empty tray"
    if len(path_parts) > 1 and not 'empty' in path_parts[-2].lower():
        parent_dir = path_parts[-2].lower()
        # Try to match parent directory with known variations
        for standard_name, variations in drug_name_mapping.items():
            for variation in variations:
                if variation.lower() in parent_dir:
                    return standard_name
        # If no match to known variations, return as is
        return parent_dir
    
    return None

def is_valid_measurement(file_path):
    """Check if this is a valid measurement path (in Group folder, not empty tray)"""
    # Only include files from Group folders
    if not '/Group/' in file_path and not '\\Group\\' in file_path:
        return False
    
    # Skip empty trays
    if 'empty tray' in file_path.lower():
        return False
    
    # Check if it's a measurement.npy file
    if not file_path.endswith('measurement.npy'):
        return False
    
    return True

def create_labels_file(data_dir, output_file="data/labels.txt"):
    """
    Walk through the data_processed directory and create a labels.txt file
    Format: <relative_path_to_image.npy> <integer_label>
    
    Only includes:
    - Files in "Group" folders
    - Skips empty trays
    - Only includes measurement.npy files
    """
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all measurement.npy files
    file_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower() == "measurement.npy":
                # Get path relative to data_dir
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                
                # Only include valid measurements
                if is_valid_measurement(full_path):
                    file_paths.append(rel_path)
    
    # Group by drug class
    class_to_files = defaultdict(list)
    skipped_files = []
    
    for file_path in file_paths:
        drug_name = extract_drug_name(file_path)
        if drug_name:
            class_to_files[drug_name].append(file_path)
        else:
            skipped_files.append(file_path)
    
    # Assign integer labels to each class
    class_to_label = {}
    for i, drug_name in enumerate(sorted(class_to_files.keys())):
        class_to_label[drug_name] = i
    
    # Create mapping for the output file
    output_lines = []
    for drug_name, files in class_to_files.items():
        label = class_to_label[drug_name]
        for file_path in files:
            output_lines.append(f"{file_path} {label}")
    
    # Write labels file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    # Print summary
    print(f"Created labels file at {output_file}")
    print(f"Found {len(file_paths)} measurement files")
    print(f"Included {sum(len(files) for files in class_to_files.values())} files across {len(class_to_files)} classes")
    print(f"Skipped {len(skipped_files)} files (empty trays or unrecognized drug types)")
    
    for drug_name, label in class_to_label.items():
        print(f"  Class {label}: {drug_name} ({len(class_to_files[drug_name])} files)")
    
    return class_to_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labels file for hyperspectral classification")
    parser.add_argument("--data_dir", type=str, default="./data_processed", 
                        help="Processed data directory containing NPY files")
    parser.add_argument("--output", type=str, default="./data/labels.txt", 
                        help="Output path for the labels file")
    
    args = parser.parse_args()
    
    create_labels_file(args.data_dir, args.output)