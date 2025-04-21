#!/usr/bin/env python3
import os
import numpy as np
import spectral.io.envi as envi
import shutil
from tqdm import tqdm
import argparse

def read_hdr_file(hdr_path):
    """Read an ENVI header file and return the metadata dictionary"""
    with open(hdr_path, 'r') as f:
        lines = f.readlines()
    
    metadata = {}
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            metadata[key.strip()] = value.strip()
    
    return metadata

def convert_raw_to_npy(raw_path, hdr_path, output_path):
    """Convert a RAW file to NPY format using its header information"""
    try:
        # Read the hyperspectral data using the spectral library
        img = envi.open(hdr_path, raw_path)
        data = img.load()
        
        # Convert to numpy array
        data_array = np.array(data)
        
        # Save as NPY file
        np.save(output_path, data_array)
        
        return True
    except Exception as e:
        print(f"Error converting {raw_path}: {str(e)}")
        return False

def process_directory(src_dir, dest_dir):
    """Process all measurement.raw files in source directory, maintaining folder structure"""
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all measurement.raw files
    measurement_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower() == "measurement.raw":
                raw_path = os.path.join(root, file)
                hdr_path = os.path.join(root, "measurement.hdr")
                
                if os.path.exists(hdr_path):
                    # Get relative path from src_dir to maintain structure
                    rel_path = os.path.relpath(root, src_dir)
                    dest_path = os.path.join(dest_dir, rel_path)
                    
                    # Create destination directory
                    os.makedirs(dest_path, exist_ok=True)
                    
                    # Full path for output npy file
                    output_path = os.path.join(dest_path, "measurement.npy")
                    
                    measurement_files.append((raw_path, hdr_path, output_path))
    
    # Process all files with progress bar
    print(f"Found {len(measurement_files)} measurements to process")
    
    successful = 0
    for raw_path, hdr_path, output_path in tqdm(measurement_files):
        if convert_raw_to_npy(raw_path, hdr_path, output_path):
            successful += 1
            
            # Also copy the jpg and xml files if they exist
            jpg_path = os.path.join(os.path.dirname(raw_path), "measurement.jpg")
            if os.path.exists(jpg_path):
                shutil.copy2(jpg_path, os.path.join(os.path.dirname(output_path), "measurement.jpg"))
                
            xml_path = os.path.join(os.path.dirname(raw_path), "measurement.xml")
            if os.path.exists(xml_path):
                shutil.copy2(xml_path, os.path.join(os.path.dirname(output_path), "measurement.xml"))
    
    print(f"Successfully converted {successful} out of {len(measurement_files)} measurements")
    return successful

def get_num_bands(data_dir):
    """Find the number of spectral bands by looking at the first .hdr file"""
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdr'):
                hdr_path = os.path.join(root, file)
                metadata = read_hdr_file(hdr_path)
                if 'bands' in metadata:
                    return int(metadata['bands'])
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert hyperspectral data from RAW/HDR to NPY format")
    parser.add_argument("--src_dir", type=str, default="./data", 
                        help="Source directory containing original hyperspectral data")
    parser.add_argument("--dest_dir", type=str, default="./data_processed", 
                        help="Destination directory for processed NPY files")
    
    args = parser.parse_args()
    
    print(f"Converting data from {args.src_dir} to {args.dest_dir}")
    
    # Check for number of spectral bands
    num_bands = get_num_bands(args.src_dir)
    if num_bands:
        print(f"Found {num_bands} spectral bands in the dataset")
    else:
        print("Could not determine number of spectral bands")
    
    # Process all directories
    process_directory(args.src_dir, args.dest_dir)
    
    print("Conversion complete!")