# Hyperspectral Image Classification for Drug Identification

This project focuses on using hyperspectral imaging and deep learning techniques to classify different types of pharmaceutical pills based on their spectral signatures.

## Project Goal

The primary objective is to develop and train a classification model capable of accurately identifying drug types from hyperspectral images. This involves:
1.  Organizing hyperspectral data and creating a label map.
2.  Processing hyperspectral data cubes.
3.  Segmenting individual pills within the images, potentially using methods like the Segment Anything Model (SAM).
4.  Extracting representative patches from segmented pills.
5.  Splitting data into training and testing sets based on *unique pills*, not patches, to avoid data leakage.
6.  Training deep learning models (e.g., adapted ResNet, VGG) on these patches.
7.  Evaluating the model's performance in classifying different drug types on the held-out test set.

## Learning Objectives

- Understand the basics of hyperspectral image data (.raw, .hdr files, potentially preprocessed .npy).
- Implement data loading and preprocessing pipelines for hyperspectral images.
- Integrate image segmentation techniques (e.g., SAM2) to isolate objects of interest (pills).
- Develop strategies for sampling patches from segmented objects.
- Implement correct train/test splitting at the image/pill level.
- Adapt standard CNN architectures (like ResNet, VGG) for high-dimensional hyperspectral inputs.
- Implement training and evaluation loops for hyperspectral classification.
- Utilize appropriate loss functions and evaluation metrics for multi-class classification.
- Manage experiments using configuration files and command-line arguments.
- Implement checkpointing and logging (e.g., TensorBoard) for monitoring training progress.

## Data Organization Strategy

The raw data might be organized in nested directories, for example: `./data/<drug_name_date>/Group/M<id>/<filename>.raw/.hdr`. To use this data effectively and consistently across different experiments or data drops:

1.  **Analyze Structure (Optional):** Before preprocessing, understand the directory layout. You can use the `tree` command (install if necessary: `sudo apt install tree` on Debian/Ubuntu, `brew install tree` on macOS) to save the structure to a file:
    ```bash
    # Run from the project root directory
    tree ./data > data_structure.txt
    ```
    Examine `data_structure.txt` to see how files are organized.
2.  **Preprocessing (Optional but Recommended):** Convert the raw hyperspectral data (e.g., ENVI `.raw`/`.hdr` pairs) into a more standard format like NumPy arrays (`.npy`). This simplifies loading. Store these `.npy` files, perhaps mirroring the original structure or flattening it slightly.
3.  **Create `labels.txt`:** Generate a single text file (e.g., `./data/labels.txt`) that acts as a manifest for your dataset. Each line should contain:
    `<relative_path_to_image.npy> <integer_label>`
    -   `<relative_path_to_image.npy>`: The path to an individual pill's hyperspectral data file, relative to the main data directory (e.g., `drop-4/Zopiklon 2025-01-14/Group/M0003/measurement.npy`).
    -   `<integer_label>`: A unique integer representing the drug class (e.g., 0 for Zopiklon, 1 for Aspirin, etc.).
4.  **Scripting:** Create a helper script (e.g., `prepare_data.py`, not provided here) to automate steps 2 and 3. This script would:
    -   Walk through the raw data directories (using the structure identified in step 1).
    -   Identify drug names (e.g., from folder names) and assign integer labels.
    -   Load/convert raw data to `.npy`.
    -   Write the relative paths and labels to `labels.txt`.
5.  **Usage:** The training (`train_classification_hyper.py`) and evaluation (`run_classification_hyper.py`) scripts will then use the `--data_dir` argument pointing to the base directory (e.g., `./data/`) and the `--label_file` argument pointing to the generated `labels.txt`.

This approach decouples the data structure details from the training/evaluation logic and ensures all relevant data is captured.

## Key Files Involved

```
src/train_classification_hyper.py # Main training script for hyperspectral data
src/run_classification_hyper.py   # Evaluation script for hyperspectral data
src/classification_model.py     # Defines the overall classification model structure
src/networks.py                 # Contains encoder architectures (ResNet, VGG)
src/classification_cnn.py       # Contains generic train/evaluate functions
src/net_utils.py                # Utility functions/blocks for networks
src/sam2.py                     # Placeholder for SAM2 integration logic
bash/train_classification_hyper_resnet.sh # Example script to run training
bash/train_classification_hyper_vgg.sh   # Example script to run training
# Potentially add prepare_data.py for data organization
```

## Understanding `train_classification_hyper.py` Arguments

Compared to a standard image classification script (like `train_classification_cnn.py` which might use built-in datasets like CIFAR10), `train_classification_hyper.py` requires more specific arguments due to the nature of the data:

-   `--data_dir`, `--label_file`: Needed because we are loading custom data from a specific structure, not a standard `torchvision` dataset.
-   `--num_patches_per_image`, `--patch_size`: Define the strategy for extracting smaller, fixed-size inputs for the CNN from potentially larger, variable-sized segmented pills.
-   `--train_split_ratio`: Explicitly controls the split between training and testing data based on unique pills.
-   `--num_channels`: Essential for hyperspectral data, as the number of input channels (spectral bands) is much higher than standard RGB (3) or grayscale (1) and must be specified for the model architecture.

## Tasks

- **Data Preparation:**
    - Implement the data organization strategy described above (preprocessing, `labels.txt` generation).
- **Segmentation and Patching:**
    - Ensure the `SAM2` class/functions correctly generate masks or bounding boxes for pills.
    - Verify the patch extraction logic in `HyperspectralPatchDataset` (`_adjust_bbox`, cropping).
- **Dataset Creation & Splitting:**
    - Confirm the `HyperspectralPatchDataset` correctly loads data based on the provided sample list.
    - Verify that `train_test_split` in `train_classification_hyper.py` and `run_classification_hyper.py` correctly splits based on unique images using stratification.
    - Ensure the `DataLoader` uses the `collate_fn_skip_none` function.
- **Model Adaptation:**
    - Double-check that the `ClassificationModel` and encoder networks (`ResNet18Encoder`, `VGGNet11Encoder`) correctly handle the `--num_channels` input.
    - Ensure the final classification layer outputs `num_classes` logits.
- **Training:**
    - Run the training script (`bash/train_*.sh`) using the prepared `labels.txt`.
    - Monitor training using TensorBoard via the `--checkpoint_path`.
- **Evaluation:**
    - Implement or adapt the `evaluate` function in `classification_cnn.py` if needed (e.g., to calculate/save confusion matrix, per-class metrics).
    - Run the evaluation script (`run_classification_hyper.py`) pointing to a trained checkpoint.
- **Experimentation:**
    - Use the bash scripts to run experiments with different hyperparameters.
    - Document results.

## Reporting Results

*(This section can be filled in after running experiments)*

Report the final classification accuracy achieved on the **test set** (derived from the `train_split_ratio`) for different model configurations. Include details like:
- Model Architecture (e.g., ResNet-18, VGG-11 adapted for N channels)
- Number of Input Channels (`--num_channels`)
- Patch Size (`--patch_size`)
- Key Training Hyperparameters (LR, Epochs, Batch Size)
- Overall Test Accuracy
- Per-class Accuracy (if relevant)
- Confusion Matrix (optional but helpful)

Example:
```
Model: ResNet-18 (adapted for 256 channels)
Patch Size: 224x224
Test Accuracy: XX.XX%
```


Notes:

3. Regarding the question about SAM segmentation and variable pill counts
Regarding your question about extracting data from HSI cubes with different numbers of pills (6 vs 100):

You're right to be concerned, but the current implementation handles this gracefully. Let's look at how:

The num_patches_per_image parameter (which you would set to 100) controls how many times each hyperspectral image appears in the dataset, not how many pills it will extract from a single image.

For each appearance of an image in the dataset, the code:

Loads the full hyperspectral cube
Uses SAM2 to detect all pills in that image
Randomly selects ONE pill from the detected pills
Creates a patch around that one pill
For images with only 6 pills:

SAM2 will only detect those 6 pills
When randomly selecting from these 6 pills multiple times (up to 100 times), you'll get repeats
This means you'll end up sampling the same 6 pills multiple times
For images with 100 pills:

SAM2 will detect many more pills
The random selection will have more variety
This approach naturally balances your dataset - cubes with more pills contribute more unique data points, while cubes with fewer pills contribute fewer unique points but still appear the same number of times in your training iterations.

To run these scripts:

First, install the required dependency for processing ENVI files:
Run the conversion script:
Generate the labels file:
After this, you can run your training script as normal, but consider using a higher num_patches_per_image (like 100) to ensure you get good coverage of all pills:
.
The scripts are designed to maintain the directory structure and properly identify drug classes from your data folder structure.

also run pip install -e . in sam2 to install the SAM2 package in editable mode.

in case it stays like blobs

Let's clarify the visualization approaches:

preproc_patch.py Visualization (#attachment=preproc_patch.py "Excerpt from preproc_patch.py, lines 271 to 298"): This script visualizes the entire original hyperspectral image (converted to RGB using hsi_to_rgb_display) and draws a red bounding box showing where the patch was extracted. This is useful for verifying the patch location within the larger image.
augment_patches.py Visualization (#attachment=augment_patches.py "Excerpt from augment_patches.py, lines 138 to 163"): This script visualizes the extracted patch itself before augmentation and the same patch after augmentation (e.g., adding noise), displaying them side-by-side. This is useful for verifying the effect of the augmentation.

makes sense bc zoomed all the way in

if no significant results, increase the transformations