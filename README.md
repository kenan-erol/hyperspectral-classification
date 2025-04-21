# Hyperspectral Image Classification for Drug Identification

This project focuses on using hyperspectral imaging and deep learning techniques to classify different types of pharmaceutical pills based on their spectral signatures.

## Project Goal

The primary objective is to develop and train a classification model capable of accurately identifying drug types from hyperspectral images. This involves:
1.  Processing hyperspectral data cubes.
2.  Segmenting individual pills within the images, potentially using methods like the Segment Anything Model (SAM).
3.  Extracting representative patches from segmented pills.
4.  Training deep learning models (e.g., adapted ResNet, VGG) on these patches.
5.  Evaluating the model's performance in classifying different drug types.

## Learning Objectives

- Understand the basics of hyperspectral image data (.raw, .hdr files).
- Implement data loading and preprocessing pipelines for hyperspectral images.
- Integrate image segmentation techniques (e.g., SAM2) to isolate objects of interest (pills).
- Develop strategies for sampling patches from segmented objects.
- Adapt standard CNN architectures (like ResNet, VGG) for high-dimensional hyperspectral inputs.
- Implement training and evaluation loops for hyperspectral classification.
- Utilize appropriate loss functions and evaluation metrics for multi-class classification.
- Manage experiments using configuration files and command-line arguments.
- Implement checkpointing and logging (e.g., TensorBoard) for monitoring training progress.

## Key Files Involved

```
src/train_classification_hyper.py # Main training script for hyperspectral data
src/classification_model.py     # Defines the overall classification model structure
src/networks.py                 # Contains encoder architectures (ResNet, VGG)
src/classification_cnn.py       # Contains generic train/evaluate functions (may need adaptation)
src/net_utils.py                # Utility functions/blocks for networks
# Potentially add scripts related to SAM2 integration or specific data loading utilities
bash/train_classification_hyper_resnet.sh # Example script to run training
bash/train_classification_hyper_vgg.sh   # Example script to run training
# Add evaluation scripts if created (e.g., run_classification_hyper.py)
```

## Tasks

- **Data Preparation:**
    - Write functions to load hyperspectral data (e.g., using spectral libraries or custom readers for ENVI format).
    - Understand and utilize metadata from `.hdr` files (wavelengths, dimensions, etc.).
- **Segmentation and Patching:**
    - Integrate SAM2 (or another segmentation method) to generate masks for pills in the hyperspectral images.
    - Implement logic to extract patches from the masked pill regions. Consider strategies like sampling multiple patches per pill or using the bounding box.
- **Dataset Creation:**
    - Create a PyTorch `Dataset` class (`HyperspectralPatchDataset`) to handle loading images, generating/using masks, extracting patches, and applying transformations.
    - Implement a `DataLoader` with appropriate batching and collation, handling potential errors during data loading (e.g., skipping None samples).
- **Model Adaptation:**
    - Modify the `ClassificationModel` and underlying encoder networks (`ResNet18Encoder`, `VGGNet11Encoder`) to accept the correct number of input channels (e.g., 256 based on your data).
    - Ensure the final classification layer in the decoder outputs the correct number of drug classes.
- **Training:**
    - Configure the `train_classification_hyper.py` script to use the new dataset and adapted model.
    - Set up the optimizer (e.g., Adam) and learning rate scheduler.
    - Implement the training loop, including forward pass, loss calculation (e.g., CrossEntropyLoss), backpropagation, and optimizer steps.
    - Integrate TensorBoard logging for loss, accuracy, and potentially image samples.
- **Evaluation:**
    - Implement an evaluation loop to assess model performance on a held-out test set.
    - Calculate relevant metrics (e.g., overall accuracy, per-class accuracy, confusion matrix).
- **Experimentation:**
    - Use the provided bash scripts (or create new ones) to run training experiments with different hyperparameters (learning rate, batch size, epochs, model type).
    - Document the results obtained.

## Reporting Results

*(This section can be filled in after running experiments)*

Report the final classification accuracy achieved on the test set for different model configurations. Include details like:
- Model Architecture (e.g., ResNet-18, VGG-11)
- Key Hyperparameters
- Overall Test Accuracy
- Per-class Accuracy (if relevant)
- Confusion Matrix (optional but helpful)

Example:
```
Model: ResNet-18 (adapted for 256 channels)
Test Accuracy: XX.XX%
```