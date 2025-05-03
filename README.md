# CPSC 290: Hyperspectral Image Classification for Drug Identification

**Student:** [Your Name]
**Advisor:** [Advisor's Name], Department of Computer Science
**Director of Undergraduate Studies (DUS):** [Current DUS Name]

## Project Abstract

This project investigates the application of hyperspectral imaging (HSI) combined with deep learning for the classification of pharmaceutical pills. HSI captures detailed spectral information beyond human vision, potentially enabling non-destructive identification of drug types and differentiation between authentic and counterfeit samples. The project involves developing a pipeline that includes: processing raw HSI data, utilizing the Segment Anything Model (SAM 2) for pill segmentation, extracting representative spectral patches from segmented pills, training adapted Convolutional Neural Network (CNN) models (ResNet-18, VGG-11) for both multi-class drug identification and binary real-vs-fake classification, and evaluating model performance. The goal is to assess the feasibility and effectiveness of HSI and deep learning for rapid drug analysis.

## Student Contribution

This project is undertaken as part of CPSC 290. The student's primary contributions include:
*   Developing and implementing the data processing pipeline to convert and organize hyperspectral data.
*   Integrating the pre-trained SAM 2 model for segmenting pills within the hyperspectral images.
*   Designing and implementing the patch extraction strategy from segmented masks.
*   Adapting standard ResNet-18 and VGG-11 architectures to accept high-dimensional hyperspectral input.
*   Implementing the training and evaluation loops for both drug type classification and real/fake classification tasks.
*   Conducting experiments, analyzing results, and documenting the project findings.
*   Creating the necessary scripts to automate the workflows.

*(Note: The Segment Anything Model (SAM 2) was developed by Meta AI and is used here as a tool for segmentation. Its development is not part of this project's contribution.)*

## Methodology Overview

1.  **Data Conversion:** Raw hyperspectral data (e.g., ENVI `.raw`/`.hdr`) is converted into NumPy arrays (`.npy`) for easier processing.
2.  **Segmentation:** The SAM 2 model is employed to generate segmentation masks for individual pills within the processed `.npy` images.
3.  **Patch Extraction:** Based on the generated masks, square patches containing representative spectral information for each pill are extracted. Patches are resized to a standard dimension (e.g., 224x224).
4.  **Data Augmentation (for Real/Fake):** To create a dataset for real vs. fake classification, the extracted 'real' patches are augmented (e.g., adding noise, scaling intensity) to simulate 'fake' samples.
5.  **Label Generation:** Text files (`labels*.txt`) are created mapping the relative path of each patch file to its corresponding class label (integer for drug type, or 0/1 for fake/real).
6.  **Model Training:** CNN models (ResNet-18, VGG-11), modified to handle the high number of spectral channels, are trained on the extracted patches. Separate models are trained for:
    *   Multi-class drug identification.
    *   Binary real vs. fake classification.
7.  **Evaluation:** Trained models are evaluated on a held-out test set of patches to assess accuracy, confusion matrices, and other relevant metrics.

## Key Files Involved
Data Conversion & Preparation
convert_data.py # Converts .raw/.hdr to .npy (Run once initially) preproc_patch.py # Segments pills using SAM2 and extracts patches from .npy files prepare_labels.py # Generates labels.txt for multi-class patches (often called by preproc_patch.py) augment_patches.py # Creates 'fake' data by augmenting real patches prep_real_fake.py # Generates labels_real_fake.txt for binary classification

Core Model & Training Logic
datasets.py # Defines PyTorch Dataset classes for loading patches classification_model.py # Defines the overall classification model structure (encoder + decoder) networks.py # Contains encoder architectures (ResNet, VGG adaptations) classification_cnn.py # Contains generic train/evaluate functions (used by train_classification_hyper.py) train_classification_hyper.py # Main training script for hyperspectral patch data run_classification_hyper.py # Evaluation script for hyperspectral patch data log_utils.py # Utilities for plotting/logging

SAM2 Integration (Used by preproc_patch.py)
sam2/build_sam.py automatic_mask_generator.py

... other sam2 components ...
Execution Scripts

preproc_hyper.sh # Runs patch extraction augment_patch.sh # Runs data augmentation for fake set prep_rf.sh # Runs label preparation for real/fake set train_classification_hyper_resnet.sh # Example: Train ResNet for drug classification train_classification_hyper_vgg.sh # Example: Train VGG for drug classification train_classification_hyper_rf_resnet.sh # Example: Train ResNet for real/fake train_classification_hyper_rf_vgg.sh # Example: Train VGG for real/fake run_classification_cnn_hyper_resnet.sh # Example: Evaluate ResNet for drug classification run_classification_cnn_hyper_rf_resnet.sh # Example: Evaluate ResNet for real/fake

... other run scripts ...
Output Directories (Examples)
data_processed/ # Stores .npy versions of original images data_processed_patch/ # Stores extracted patches and multi-class labels.txt data_real_fake/ # Stores 'real' (symlinked) and 'fake' (augmented) patches hyper_checkpoints/ # Stores trained model checkpoints and logs


## Deliverables (CPSC 290)

1.  **Source Code:** This Git repository containing all scripts and code developed for the project.
2.  **Trained Models:** Checkpoints (`.pth` files) for the best performing models for both drug classification and real/fake classification, saved in the `hyper_checkpoints/` directory.
3.  **Final Report:** A comprehensive written report detailing the project background, methodology, experiments, results, analysis, and conclusions (submitted as PDF).
4.  **Project Web Page:** A simple HTML page summarizing the project, linking to the code repository and final report, and showcasing key results/visualizations (as required by CPSC 290).
5.  **Electronic Abstract:** A 250-300 word abstract submitted via the CPSC 290 course script.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd hyperspectral-classification
    ```
2.  **Create a Python environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # Ensure you have a requirements.txt file
    # You might need to install PyTorch separately based on your CUDA version:
    # See: https://pytorch.org/get-started/locally/
    ```
4.  **Download SAM 2 Checkpoint:** Place the desired SAM 2 checkpoint file (e.g., `sam2.1_hiera_base_plus.pt`) into the `./sam2/checkpoints/` directory (or adjust paths in scripts). See `sam2/README.md` for download links.
5.  **Prepare Data:** Place your raw hyperspectral data (e.g., in a `./data/` directory). Run `python tools/convert_data.py --src_dir ./data --dest_dir ./data_processed` if you have `.raw`/`.hdr` files.

## Workflow A: Multi-Class Drug Classification

This workflow trains a model to identify the type of drug from a hyperspectral patch.

1.  **Extract Patches using SAM 2:**
    *   Run the preprocessing script. This uses SAM 2 to segment pills in the `.npy` images (from `data_processed/`) and saves extracted patches to `data_processed_patch/patches/`. It also creates a `labels.txt` file in `data_processed_patch/` mapping these patches to their drug class labels.
    *   ```bash
      bash bash/preproc_hyper.sh
      ```
    *   Check `data_processed_patch/visualizations/` for examples of segmented bounding boxes.

2.  **Train the Classification Model:**
    *   Choose a model (e.g., ResNet) and run the corresponding training script. This script reads patches and labels from `data_processed_patch/`.
    *   ```bash
      bash bash/train_classification_hyper_resnet.sh
      # OR
      # bash bash/train_classification_hyper_vgg.sh
      ```
    *   Monitor training progress. Checkpoints and logs will be saved in `hyper_checkpoints/resnet/` (or `hyper_checkpoints/vgg/`). Check the `transform_viz` subfolder for examples of data augmentation applied during training.

3.  **Evaluate the Model:**
    *   Run the evaluation script, pointing it to the desired trained checkpoint.
    *   ```bash
      # Example using a specific ResNet checkpoint (modify path as needed)
      bash bash/run_classification_cnn_hyper_resnet.sh --checkpoint_path 'hyper_checkpoints/resnet/model-XX.pth'
      ```
    *   The script will print accuracy and potentially save confusion matrices or other visualizations.

## Workflow B: Real vs. Fake Classification

This workflow trains a model to distinguish between original ('real') hyperspectral patches and artificially augmented ('fake') ones.

1.  **Ensure Patches Exist:** Complete Step 1 of Workflow A (`bash bash/preproc_hyper.sh`) first. You need the original patches in `data_processed_patch/patches/`.

2.  **Create Augmented ('Fake') Data:**
    *   Run the augmentation script. This reads the 'real' patches, applies augmentations (noise, scaling, offset), and saves the results as 'fake' patches in `data_real_fake/fake/`. It also creates visualizations comparing some real/fake pairs.
    *   ```bash
      bash bash/augment_patch.sh
      ```
    *   *(Note on Bands 81-131 in `augment_patch.sh` example):* **[You need to explain WHY these specific bands were chosen for augmentation. Was it based on prior knowledge, specific spectral features of interest, or an arbitrary choice? E.g., "Bands 81-131 were targeted for augmentation as they correspond to a spectral region known to be sensitive to chemical composition variations for these specific drugs."]**
    *   (Optional) Run `bash bash/compare_rf.sh` (adjust paths inside the script) to visually compare a specific real patch and its generated fake counterpart, including spectral plots.

3.  **Prepare Binary Labels:**
    *   Run the label preparation script. This script scans the `data_real_fake/` directory (containing both real and fake patches - note: the 'real' data might be symlinked or copied here depending on your setup) and creates `labels_real_fake.txt` with binary labels (1 for real, 0 for fake).
    *   ```bash
      bash bash/prep_rf.sh
      ```
    *   *Important:* Ensure your `data_real_fake` directory is correctly populated. The `augment_patches.py` script saves fakes to `data_real_fake/fake`. You might need to manually create `data_real_fake/real` and either copy or symlink the original patches from `data_processed_patch/patches` into it before running `prep_rf.sh`. The `prep_real_fake.py` script expects `real` and `fake` subdirectories.

4.  **Train the Real/Fake Classifier:**
    *   Run the appropriate training script, pointing to the `data_real_fake` directory and the binary label file.
    *   ```bash
      bash bash/train_classification_hyper_rf_resnet.sh
      # OR
      # bash bash/train_classification_hyper_rf_vgg.sh
      ```
    *   Checkpoints and logs will be saved in `hyper_checkpoints/resnet_rf/` (or `vgg_rf`).

5.  **Evaluate the Real/Fake Classifier:**
    *   Run the evaluation script using a trained real/fake checkpoint.
    *   ```bash
      # Example using a specific ResNet_rf checkpoint (modify path as needed)
      bash bash/run_classification_cnn_hyper_rf_resnet.sh --checkpoint_path 'hyper_checkpoints/resnet_rf/model-XX.pth'
      ```
    *   Review the accuracy and confusion matrix for real vs. fake classification.

## Results

*(Placeholder: Summarize key findings, accuracy metrics for both classification tasks, interesting observations, etc. Include figures or link to visualizations if possible.)*

Example:
*   The ResNet-18 model achieved XX.X% accuracy on the multi-class drug identification task.
*   The VGG-11 model achieved YY.Y% accuracy for real vs. fake classification.
*   Confusion matrices indicate confusion between drugs A and B...
*   Augmentations applied to bands 81-131 proved effective/ineffective in creating challenging fake samples...

## Future Work

*(Placeholder: Suggest potential improvements or next steps, e.g., exploring different architectures, more sophisticated augmentation, testing on more diverse data, deploying the model.)*

## Acknowledgements

*   This work utilizes the Segment Anything Model 2 (SAM 2). We acknowledge the contributions of the SAM 2 authors and Meta AI.
    *   Kirillov, A., et al. (2024). *Segment Anything in Images and Videos*. arXiv preprint arXiv:2407.16131. [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
*   This research was conducted under the supervision of [Advisor's Name] for CPSC 290 at Yale University.

## Code and Data Availability

*   The source code for this project is available in this Git repository.
*   The hyperspectral data used in this project is [State data availability: e.g., "available upon reasonable request to the advisor", "located at [link/path]", "proprietary and not publicly available"]. All code required to process the data (once obtained) is included in the repository.