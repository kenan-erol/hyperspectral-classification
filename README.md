# CPSC 290: Hyperspectral Image Classification for Drug Identification

**Student:** Kenan Erol
**Advisor:** Professor Alex Wong, Department of Computer Science
**Director of Undergraduate Studies (DUS):** Professor Theodore Kim

## Project Abstract

Counterfeit pharmaceuticals pose a significant global health risk. These counterfeit drugs are visually indistinguishable from their real counterparts. These similarities lead people to mistakenly take ineffective or even harmful active pharmaceutical ingredients (API), potentially causing serious adverse health effects or death.

This work explores the potential of hyperspectral imaging (HSI), which captures detailed spectral information beyond human vision, combined with deep learning for non destructive pill analysis. HSI provides a unique spectral analysis for materials by measuring reflectance across hundreds of narrow wavelength bands. We developed a pipeline to put such HSI data through preprocessing, segmentation using Segment Anything Model 2 (SAM2), spectral patch extraction, and training of adapted Convolutional Neural Networks (CNNs).

We address two key tasks: multi-class classification of six different drug types, and binary classification distinguishing authentic spectral patches from synthetically generated “fake” patches designed to simulate the spectral differences one might expect. We used datasets acquired by a Lumo spectral imager scanning trays containing 100 pills, producing 256 bands. The drug classes used were bromazolam, ecstasy (MDMA), clonazepam, oxycodone, tramadol, and zopiclone. Our adapted ResNet-18 model achieves 100\% accuracy on the drug classification task. The model achieves 98.77\% accuracy on the real vs fake task when using augmentations targeted at specific bands. Augmentations to bands 0-130 and 81-130 were tried with results becoming more effective based on the strength of the augmentations. These results demonstrate the promise of HSI and deep learning for pharmaceutical analysis, while highlighting the need for further validation with physically acquired counterfeit samples.

## Student Contribution

This project is undertaken as part of CPSC 290. The student's primary contributions include:
*   Developing and implementing the data processing pipeline to convert and organize hyperspectral data.
*   Integrating the pre-trained SAM 2 model for segmenting pills within the hyperspectral images.
*   Designing and implementing the patch extraction strategy from segmented masks.
*   Adapting standard ResNet-18 architecture to accept high-dimensional hyperspectral input.
*   Implementing the training and evaluation loops for both drug type classification and real/fake classification tasks.
*   Conducting experiments, analyzing results, and documenting the project findings.
*   Creating the necessary scripts to automate the workflows.

*(Note: The Segment Anything Model (SAM 2) was developed by Meta AI and is used here as a tool for segmentation. Its development is not part of this project's contribution.)*

## Methodology Overview

1.  **Data Conversion:** Raw hyperspectral data (e.g., ENVI `.raw`/`.hdr`) is converted into NumPy arrays (`.npy`) for easier processing.
2.  **Segmentation:** The SAM 2 model is employed to generate segmentation masks for individual pills within the processed `.npy` images.
3.  **Patch Extraction:** Based on the generated masks, square patches containing representative spectral information for each pill are extracted. Patches are resized to a standard dimension (e.g., 224x224).
4.  **Data Augmentation (for Real/Fake):** To create a dataset for real vs. fake classification, the extracted 'real' patches are augmented (e.g., adding noise, scaling intensity) to simulate 'fake' samples.
5.  **Label Generation:** Text files (`labels*.txt`) are created using the preparation scripts in `tools/`, mapping the relative path of each patch file to its corresponding class label (integer for drug type, or 0/1 for fake/real).
6.  **Model Training:** ResNet-18 modified to handle the high number of spectral channels, are trained on the extracted patches. Separate models are trained for:
    *   Multi-class drug identification.
    *   Binary real vs. fake classification.
7.  **Evaluation:** Trained models are evaluated on a held-out test set of patches to assess accuracy, confusion matrices, and other relevant metrics.

## Key Files Involved
### Data Conversion & Preparation
convert_data.py
-  Converts .raw/.hdr to .npy (Run once initially) 

preproc_patch.py 
- Segments pills using SAM2 and extracts patches from .npy files 

prepare_labels.py 
- Generates labels.txt for multi-class patches (often called by preproc_patch.py) 

augment_patches.py 
- Creates 'fake' data by augmenting real patches 

prep_real_fake.py 
- Generates labels_real_fake.txt for binary classification

### Core Model & Training Logic
datasets.py 
- Defines PyTorch Dataset classes for loading patches 

classification_model.py 
- Defines the overall classification model structure (encoder + decoder) 

networks.py 
- Contains encoder architectures (ResNet, VGG adaptations) 

classification_cnn.py 
- Contains generic train/evaluate functions (used by train_classification_hyper.py) 

train_classification_hyper.py
- Main training script for hyperspectral patch data 

run_classification_hyper.py 
- Evaluation script for hyperspectral patch data

### SAM2 Integration (Used by preproc_patch.py)
sam2/build_sam.py 

- Mainly used for automatic_mask_generator.py which generated the masks necessary for the segmentation

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
	source bash/preproc_hyper.sh
	```
*   Check `data_processed_patch/visualizations/` for examples of segmented bounding boxes.

2.  **Train the Classification Model:**
*   Choose a model (e.g., ResNet) and run the corresponding training script. This script reads patches and labels from `data_processed_patch/`.
*   ```bash
	bash bash/train_classification_hyper_resnet.sh
	```
    *   Monitor training progress. Checkpoints and logs will be saved in `hyper_checkpoints/resnet/` (or `hyper_checkpoints/vgg/`). Check the `transform_viz` subfolder for examples of data augmentation applied during training.

3.  **Evaluate the Model:**
    *   Run the evaluation script, pointing it to the desired trained checkpoint.
*   ```bash
	# Example using a specific ResNet checkpoint (modify path as needed)
	bash bash/run_classification_cnn_hyper_resnet.sh --checkpoint_path 'hyper_checkpoints/resnet/model-XX.pth'
	```
    *   The script will print accuracy and save confusion matrices or other visualizations.

## Workflow B: Real vs. Fake Classification

This workflow trains a model to distinguish between original ('real') hyperspectral patches and artificially augmented ('fake') ones.

1.  **Ensure Patches Exist:** Complete Step 1 of Workflow A (`bash bash/preproc_hyper.sh`) first. You need the original patches.

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

	The arguments used: 

	noise std dev 0.05 scale factor range 0.9 1.1 offset range -0.05 0.05 for weak => guessed all real

	0.25, 0.6 1.4, -0.2 0.2 strong, 98%

	intermediate

	mod bands 0-131 with weak + -0.2 0.2 bc ecstasy study

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

## Acknowledgements

I would like to express my sincere gratitude to my advisor, Professor Alex Wong, for his guidance, support, and valuable insights throughout this project. I also thank Dr. Jolene Bressi and Ian Martin from the Yale School of Public Health and Professor Simon Dunne from the Swedish National Forensic Centre for providing the HSI dataset crucial for this work. I thank Professor Holly Rushmeier for her insights into using HSI data more effectively. This research utilized the Segment Anything Model 2 (SAM2) developed by Meta AI \cite{ravi2024sam2}. Computational resources were provided by the Yale Grace cluster. This project was undertaken as part of the CPSC 290 Directed Research course at Yale University under the supervision of the Director of Undergraduate Studies, Professor Theodore Kim.

## Code and Data Availability

*   The source code for this project is available in this Git repository.
*   All code required to process the data (once obtained) is included in the repository.