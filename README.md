# Retinal Vessel Segmentation in Fundus Images

Course project for automatic retinal blood vessel segmentation using deep learning models on fundus images.

## Project Overview

Retinal vessel segmentation is an important medical image analysis task because the retinal vascular structure can support analysis of diseases such as diabetic retinopathy, hypertension, arteriosclerosis, glaucoma, and other ophthalmologic or systemic conditions. Manual vessel annotation is time-consuming and can vary between experts, especially for thin and low-contrast vessels.

This project trains patch-based segmentation models to automatically extract vessel masks from retinal fundus images. DRIVE is used as the primary dataset for training, validation, and internal testing. STARE is used as an external test set to evaluate generalization under dataset shift.

## Datasets

The project uses two public retinal vessel segmentation datasets:

- **DRIVE**
  - 40 color fundus images.
  - Official split: 20 training images and 20 test images.
  - Includes manual vessel annotations and field-of-view masks.
  - In the clean experimental protocol, the official DRIVE training set is split into 15 training images and 5 validation images.

- **STARE**
  - 20 color fundus images.
  - Includes two expert vessel annotations, AH and VK.
  - Used only as an external test set.
  - Field-of-view masks are generated automatically because the local STARE folder does not include official FOV masks.

Expected dataset layout:

```text
archive/
  DRIVE/
    training/
      images/
      1st_manual/
      mask/
    test/
      images/
      1st_manual/
      2nd_manual/
      mask/
  STARE/
    stare-images/
    labels-ah/
    labels-vk/
```

## Methods

The project uses a patch-based supervised segmentation pipeline:

1. Load fundus images, vessel annotations, and FOV masks.
2. Apply preprocessing:
   - RGB input
   - green-channel input
   - green-channel input with CLAHE
3. Extract `128 x 128` training patches from valid retinal regions.
4. Train segmentation models using binary cross-entropy plus Dice loss.
5. Run full-image inference using overlapping patches.
6. Select binarization thresholds on the DRIVE validation set.
7. Evaluate final models on DRIVE test and STARE external test images.

Training augmentation is applied to training patches using random horizontal flips, vertical flips, and rotations by multiples of 90 degrees.

## Model Experiments

The main notebook evaluates five U-Net-based experiments:

| ID | Input | Architecture |
|---|---|---|
| E1 | RGB | U-Net |
| E2 | Green channel | U-Net |
| E3 | Green + CLAHE | U-Net |
| E4 | Green channel | Attention U-Net |
| E5 | Green + CLAHE | Attention U-Net |

A second notebook evaluates a weaker **NoSkip encoder-decoder** model to test the importance of U-Net skip connections:

| ID | Input | Architecture |
|---|---|---|
| M2-E1 | RGB | NoSkip encoder-decoder |
| M2-E2 | Green channel | NoSkip encoder-decoder |
| M2-E3 | Green + CLAHE | NoSkip encoder-decoder |

## Main Results

### DRIVE Test Set

The best DRIVE Dice score is achieved by the Green + CLAHE U-Net, but the top U-Net variants are close.

| Model | Dice |
|---|---:|
| Green + CLAHE U-Net | 0.821 |
| Green Attention U-Net | 0.821 |
| Green U-Net | 0.819 |
| Green + CLAHE Attention U-Net | 0.817 |
| RGB U-Net | 0.813 |

### STARE External Test Set

The Green U-Net generalizes best to STARE.

| Model | STARE AH Dice | STARE VK Dice |
|---|---:|---:|
| Green U-Net | 0.776 | 0.773 |
| Green Attention U-Net | 0.762 | 0.764 |
| Green + CLAHE Attention U-Net | 0.755 | 0.771 |
| Green + CLAHE U-Net | 0.743 | 0.744 |
| RGB U-Net | 0.391 | 0.356 |

The key finding is that green-channel preprocessing greatly improves cross-dataset robustness compared with RGB input.

### Skip Connection Analysis

The NoSkip encoder-decoder models perform substantially worse than the corresponding U-Net models, especially on DRIVE. This supports the importance of skip connections for preserving thin vessel structures.

Approximate DRIVE Dice scores:

| Model | DRIVE Dice |
|---|---:|
| U-Net variants | 0.813-0.821 |
| NoSkip variants | 0.694-0.701 |

## Key Conclusions

- Green-channel preprocessing is the most important improvement for cross-dataset robustness.
- Green + CLAHE U-Net performs best on DRIVE, but Green U-Net is the most reliable final model because it generalizes best to STARE.
- Attention U-Net adds complexity but does not clearly outperform standard U-Net in this project.
- U-Net skip connections are important for thin vessel segmentation; removing them causes a large Dice drop.
- Dice and sensitivity are more informative than accuracy because vessel pixels are a minority class.

## Project Files

```text
retinal_vessel_segmentation_colab.ipynb
  Early Colab notebook with baseline experiments and initial threshold analysis.

retinal_vessel_segmentation_new.ipynb
  Main clean experimental notebook. Use this as the primary project notebook.

retinal_vessel_segmentation_model2.ipynb
  NoSkip encoder-decoder experiments and U-Net vs NoSkip comparison.

models/
  Saved Keras models for the main U-Net and Attention U-Net experiments.

models_model2/
  Saved Keras models for NoSkip experiments.

report_outputs/
  Tables and figures for the main experiments.

report_outputs_model2/
  Tables and figures for the NoSkip experiments.
```

## Important Output Files

Main experiment tables:

```text
report_outputs/table_1_dataset_summary.csv
report_outputs/table_2_drive_test.csv
report_outputs/table_3_stare_external.csv
report_outputs/table_5_postprocessing.csv
report_outputs/table_6_threshold_effect.csv
```

Model 2 comparison tables:

```text
report_outputs_model2/table_2_drive_test_model2.csv
report_outputs_model2/table_3_stare_external_model2.csv
report_outputs_model2/unet_vs_model2_comparison.csv
report_outputs_model2/unet_vs_noskip_delta_table.csv
```

Useful figures:

```text
report_outputs/figure_1_dataset_examples.png
report_outputs/figure_2_preprocessing_comparison.png
report_outputs/figure_3_architecture_overview.png
report_outputs/figure_4_threshold_sweep.png
report_outputs/figure_5_drive_qualitative_comparison.png
report_outputs/figure_6_stare_error_map.png
report_outputs/figure_7a_testset_dice_comparison.png
report_outputs/figure_7b_testset_sensitivity_comparison.png
report_outputs_model2/figure_unet_vs_model2_dice_comparison.png
```

## Reproducing the Experiments

The notebooks were designed to run on Google Colab with GPU acceleration.

1. Upload or mount the project folder in Colab.
2. Ensure the dataset is available under `archive/`.
3. Run `retinal_vessel_segmentation_new.ipynb` for the main U-Net and Attention U-Net experiments.
4. Run `retinal_vessel_segmentation_model2.ipynb` for the NoSkip architecture comparison.
5. Generated tables and figures will be saved under `report_outputs/` and `report_outputs_model2/`.

The notebooks include logic for saving and reusing trained `.keras` models.

## Dependencies

Main Python libraries:

```text
tensorflow
keras
numpy
pandas
matplotlib
opencv-python
Pillow
scikit-learn
```

## Authors

Emirhan Utku and Suleyman Yolcu

AIN 412 Final Project
