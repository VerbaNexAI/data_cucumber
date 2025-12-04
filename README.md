# Cucumber Image Dataset Augmentation

This repository contains two Jupyter notebooks for augmenting cucumber image datasets using classical data augmentation techniques and generative AI with StyleGAN2-ADA.

## Overview

The project provides two different approaches to generate additional training data from cucumber images:

1. **Classical Data Augmentation**: Uses traditional image transformation techniques
2. **Generative Augmentation**: Uses StyleGAN2-ADA to generate synthetic cucumber images

## Requirements

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- CUDA-compatible GPU (recommended for StyleGAN2-ADA)

## Installation

Install the required dependencies:

```bash
pip install jupyter numpy pandas opencv-python albumentations==1.4.0
```

For StyleGAN2-ADA notebook, additional requirements will be installed automatically within the notebook:
```bash
pip install torch torchvision ninja
```

## Dataset Structure

The notebooks expect the following dataset structure:

```
data_cucumber_images_jpg/
    class1/
        image1.jpg
        image2.jpg
        ...
    class2/
        image1.jpg
        ...
```

## Usage

### 1. Classical Data Augmentation

File: `dataset-classical-augmentation.ipynb`

This notebook applies traditional image augmentation techniques including:
- Horizontal and vertical flips
- Random rotations
- Brightness and contrast adjustments
- Color transformations (Hue, Saturation, Value)
- Gaussian blur and noise

**Steps:**
1. Open the notebook: `jupyter notebook dataset-classical-augmentation.ipynb`
2. Update the `DATA_ROOT` variable to point to your dataset location
3. Configure augmentation parameters (optional):
   - `N_AUG_PER_IMAGE`: Number of augmented images per original image (default: 3)
4. Run all cells sequentially
5. Output:
   - Augmented images saved in organized folders by class
   - `augmentation_log.csv`: Traceability log of all generated images
   - `cucumber_images_augmented_zip.zip`: Compressed archive of augmented images

### 2. Generative Augmentation with StyleGAN2-ADA

File: `generative-dataset-stylegan2-ada.ipynb`

This notebook trains a StyleGAN2-ADA model to generate synthetic cucumber images.

**Steps:**
1. Open the notebook: `jupyter notebook generative-dataset-stylegan2-ada.ipynb`
2. Update the `DATASET_NAME` and `SUBFOLDER` variables to match your dataset location
3. Run all cells sequentially:
   - Cell 1: Flattens dataset structure for StyleGAN2
   - Cell 2: Resizes images to 128x128 pixels
   - Cell 3: Installs StyleGAN2-ADA dependencies and clones repository
   - Cell 4: Trains the model (adjustable parameters):
     - `--batch`: Batch size (default: 16)
     - `--kimg`: Training duration in thousands of images (default: 200)
     - `--mirror`: Enable horizontal mirroring augmentation
   - Cell 5: Locates the trained model checkpoint
   - Cell 6: Generates synthetic images using the trained model

**Training Parameters:**
- Default training: 200k images (~200 iterations)
- GPU memory required: ~8GB minimum
- Training time: Varies based on GPU (several hours)

**Output:**
- Trained model checkpoints in `/kaggle/working/sg2_runs`
- Generated synthetic images
- Training logs and metrics

## Configuration Options

### Classical Augmentation

Modify the `transform` pipeline in the notebook to adjust augmentation intensity:

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),           # 50% probability
    A.VerticalFlip(p=0.3),             # 30% probability
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    # Add or remove transformations as needed
])
```

### StyleGAN2-ADA Training

Adjust training parameters in Cell 4:

```bash
--batch=16       # Reduce if GPU memory is limited
--kimg=200       # Increase for longer training
--mirror=1       # Use horizontal mirroring (recommended)
--gpus=1         # Number of GPUs to use
```

## Output Files

### Classical Augmentation Output:
- `augmentation_log.csv`: Details of each generated image
- `cucumber_images_augmented_zip.zip`: All augmented images compressed

### StyleGAN2-ADA Output:
- Model checkpoints: `network-snapshot-*.pkl`
- Generated images organized by class
- Training logs and statistics

## Notes

- Both notebooks are designed to run on Kaggle or similar cloud platforms but can be adapted for local execution
- Update all file paths if running locally instead of on Kaggle
- The StyleGAN2-ADA notebook requires significant computational resources
- Generated images maintain the original class structure for easy integration into training pipelines

## Troubleshooting

**Out of Memory Errors (StyleGAN2-ADA):**
- Reduce `--batch` parameter
- Use smaller image resolution (current: 128x128)
- Ensure GPU has at least 8GB VRAM

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For albumentations, use version 1.4.0 specifically

**Dataset Not Found:**
- Verify the `DATA_ROOT` or `data_root` paths are correct
- Check that images are in JPG/JPEG/PNG format
- Ensure directory structure matches expected format
