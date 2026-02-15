# Spine X-ray Dataset Information

## Overview

This project requires spine X-ray images classified into 5 severity grades (0-4) for training the deep learning model.

## Dataset Structure Required

```
dataset/
├── train/
│   ├── grade_0/  (Healthy)
│   ├── grade_1/  (Doubtful)
│   ├── grade_2/  (Minimal)
│   ├── grade_3/  (Moderate)
│   └── grade_4/  (Severe)
├── val/
│   ├── grade_0/
│   ├── grade_1/
│   ├── grade_2/
│   ├── grade_3/
│   └── grade_4/
└── test/
    ├── grade_0/
    ├── grade_1/
    ├── grade_2/
    ├── grade_3/
    └── grade_4/
```

## Public Datasets for Spine Analysis

### 1. **LSDC (Lumbar Spine Dataset with Degenerative Changes)**

**Description**: Large-scale dataset for lumbar spine degeneration classification

**Source**: 
- GitHub: https://github.com/rwindsor1/LSDC-challenge
- Paper: https://arxiv.org/abs/2104.14128

**Content**:
- Lumbar spine MRI and X-ray images
- Multiple views (sagittal, axial)
- Degenerative changes annotations

**How to Use**:
1. Visit the GitHub repository
2. Follow download instructions
3. Extract and organize into grade folders
4. May need to convert MRI to appropriate format

---

### 2. **SpineWeb Dataset Collection**

**Description**: Comprehensive collection of spine imaging datasets

**Source**: http://spineweb.digitalimaginggroup.ca/

**Content**:
- Multiple spine-related datasets
- X-ray and CT images
- Various pathologies

**Datasets Available**:
- Dataset 1: Anterior-posterior lumbar spine radiographs
- Dataset 2: Lateral lumbar spine radiographs
- Dataset 3: Vertebra segmentation
- Dataset 16: Lumbar spine X-ray images

**How to Access**:
1. Visit SpineWeb website
2. Register for access
3. Download relevant datasets
4. Organize according to severity grades

---

### 3. **Kaggle Spine Datasets**

#### Option A: Spine Fracture Detection
- **Link**: Search "spine fracture" on Kaggle
- **Features**: X-ray images with fracture annotations
- May need relabeling for 5-grade classification

#### Option B: Lumbar Spine Degeneration
- **Link**: Search "lumbar spine degeneration" on Kaggle
- **Features**: Various spine condition classifications

**How to Use Kaggle Datasets**:
```bash
# Install Kaggle CLI
pip install kaggle

# Set up API credentials
# Download your kaggle.json from Kaggle website
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (example)
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip -d dataset/
```

---

### 4. **RSNA Cervical Spine Fracture Detection**

**Description**: Kaggle competition dataset for cervical spine analysis

**Source**: https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection

**Content**:
- CT scans of cervical spine
- Fracture detection labels
- High-quality medical images

**Note**: This is CT data, not X-ray. May need adaptation or use for transfer learning.

---

### 5. **NIH Clinical Center Spine X-ray Dataset**

**Description**: Part of NIH ChestX-ray14 extended collection

**Source**: 
- NIH Clinical Center: https://nihcc.app.box.com/v/ChestXray-NIHCC
- May include spine X-rays in the collection

**How to Access**:
1. Visit NIH Box link
2. Download image archives
3. Filter for spine X-rays
4. Organize by severity

---

## Creating Your Own Dataset

If public datasets don't meet your needs, you can:

### Option 1: Use Synthetic/Augmented Data
- Start with small labeled dataset
- Use heavy augmentation (rotation, flip, brightness, zoom)
- Generate pseudo-labels using pre-trained models

### Option 2: Manual Labeling
- Collect spine X-rays (ensure proper permissions)
- Use tools like LabelImg or CVAT
- Get medical expert validation
- Organize into grade folders

### Option 3: Transfer Learning Approach
- Use pre-trained models from similar tasks
- Fine-tune on small spine dataset
- Progressively improve with more data

---

## Data Preparation Steps

### 1. Download Dataset
Choose one of the datasets above and download

### 2. Organize into Folders
```bash
# Create folder structure
mkdir -p dataset/{train,val,test}/{grade_0,grade_1,grade_2,grade_3,grade_4}

# Move images to appropriate folders based on labels
# Example script:
python organize_dataset.py --input raw_data/ --output dataset/
```

### 3. Split Data
Recommended split ratio:
- Training: 70-80%
- Validation: 10-15%
- Test: 10-15%

```python
# Example Python script
import splitfolders

splitfolders.ratio('raw_dataset', 
                   output='dataset',
                   seed=42, 
                   ratio=(0.7, 0.15, 0.15))
```

### 4. Verify Dataset
Run the data preparation notebook to check:
- Class distribution
- Image quality
- Correct folder structure

---

## Image Requirements

**Format**: JPG, JPEG, or PNG
**Size**: Original size (will be resized to 224x224 during training)
**Orientation**: Consistent (all AP or all lateral)
**Quality**: Good contrast, minimal noise

**Preprocessing Done Automatically**:
- Resizing to 224x224
- Normalization
- Data augmentation (during training)

---

## Dataset Statistics (Example)

For reference, here's a typical distribution:

| Grade | Label    | Train | Val  | Test | Total |
|-------|----------|-------|------|------|-------|
| 0     | Healthy  | 800   | 120  | 80   | 1000  |
| 1     | Doubtful | 600   | 90   | 60   | 750   |
| 2     | Minimal  | 800   | 120  | 80   | 1000  |
| 3     | Moderate | 700   | 105  | 70   | 875   |
| 4     | Severe   | 500   | 75   | 50   | 625   |

**Total**: ~4,250 images (this is just an example)

---

## Important Notes

⚠️ **Medical Data Privacy**
- Ensure all medical images are properly de-identified
- Comply with HIPAA/GDPR regulations
- Get proper permissions before using clinical data

⚠️ **Data Quality**
- Remove duplicate images
- Check for mislabeled data
- Ensure consistent image quality

⚠️ **Ethical Considerations**
- Only use data you have rights to use
- Respect patient privacy
- Get ethical approval if needed for research

---

## Need Help?

If you're having trouble finding or preparing a dataset:

1. **Start Small**: Use 100-200 images per class for initial testing
2. **Use Augmentation**: Heavy data augmentation can help with small datasets
3. **Transfer Learning**: Pre-trained models work well even with limited data
4. **Synthetic Data**: Consider using GANs to generate additional training data

---

## Quick Start Script

```bash
# Create basic folder structure
bash scripts/create_dataset_structure.sh

# Download a sample Kaggle dataset (example)
kaggle datasets download -d <spine-dataset-name>
unzip dataset.zip -d dataset/

# Organize into train/val/test splits
python scripts/split_dataset.py

# Verify dataset
python scripts/verify_dataset.py
```

---

For questions or issues with datasets, please open an issue on GitHub or contact the project maintainer.
