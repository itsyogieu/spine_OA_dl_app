#!/bin/bash

# Script to create dataset folder structure for spine X-ray classification

echo "Creating Spine X-ray Dataset Structure..."
echo "=========================================="

# Create main dataset directories
mkdir -p dataset/{train,val,test}/{grade_0,grade_1,grade_2,grade_3,grade_4}

# Create models directory
mkdir -p src/models

# Create assets directory
mkdir -p assets

echo ""
echo "✅ Folder structure created successfully!"
echo ""
echo "Dataset structure:"
echo "  dataset/"
echo "  ├── train/"
echo "  │   ├── grade_0/ (Healthy)"
echo "  │   ├── grade_1/ (Doubtful)"
echo "  │   ├── grade_2/ (Minimal)"
echo "  │   ├── grade_3/ (Moderate)"
echo "  │   └── grade_4/ (Severe)"
echo "  ├── val/"
echo "  │   └── [same structure as train]"
echo "  └── test/"
echo "      └── [same structure as train]"
echo ""
echo "Next steps:"
echo "1. Download spine X-ray dataset (see DATASET_INFO.md)"
echo "2. Organize images into appropriate grade folders"
echo "3. Run 01_data_preparation.ipynb to verify dataset"
echo ""
