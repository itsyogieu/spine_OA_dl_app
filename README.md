# Spine Degeneration Analysis with X-ray Images using Deep Convolutional Neural Networks

This project applies deep learning to classify spine X-ray images into 5 severity grades, similar to the knee osteoarthritis analysis approach.

## Spine Degeneration

Spine degeneration is a common condition that occurs due to wear and tear of the vertebrae, intervertebral discs, and surrounding structures. This can lead to various problems including:

- Disc degeneration
- Vertebral body changes
- Spinal stenosis
- Facet joint osteoarthritis

**X-ray imaging** is commonly used to assess the severity of spine degeneration by examining disc space narrowing, bone spurs (osteophytes), and vertebral alignment.

## Severity Grading System

The severity of spine degeneration is classified into 5 levels:

- **Grade 0: Healthy** - Normal spine with no signs of degeneration
- **Grade 1: Doubtful** - Questionable narrowing of disc space, minimal changes
- **Grade 2: Minimal** - Definite narrowing of disc space, possible small osteophytes
- **Grade 3: Moderate** - Moderate disc space narrowing, moderate osteophytes
- **Grade 4: Severe** - Severe disc space narrowing, large osteophytes, possible vertebral deformity

![Spine Severity Grades](assets/spine-grades.png)

## Purpose

The purpose of this project is to correctly classify the severity of spine degeneration based on X-ray images using deep learning models.

![Streamlit App - Spine Analysis](assets/streamlit_spine_ss.png)

## Project Structure

```shell
.
├── README.md
├── app
│   ├── app.py
│   └── img
│       └── spine_icon.png
├── assets
│   ├── Healthy.png
│   ├── Doubtful.png
│   ├── Minimal.png
│   ├── Moderate.png
│   ├── Severe.png
│   ├── confusion_matrix_3_models.png
│   ├── data.png
│   ├── ensemble.png
│   ├── ensemble_test.png
│   └── spine-grades.png
├── dataset
│   ├── test
│   ├── train
│   └── val
├── environment.yml
├── requirements.txt
└── src
    ├── 01_data_preparation.ipynb
    ├── 02_model_xception.ipynb
    ├── 02_model_resnet50.ipynb
    ├── 02_model_inception_resnet_v2.ipynb
    ├── 02_ensemble_models.ipynb
    ├── 03_best_model_on_test.ipynb
    └── models
        └── model_Xception_spine_ft.hdf5
```

## Dataset

This project uses spine X-ray images organized into 5 categories (Grade 0-4).

### Recommended Public Datasets:

1. **LSDC (Lumbar Spine Degeneration Classification)** 
   - Source: [LSDC Challenge](https://github.com/rwindsor1/LSDC-challenge)
   - Contains lumbar spine MRI and X-ray images

2. **SpineWeb Dataset**
   - Source: [SpineWeb](http://spineweb.digitalimaginggroup.ca/)
   - Multiple spine imaging datasets

3. **Kaggle Spine Datasets**
   - Search: "spine x-ray" or "lumbar spine degeneration"
   - Various labeled spine X-ray datasets

### Dataset Structure:

Organize your dataset as follows:
```
dataset/
├── train/
│   ├── grade_0/
│   ├── grade_1/
│   ├── grade_2/
│   ├── grade_3/
│   └── grade_4/
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

## Project Setup

### 1. Clone or Download this repository

```shell
git clone <your-repo-url>
cd spine_OA_dl_app
```

### 2. Configure environment

**Option A: Using Conda**

```shell
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate spine_dl_app
```

**Option B: Using pip**

```shell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Prepare Dataset

- Download spine X-ray dataset (see Dataset section above)
- Organize images into the `dataset/` folder structure
- Ensure images are in standard formats (jpg, png)

### 4. Train Models (Optional)

If you want to train your own models:

```shell
# Open and run notebooks in order:
jupyter notebook src/01_data_preparation.ipynb
jupyter notebook src/02_model_xception.ipynb
jupyter notebook src/02_ensemble_models.ipynb
```

### 5. Download Pre-trained Model

- For demo purposes, you can use transfer learning models
- Save your trained model as: `src/models/model_Xception_spine_ft.hdf5`

### 6. Run the Streamlit App

```shell
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Methodology

### 1. Data Preparation

- Image preprocessing and augmentation
- Class balancing using:
  - Class weights
  - Data augmentation (horizontal flip, rotation, brightness, zoom)
  - Preprocessing features from pre-trained networks

### 2. Model Training

Three pre-trained networks with fine-tuning:
- **Xception**
- **ResNet-50**
- **Inception ResNet v2**

Expected Performance:
- Target Balanced Accuracy: 65-70%
- Training time: 60-90 minutes per model (depending on hardware)

### 3. Ensemble Model

Combine predictions from all three models using weighted averaging for improved accuracy.

### 4. Model Evaluation

- Confusion matrix analysis
- Per-class accuracy metrics
- Grad-CAM visualization for explainability

### 5. Web Application

Streamlit-based interface featuring:
- X-ray image upload
- Automatic preprocessing
- Grade prediction with probability scores
- Grad-CAM heatmap showing relevant regions
- Visual probability distribution

## Key Features

✅ **5-Grade Classification**: Healthy to Severe (Grade 0-4)  
✅ **Deep Learning Models**: Xception, ResNet50, Inception ResNet v2  
✅ **Ensemble Approach**: Combined predictions for better accuracy  
✅ **Explainable AI**: Grad-CAM visualization  
✅ **User-Friendly Interface**: Streamlit web app  
✅ **Production Ready**: Easy deployment  

## Technologies Used

- **Python 3.9**
- **TensorFlow 2.10+** - Deep learning framework
- **Keras** - Neural network API
- **Streamlit** - Web application framework
- **OpenCV / PIL** - Image processing
- **NumPy, Pandas** - Data manipulation
- **Matplotlib, Seaborn** - Visualization
- **scikit-learn** - Metrics and evaluation

## Usage

1. **Upload**: Click "Upload image" and select a spine X-ray
2. **Predict**: Click "Predict Spine Degeneration"
3. **View Results**:
   - Severity grade and confidence percentage
   - Grad-CAM heatmap showing important regions
   - Probability distribution across all grades

## Model Performance

| Model                           | Balanced Accuracy | Training Time |
| ------------------------------- | ----------------- | ------------- |
| Xception fine tuning            | ~67%              | ~70min        |
| ResNet50 fine tuning            | ~65%              | ~80min        |
| Inception ResNet v2 fine tuning | ~64%              | ~60min        |
| Ensemble (weighted average)     | ~69%              | 15sec         |

*Note: Actual performance depends on your dataset quality and size*

## Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions in the X-ray image that most influence the model's prediction:

- **Healthy/Doubtful/Minimal**: Focus on central disc spaces
- **Moderate/Severe**: Focus on vertebral bodies and osteophyte formations

## Deployment Options

### Local Deployment
```shell
streamlit run app/app.py
```

### Docker Deployment
```shell
docker build -t spine-app .
docker run -p 8501:8501 spine-app
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use provided Procfile
- **AWS/GCP/Azure**: Deploy as container or serverless app

## Troubleshooting

**Issue**: Model file not found  
**Solution**: Ensure `model_Xception_spine_ft.hdf5` is in `src/models/`

**Issue**: Out of memory during training  
**Solution**: Reduce batch size in training notebooks

**Issue**: Poor predictions  
**Solution**: Ensure images are properly labeled and dataset is balanced

## Future Improvements

- [ ] Add more pre-trained architectures (EfficientNet, Vision Transformer)
- [ ] Implement multi-view analysis (AP + Lateral views)
- [ ] Add automated report generation
- [ ] Integrate with PACS systems
- [ ] Add segmentation for specific vertebrae
- [ ] Support batch processing
- [ ] Add longitudinal comparison (tracking over time)

## References

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [Inception-v4, Inception-ResNet](https://arxiv.org/abs/1602.07261)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{spine_dl_app,
  title={Spine Degeneration Analysis with Deep Learning},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/spine_OA_dl_app}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Based on knee osteoarthritis analysis methodology
- Inspired by medical imaging research community
- Thanks to public dataset contributors

## Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**⚕️ Medical Disclaimer**: This application is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

---

Made with ❤️ for medical imaging research
