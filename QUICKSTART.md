# Quick Start Guide - Spine Degeneration Analysis App

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

**Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate spine_dl_app
```

**Using pip**
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

You have 3 options:

**Option A: Use Your Own Dataset**
- Place images in `dataset/train/`, `dataset/val/`, `dataset/test/`
- Organize into folders: `grade_0`, `grade_1`, `grade_2`, `grade_3`, `grade_4`

**Option B: Download Public Dataset**
- See `DATASET_INFO.md` for dataset sources
- Download from Kaggle, SpineWeb, or other sources
- Organize according to folder structure

**Option C: Demo Mode (Quick Test)**
- Use just a few sample images per class
- Good for testing the app functionality

### Step 3: Run the App (Demo Mode)

Even without a trained model, you can test the interface:

```bash
streamlit run app/app.py
```

**Note**: You'll see a warning about missing model file. This is normal!

### Step 4: Train Your Model (Optional)

If you have a dataset ready:

```bash
# Open Jupyter
jupyter notebook

# Run notebooks in order:
# 1. src/01_data_preparation.ipynb
# 2. src/02_model_xception.ipynb
# 3. src/02_ensemble_models.ipynb
```

Training takes 60-90 minutes per model (depending on your hardware).

### Step 5: Use Pre-trained Model (Alternative)

If you don't want to train from scratch:

1. Download a pre-trained Xception model
2. Save it as: `src/models/model_Xception_spine_ft.hdf5`
3. Run the app

---

## 📁 Folder Structure

```
spine_OA_dl_app/
├── app/
│   └── app.py              # Streamlit application
├── src/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_xception.ipynb
│   └── models/             # Save trained models here
├── dataset/                # Your dataset goes here
│   ├── train/
│   ├── val/
│   └── test/
├── requirements.txt
└── README.md
```

---

## 🎯 Usage

1. **Upload X-ray**: Click "Upload image" in sidebar
2. **Predict**: Click "Predict Spine Degeneration Grade" button
3. **View Results**: See grade, confidence, Grad-CAM heatmap

---

## 🔍 What Each File Does

| File | Purpose |
|------|---------|
| `app/app.py` | Main Streamlit web application |
| `01_data_preparation.ipynb` | Load and prepare dataset |
| `02_model_xception.ipynb` | Train Xception model |
| `02_ensemble_models.ipynb` | Combine multiple models |
| `environment.yml` | Conda environment setup |
| `requirements.txt` | Python packages |

---

## ⚠️ Common Issues

**Issue**: "Model file not found"
- **Solution**: Train a model first OR download pre-trained model

**Issue**: "No images found in dataset"
- **Solution**: Check dataset folder structure

**Issue**: "Out of memory during training"
- **Solution**: Reduce batch size in notebooks

**Issue**: "Streamlit not found"
- **Solution**: `pip install streamlit`

---

## 💡 Tips

- **Start small**: Test with 50-100 images per class
- **Use GPU**: Training is much faster with GPU
- **Data quality**: Good quality images = better results
- **Augmentation**: Helps when you have limited data

---

## 📚 Next Steps

1. ✅ Install dependencies
2. ✅ Test the app interface
3. ✅ Prepare your dataset
4. ✅ Train models
5. ✅ Evaluate performance
6. ✅ Deploy for production use

---

## 🤝 Need Help?

- Check `README.md` for detailed documentation
- See `DATASET_INFO.md` for dataset sources
- Open an issue on GitHub for bugs

---

**Ready?** Let's start! 🎉

```bash
streamlit run app/app.py
```
