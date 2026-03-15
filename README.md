# WasteAI: Smart Waste Segregation Using Deep Learning

WasteAI is a deep learning project that classifies waste images into 6 categories and serves predictions through a deployable Streamlit web app.

The workflow is split into two parts:

- training and experimentation in Google Colab using a notebook
- inference and deployment through a local or cloud-hosted Streamlit app

## What This Project Does

The model classifies an uploaded waste image into one of these categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

After prediction, the app shows:

- the predicted class
- confidence score
- disposal guidance
- recyclability information
- environmental impact insights

## Current Project Files

```text
.
├── app.py
├── README.md
├── requirements.txt
├── waste_best_model_InceptionV3.h5
└── Waste_Segregation_Complete_Project_A.ipynb
```

### File Roles

- `app.py`: Streamlit application for inference and deployment
- `requirements.txt`: dependencies needed to run the app
- `waste_best_model_InceptionV3.h5`: trained model currently detected by the app
- `Waste_Segregation_Complete_Project_A.ipynb`: notebook used for training, evaluation, and experiments

## Project Highlights

- Waste image classification with deep learning
- 6-class prediction pipeline
- Transfer learning with pretrained CNN backbones
- Deployable Streamlit frontend
- Automatic model discovery from the project folder
- Disposal recommendation logic for each class
- Environmental impact dashboard

## Models Used During Training

The training notebook compares multiple CNN backbones:

| Model | Use Case |
|------|----------|
| MobileNetV2 | Lightweight and fast inference |
| ResNet50 | Strong deep residual baseline |
| VGG16 | Classic benchmark architecture |
| InceptionV3 | Strong multi-scale feature extraction |

Each model uses transfer learning with a custom head:

```text
Base model (frozen)
    -> GlobalAveragePooling2D
    -> BatchNormalization
    -> Dense(256, relu)
    -> Dropout(0.5)
    -> Dense(128, relu)
    -> Dropout(0.3)
    -> Dense(6, softmax)
```

## App Features

The Streamlit app includes 3 main sections.

### 1. Classify Waste

- Upload an image
- Run inference on the trained model
- View class confidence values
- Read disposal steps for the predicted waste type

### 2. Model Insights

- Model family comparison
- Training configuration summary
- Classification-head overview

### 3. Environmental Impact

- CO2 savings by recyclable category
- Water savings by category
- Energy savings by category
- Decomposition timeline comparison

## Dataset Structure Used for Training

Expected dataset layout in Colab:

```text
waste_dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Training configuration used in the notebook:

- Image size: 224 x 224
- Batch size: 32
- Training / validation split: 80 / 20
- Data augmentation enabled for training data

## Tech Stack

### App

- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow
- Plotly

### Training

- Google Colab
- TensorFlow GPU runtime
- ImageDataGenerator augmentation pipeline

## How Model Loading Works

The app searches the project folder for common model names such as:

- `model.h5`
- `waste_model.h5`
- `best_model.h5`
- `waste_best_model_MobileNetV2.h5`
- `waste_best_model_ResNet50.h5`
- `waste_best_model_VGG16.h5`
- `waste_best_model_InceptionV3.h5`
- `waste_finetuned_MobileNetV2.h5`

Your current setup already includes:

- `waste_best_model_InceptionV3.h5`

So the app should load the model automatically.

## Local Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run with the project virtual environment

On this machine, the safest command is:

```bash
c:/Users/asus/OneDrive/Desktop/p/.venv/Scripts/python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

If your virtual environment is already activated, you can also run:

```bash
streamlit run app.py
```

## Requirements

Application dependencies:

```text
streamlit>=1.32.0
tensorflow>=2.10.0
numpy>=1.21.0
Pillow>=9.0.0
plotly>=5.0.0
```

Recommended local baseline:

- Python 3.10+
- 8 GB RAM minimum
- GPU optional for inference
- GPU recommended for training

## Notebook Workflow

The notebook is responsible for:

- environment setup
- Google Drive dataset access
- EDA
- preprocessing and augmentation
- model training
- model comparison
- fine-tuning
- saving trained models

The notebook is the training workspace. The Streamlit app is the deployment workspace.

## Deployment

### Streamlit Community Cloud

Recommended files to upload to GitHub:

```text
app.py
requirements.txt
README.md
model file (.h5)
```

Deployment steps:

1. Push the project to GitHub.
2. Open `https://share.streamlit.io`.
3. Connect your repository.
4. Select `app.py` as the main file.
5. Deploy.

### Important Note About Large Model Files

If the `.h5` file is too large for GitHub or the deployment platform:

- use Git LFS
- host the model externally
- or modify `app.py` to download the model at startup

## Suggested Next Steps

If the app is already working locally, the next practical steps are:

1. Test with several real waste images.
2. Push the deployable files to GitHub.
3. Deploy on Streamlit Community Cloud.
4. Keep the notebook only for retraining or experimentation.

## Possible Future Improvements

- Add prediction history to the UI
- Add Grad-CAM explanation directly inside the app
- Add camera capture support
- Add model download from URL for easier deployment
- Add multilingual disposal recommendations
- Export a smaller inference model if needed

## Summary

This project combines deep learning model training in Colab with a separate, deployable Streamlit interface. That separation is the correct structure for real-world usage: train in the notebook, deploy through the app.
