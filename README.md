# 🌸 Flower Classifier using ResNet50 and Keras

This project is a deep learning image classifier built using Keras and TensorFlow. It identifies different types of flowers from images using a fine-tuned **ResNet50** model.

## 📁 Dataset

- Source: [Flowers Recognition Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition)
- Total Images: 4,000+
- Classes:
  - Daisy
  - Dandelion
  - Rose
  - Sunflower
  - Tulip

## 🏗️ Model Architecture

- **Base Model**: `ResNet50` pretrained on ImageNet (without the top layers).
- **Custom Layers**: Global Average Pooling, Dense layers, and Dropout for regularization.
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Input Shape**: (224, 224, 3)

## 🔧 Training Details

- Image augmentation using `ImageDataGenerator`.
- Training set: 80%
- Validation set: 20%
- Batch Size: 32
- Epochs: 10 (can be increased for better accuracy)

## 📈 Results

| Metric              | Value     |
|---------------------|-----------|
| Training Accuracy   | 94.6%     |
| Validation Accuracy | 91.3%     |

## 🌼 Streamlit Web App

A simple and interactive Streamlit app is included so you can:

1. Upload any flower image.
2. Get the predicted flower type instantly.

### ▶️ How to Run the App

```bash
git clone https://github.com/Asmaaelkashef/flower-classifier
cd flower-classifier
pip install -r requirements.txt
streamlit run app.py
