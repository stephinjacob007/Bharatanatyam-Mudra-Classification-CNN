# Bharatanatyam Mudra (Hand Gesture) Recognition

## Overview
This project develops a deep learning pipeline for recognizing **28 Bharatanatyam single-hand mudras** using convolutional neural networks. Three approaches are compared:

- Custom CNN (trained from scratch)
- MobileNetV2 (transfer learning)
- EfficientNetB0 (transfer learning)

The best-performing model is further fine-tuned, analyzed with Grad-CAM, and saved for deployment.

## Dataset
- **Dataset:** Bharatanatyam Mudra Dataset
- **Author:** Jisha Raj R.
- **License:** MIT
- **Source:** https://huggingface.co/datasets/Samarth0710/bharatanatyam-mudra-dataset

Only the **single-hand (Asamyukta Hasta)** subset is used.

- Total dataset images: **28,431**
- Single-hand images: **14,827**
- Classes: **28**

## Features

- Dataset download from Hugging Face
- Data exploration
- Data augmentation
- Custom CNN
- MobileNetV2 Transfer Learning
- EfficientNetB0 Transfer Learning
- Model comparison
- Classification report
- Confusion matrix
- Misclassification analysis
- Grad-CAM visualization
- Fine-tuning
- Final model export (.keras)

## Project Pipeline

1. Download dataset
2. Filter single-hand gestures
3. Resize images (128×128)
4. Label encoding
5. Train/Validation/Test split
6. Data augmentation
7. Train three CNN models
8. Compare performance
9. Select best model
10. Analyze predictions
11. Fine-tune best model
12. Save trained model

## Models

### Model A — Custom CNN
- 4 Convolution Blocks
- Batch Normalization
- MaxPooling
- Dropout
- Global Average Pooling
- Dense(256)

Parameters: **0.28 Million**

### Model B — MobileNetV2
ImageNet pretrained backbone with frozen feature extractor.

Parameters: **2.43 Million**

### Model C — EfficientNetB0
ImageNet pretrained EfficientNetB0 backbone.

Parameters: **4.22 Million**

## Results

| Model | Accuracy | Macro F1 |
|--------|----------|----------|
| Custom CNN | 90.47% | 0.905 |
| MobileNetV2 | 64.09% | 0.639 |
| EfficientNetB0 | 71.15% | 0.708 |

### Fine-tuned Custom CNN

| Metric | Score |
|---------|------|
| Test Accuracy | **95.55%** |
| Macro F1 | **0.9563** |

## Explainability

Grad-CAM visualizations highlight image regions contributing most to predictions.

## Repository Structure

```
bharatanatyam-mudra-classification-cnn/
│
├── Bharatanatyam_Mudra_CNN.ipynb
├── README.md
├── requirements.txt
├── saved_model/
└── images/
```

## Installation

```bash
git clone https://github.com/<username>/bharatanatyam-mudra-classification-cnn.git
cd bharatanatyam-mudra-classification-cnn
pip install -r requirements.txt
```

## Running

Open the notebook in Google Colab and enable GPU (Runtime → Change runtime type → T4 GPU).

## Future Improvements

- Train on all 50 classes
- Higher resolution (224×224)
- Webcam inference
- Streamlit/Gradio deployment
- MediaPipe hand landmarks

## Acknowledgements

Dataset by **Jisha Raj R.** under the MIT License.

## License

This repository is released under the MIT License. The dataset retains its original MIT license.
