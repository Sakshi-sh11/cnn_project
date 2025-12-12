# CNN Emotion Recognition Project

## 🔍 Project Description
This project implements a Convolutional Neural Network (CNN) to recognize human emotions from images.
The project uses the **Facial Emotion Recognition Dataset** available on Kaggle:

- Dataset: [fahadullaha/facial-emotion-recognition-dataset](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset)

The dataset contains preprocessed images grouped into 7 classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise


## 🛠 Features
- CNN-based image classification
- Preprocessing of input images
- Model training and evaluation
- Predict emotions from new images
- ## Installation

1. Clone the repository:

   git clone https://github.com/Sakshi-sh11/cnn_project.git

   cd cnn_project

2. Install the required dependencies:

   pip install torch torchvision matplotlib scikit-learn kagglehub

3. Download the dataset via kagglehub in your script:

   import kagglehub

   path = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset")

4. Run the main training script:

   python cnn_project.py


Model Architecture:

The CNN model consists of:

Feature Extractor:
3 convolutional blocks with Conv2D → ReLU → Conv2D → ReLU → MaxPool 

Number of filters: 32 → 64 → 128

Classifier:
Linear layers with ReLU and Dropout

Output layer size: 7 (number of emotion classes)

The model will train for 15 epochs and save the best model as best_model.pth

Training
Loss Function: Weighted Cross-Entropy Loss to handle class imbalance

Optimizer: Adam

Batch Size: 64

Epochs: 15

Device: GPU if available, else CPU

The script tracks training loss, training accuracy, and test accuracy for each epoch.


