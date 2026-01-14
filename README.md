# PAI_Project

# Disaster Image Classification using CNN

## 📌 Project Overview
This project focuses on **image preprocessing, training, and evaluation** of a deep learning model for **disaster image classification**.  
The goal is to classify disaster-related images into their respective categories using **Convolutional Neural Networks (CNNs)**.

The project follows a **complete machine learning pipeline**, starting from dataset inspection and preprocessing to model training and performance evaluation.

---

## 📂 Dataset Description
- **Dataset Name:** Disaster Images Dataset (Comprehensive Disaster Dataset – CDD)
- **Source:** Kaggle
- **Type:** Image Dataset
- **Categories:** Disaster-related classes such as Flood, Fire, Earthquake, Cyclone (depending on dataset structure)

### Dataset Structure

disaster-images-dataset/
└── Comprehensive Disaster Dataset(CDD)/
├── Class_1/
├── Class_2/
└── Class_n/


---

## 🧠 Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - NumPy
  - OpenCV
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - TensorFlow / Keras

---

## ⚙️ Project Pipeline

### 1️⃣ Dataset Inspection
- Verified dataset structure
- Identified class folders
- Handled nested directory structure

---

### 2️⃣ Image Preprocessing
The following preprocessing steps were applied:

1. Image loading with validation  
2. Resizing images to `224 × 224`  
3. Color space conversion (BGR → RGB)  
4. Grayscale conversion (for analysis)  
5. Noise removal using Gaussian Blur  
6. Edge detection using Canny  
7. Normalization (pixel values scaled between 0 and 1)  
8. Data augmentation (rotation, flipping, zooming)  
9. Dataset-wide preprocessing with error handling for corrupted images  

✔ Corrupted and non-image files were automatically skipped.

---

### 3️⃣ Train–Test Split
- Dataset split into **80% training** and **20% testing**
- Stratified sampling used to maintain class balance

---

### 4️⃣ Model Architecture
A **Convolutional Neural Network (CNN)** was designed with:
- Convolution layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax activation for multi-class output

---

### 5️⃣ Model Training
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  
- Metrics: Accuracy  
- Model trained for multiple epochs with validation monitoring

---

### 6️⃣ Model Evaluation
The trained model was evaluated using the following metrics:

- **Accuracy**
- **Confusion Matrix**
- **Precision**
- **Recall**
- **F1-score**
- **ROC Curve**
- **AUC (Area Under Curve)**

ROC–AUC was computed correctly using the **positive class probability** from the softmax output.

---

## 📊 Results & Analysis
- Training and validation accuracy curves were plotted
- Loss curves were analyzed to check overfitting
- Confusion matrix provided class-wise performance
- ROC curve demonstrated the model’s discriminative ability

---

## 📌 Key Learning Outcomes
- Practical understanding of image preprocessing techniques
- Handling real-world dataset issues (corrupted images, nested folders)
- Building and training CNN models
- Correct implementation of evaluation metrics
- Understanding ROC–AUC for classification models

---

## 🚀 How to Run the Project
1. Upload the dataset to Kaggle or Google Colab
2. Ensure the dataset path is correctly set
3. Run the notebook cells sequentially:
   - Preprocessing
   - Training
   - Evaluation
4. View results and performance metrics

---

## 📄 Conclusion
This project demonstrates a **complete end-to-end deep learning workflow** for image classification.  
All essential preprocessing, training, and evaluation steps were successfully implemented, making the project suitable for **academic submission, exams, and viva assessments**.

---

## ✍️ Author
**BS Artificial Intelligence Student**  
Programming for AI – Image Processing Project

---

## 📜 License
This project is for **educational and academic purposes only**.
