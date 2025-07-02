# Kaggle-Competition-2

# 🎓 Academic Performance Prediction

This project aims to predict student academic performance using various machine learning models, including deep learning (Keras) and ensemble techniques (Random Forest, XGBoost, LightGBM). The goal is to classify students into academic categories based on demographic and educational features.

## 📊 Dataset
- Source: Provided as `train.csv.zip` and `test.csv.zip`
- Features include numerical and categorical data related to student background.
- Target: Multi-class academic performance label (3 categories).

## ⚙️ Workflow
1. **Data Preprocessing**
   - Dropped irrelevant columns
   - One-Hot Encoding for target variable
   - Feature scaling using `StandardScaler`
   - Train/validation/test split (stratified)

2. **Modeling**
   - ✅ **Deep Neural Network** (TensorFlow/Keras)
     - Layers: Dense (512–128), Dropout, Batch Normalization
     - Optimizer: Adam | Loss: Categorical Crossentropy
     - Validation Accuracy: **~87%**

   - 📈 **Ensemble Models**
     - **Random Forest:** 86.6% test accuracy
     - **XGBoost:** 85.4% test accuracy
     - **LightGBM:** ✅ **87.2% test accuracy**

3. **Evaluation**
   - Accuracy, loss plots per epoch
   - Used callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
   - Predictions exported for submission

## 📁 Output Files
- `model.weights.keras`: Trained model weights
- `submission.csv`: Predicted labels (LightGBM)
- `submission1.csv`: Predicted labels (Keras DNN)

## 📦 Libraries Used
- Python (Pandas, NumPy, Matplotlib)
- Scikit-learn
- TensorFlow/Keras
- XGBoost, LightGBM

## 🧠 Conclusion
The deep neural network and LightGBM models provided the best predictive performance. This project showcases how combining deep learning and ensemble methods can lead to strong classification results on educational datasets.

---

