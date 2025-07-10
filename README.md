# Heart Disease Detection using Machine Learning

This project predicts whether a person has heart disease or not using different machine learning models. The goal is to analyze the dataset, apply preprocessing, compare models, and select the best one for deployment.

## 📁 Dataset

The dataset contains 1190 patient records with features like:

- Age
- Cholesterol level
- Resting blood pressure
- Chest pain type
- Fasting blood sugar
- ST slope, etc.

The target column is `target`, where:
- 0 = No heart disease
- 1 = Heart disease present

## ⚙️ Preprocessing

- Handled missing values and outliers (capped extreme cholesterol values)
- Encoded categorical variables using one-hot encoding
- Scaled features where required (for SVM, KNN, Logistic Regression)
- Split data into train and test (80-20 split)

## 🤖 Models Used

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

## 📊 Model Comparison

| Model              | Accuracy | F1-Score | ROC AUC |
|-------------------|----------|----------|---------|
| Logistic Regression | 85.7%    | 87.0%    | 90.7%   |
| KNN                | 88.2%    | 89.5%    | 94.2%   |
| SVM                | 89.1%    | 90.5%    | 94.9%   |
| XGBoost            | 92.8%    | 93.6%    | 97.2%   |
| **Random Forest**  | **95.3%**| **95.8%**| **97.0%**|

> ✅ Random Forest performed the best across all evaluation metrics.

## 🧠 Conclusion

- Random Forest was chosen as the final model for prediction due to its high accuracy and stability.
- The model is saved using `joblib` and ready for deployment in future phases (e.g., with Flask or Streamlit).

## 💻 How to Run

```bash
pip install -r requirements.txt
#
