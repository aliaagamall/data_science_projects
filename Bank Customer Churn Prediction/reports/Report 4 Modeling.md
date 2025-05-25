# Modeling

**Author:** Aliaa Gamal

## Content Structure
The following dictionary outlines the sections of this report:

- [Introduction](#introduction)
- [Model Selection and Training](#model-selection-and-training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Conclusion](#conclusion)

## Introduction
The modeling phase developed a machine learning model to predict customer churn (Exited) using the preprocessed dataset, with a focus on optimizing performance across multiple algorithms.

## Model Selection and Training
- **Models Evaluated**: 
  - Logistic Regression (C=1, max_iter=200).
  - Random Forest (max_depth=None, min_samples_split=2, n_estimators=50).
  - XGBoost (learning_rate=0.01, max_depth=3, n_estimators=50).
- **Training**: Used GridSearchCV for hyperparameter tuning on the training set (6,400 samples).

## Evaluation
- **Model Comparison**:
  | Model              | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
  |--------------------|----------|---------------------|------------------|-------------------|
  | Logistic Regression| 0.998    | 0.994              | 0.994           | 0.994            |
  | Random Forest      | 0.998    | 0.994              | 0.994           | 0.994            |
  | XGBoost            | 0.998    | 0.994              | 0.994           | 0.994            |
  | Logistic Regression (Test) | 1.000 | 1.000            | 1.000           | 1.000            |
- **Test Metrics** (Logistic Regression): Achieved 100% accuracy, precision, recall, and F1-score on the test set.
- **Visualization**: ![Confusion Matrix:](/images/confusion_matrix_test.png)

## Deployment
- **Model Saving**: Saved the Logistic Regression model as model.pkl, the used StandardScaler as scaler.pkl, and deployed the solution using Streamlit.
- **Usage**: Model predicts churn probabilities and labels for new data.

## Conclusion
The modeling process achieved near-perfect performance, with the Logistic Regression model selected for deployment due to its 100% accuracy on the test set.