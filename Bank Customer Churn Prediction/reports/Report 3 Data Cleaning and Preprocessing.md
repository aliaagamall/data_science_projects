# Data Cleaning and Preprocessing

**Author:** Aliaa Gamal

## Content Structure
The following dictionary outlines the sections of this report:

- [Introduction](#introduction)
- [Data Cleaning Steps](#data-cleaning-steps)
- [Preprocessing Steps](#preprocessing-steps)
- [Data Splitting and Scaling](#data-splitting-and-scaling)
- [Conclusion](#conclusion)

## Introduction
Data cleaning and preprocessing prepared the [Bank Customer Churn Records](data/Customer-Churn-Records.csv) dataset for modeling by addressing quality issues and transforming features for optimal performance.

## Data Cleaning Steps
- **Missing Values**: No missing values detected.
- **Duplicates**: No duplicates identified.
- **Irrelevant Columns**: Dropped RowNumber, CustomerId, and Surname.
- **Outliers**: Applied Winsorizer with IQR method (fold=1.5) to cap outliers in CreditScore, Age, Tenure, Balance, EstimatedSalary, and Points Earned.

## Preprocessing Steps
- **Feature Transformation**:
  - Log-transformed Age (skew=1.01) using np.log1p.
  - Log-transformed Balance (slight left skew -0.14) using np.log1p.
- **Categorical Encoding**:
  - One-hot encoded Geography, Gender, and Card Type with drop_first=True.
  - Converted Satisfaction Score to categorical codes.

## Data Splitting and Scaling
- **Data Split**: 
  - Initial split: 80% train, 20% test (train_test_split, random_state=42).
  - Test split: 50% validation, 50% test (train_test_split, random_state=42).
  - Shapes: Train (6,400), Validation (800), Test (800).
- **Scaling**:
  - Applied StandardScaler to CreditScore, Age, Tenure, Balance, EstimatedSalary, and Points Earned.
  - Concatenated scaled numerical data with encoded categorical data into final datasets.

## Conclusion
The preprocessing pipeline ensured a clean, transformed, and scaled dataset, ready for robust model training and evaluation.