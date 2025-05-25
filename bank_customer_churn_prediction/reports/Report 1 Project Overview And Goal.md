# Bank Customer Churn Prediction Project

**Author:** Aliaa Gamal

## Content Structure
The following dictionary outlines the sections of this report:

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Objective](#project-objective)
- [Project Workflow](#project-workflow)
- [Conclusion](#conclusion)

## Introduction
This project aims to predict whether a bank customer will leave (churn) or stay based on various attributes, using the [Bank Customer Churn Records](data/Customer-Churn-Records.csv) dataset sourced from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data). Customer churn is a critical issue for banks, as retaining existing customers is more cost-effective than acquiring new ones. By identifying key churn drivers, banks can develop targeted loyalty programs and retention strategies.

## Dataset Description
The dataset contains 10,000 records with 18 columns, encompassing demographic, financial, engagement, and feedback metrics. Below is a detailed breakdown:

### Dataset Dictionary

#### Identifiers
- **RowNumber**: Sequential record number.
- **CustomerId**: Unique customer identifier.
- **Surname**: Customer's last name.

#### Demographic Info
- **Geography**: Country (France, Germany, Spain).
- **Gender**: Customer gender (Male/Female).
- **Age**: Customer age (numeric).

#### Financial Info
- **CreditScore**: Credit score (300–850).
- **Balance**: Account balance (numeric).
- **EstimatedSalary**: Estimated annual salary (numeric).

#### Customer Engagement
- **Tenure**: Years with the bank (numeric).
- **NumOfProducts**: Number of bank products (1–4).
- **HasCrCard**: Credit card ownership (0 = No, 1 = Yes).
- **IsActiveMember**: Active status (0 = No, 1 = Yes).
- **Card Type**: Credit card type (e.g., Visa, MasterCard).
- **Points Earned**: Points from credit card usage (numeric).

#### Customer Feedback
- **Complain**: Complaint status (0 = No, 1 = Yes).
- **Satisfaction Score**: Complaint resolution score (1–5).

#### Target Variable
- **Exited**: Churn status (0 = Stayed, 1 = Churned).

#### Key Statistics
- **Rows**: 10,000  
- **Churn Rate**: 20.4% (2,038 customers churned)

## Project Objective
The primary goal is to build a machine learning model to accurately predict customer churn (Exited) and identify key factors (e.g., complaints, age, balance) driving it. This enables proactive retention measures, such as enhanced complaint resolution or tailored services.

## Project Workflow
The project follows a structured pipeline:
1. **Dataset Exploration** – Understand the data structure and variable meanings.
2. **EDA** – Detect trends, patterns, and issues in the data.
3. **Data Preprocessing** – Resolve issues identified in the EDA phase.
4. **Modeling** – Train and evaluate models such as Logistic Regression, Random Forest, and XGBoost.
5. **Evaluation and Insights** – Measure performance using accuracy, precision, recall, and F1-score, and translate results into actionable recommendations.

## Conclusion
This report establishes the foundation for predicting bank customer churn, leveraging a comprehensive dataset and a systematic workflow. By addressing data quality issues early and applying robust models, we aim to uncover meaningful insights and support data-driven retention strategies.