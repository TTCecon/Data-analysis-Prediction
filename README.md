# Fund Purchase Prediction Models

This repository contains implementations and analysis of four different machine learning models used for predicting fund purchases. The models included are Lasso Regression, Random Forest, Logistic Regression, and XGBoost. Each model is trained and evaluated to understand its strengths and weaknesses.

## Table of Contents

- [Introduction](#introduction)
- [Models Overview](#models-overview)
  - [Lasso Regression](#lasso-regression)
  - [Random Forest](#random-forest)
  - [Logistic Regression](#logistic-regression)
  - [XGBoost](#xgboost)
- [Implementation Details](#implementation-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Comparison](#results-and-comparison)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)


## Introduction

This project aims to predict which customers are likely to purchase funds using various machine learning models. By comparing the performance of different models, we can identify the most effective approach for this classification problem.

## Models Overview

### Lasso Regression

**Lasso (Least Absolute Shrinkage and Selection Operator) Regression** is a linear regression technique that uses L1 regularization to penalize the absolute size of the coefficients. This leads to some coefficients being shrunk to zero, effectively performing feature selection.

**Advantages:**
- Performs feature selection.
- Reduces model complexity and helps prevent overfitting.
- Simple and interpretable.

**Disadvantages:**
- Assumes linear relationships between features and the target variable.
- Can be sensitive to multicollinearity.

### Random Forest

**Random Forest** is an ensemble learning method that builds multiple decision trees and merges their results to improve accuracy and control overfitting. Each tree in the forest is built from a random subset of the training data and features.

**Advantages:**
- High accuracy and robustness.
- Handles high-dimensional data well.
- Provides feature importance.

**Disadvantages:**
- Requires significant computational resources.
- Less interpretable than linear models.

### Logistic Regression

**Logistic Regression** is a linear model used for binary classification. It estimates the probability of a binary outcome using the logistic function.

**Advantages:**
- Simple and interpretable.
- Efficient to train.
- Outputs probabilities for classification.

**Disadvantages:**
- Assumes a linear relationship between the input features and the log odds of the outcome.
- May underperform with complex relationships in the data.

### XGBoost

**XGBoost (Extreme Gradient Boosting)** is an optimized gradient boosting framework that uses decision trees. It is known for its speed and performance in machine learning competitions.

**Advantages:**
- High predictive accuracy.
- Can handle missing values.
- Incorporates regularization to prevent overfitting.

**Disadvantages:**
- Requires careful parameter tuning.
- Computationally intensive.

## Implementation Details

### Data Preprocessing

Before training the models, the data is preprocessed to handle missing values, normalize features, and split into training and testing sets. Specific steps include:

- Handling missing values.
- Encoding categorical variables.
- Normalizing numerical features.
- Splitting the data into training and testing sets.

### Model Training and Evaluation

Each model is trained on the training set and evaluated on the testing set using the following metrics:

- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix

### Results and Comparison

The performance of each model is compared based on the evaluation metrics. Detailed results and confusion matrices are provided for each model.

### Conclusion

Each model has its strengths and weaknesses. Random Forest and XGBoost showed high accuracy and robustness, while Lasso Regression provided insights into feature importance. Logistic Regression, despite its simplicity, provided a good baseline for comparison.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fund-purchase-prediction.git
   cd fund-purchase-prediction
