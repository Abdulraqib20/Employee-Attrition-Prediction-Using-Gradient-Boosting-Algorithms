# Prediction of Employee Attrition Using Gradient Boosting Algorithms (XGBoost, CatBoost, LightGBM & Ensemble of GBMs)
This repository contains notebooks and code for predicting employee attrition using popular gradient boosting algorithms: XGBoost, CatBoost, LightGBM, and an ensemble of these models using a Voting Classifier.

## Overview
Employee attrition, or employee turnover, is a critical challenge for organizations. Predicting attrition can help businesses take proactive measures to retain valuable employees and maintain a stable workforce. In this project, I explored the use of three powerful gradient boosting algorithms and an ensemble approach to predict employee attrition.

## Gradient Boosting Algorithms

### XGBoost

XGBoost is an efficient and scalable implementation of gradient boosting. It is widely used for its speed and performance in various machine learning competitions.

#### Notebooks
- [**_xgb.ipynb**](https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms/blob/main/notebooks/_xgb.ipynb) <br>
Notebook for training the XGBoost model on the employee attrition dataset. Here, the ordinal encoding was done on the ordinal features, the continuous features were normalized using standard scaling and categorical variables then encoded with one-hot encoding.

- [**_xgb.ipynb**](https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms/blob/main/notebooks/_xgb.ipynb) <br>
Notebook for training the XGBoost model on the employee attrition dataset. Here, only target encoding was applied on the categorical features with other features (non-categorical) left untouched.

### CatBoost

CatBoost is a gradient boosting library that excels in handling categorical features without the need for extensive preprocessing. It is known for its robustness and high performance.

#### Notebooks
- [**_cgb.ipynb**](https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms/blob/main/notebooks/_cgb.ipynb) <br>
Notebook for training the CatBoost model on the employee attrition dataset. Here, the ordinal encoding was done on the ordinal features, the continuous features were normalized using standard scaling and categorical variables then encoded with one-hot encoding.

- [**_cgb_cat.ipynb**](https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms/blob/main/notebooks/_cgb_cat.ipynb) <br>
Notebook for training the CatBoost model on the employee attrition dataset. Here, the ordinal encoding was done on the ordinal features, the continuous features were normalized using standard scaling and categorical variables handled internally by the CatBoost algorithm.

### LightGBM

LightGBM is a gradient boosting framework that is designed for distributed and efficient training. It is known for its speed and ability to handle large datasets.

#### Notebooks
- [**_lgb.ipynb**](https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms/blob/main/notebooks/_lgb.ipynb) <br>
Notebook for training the LightGBM model on the employee attrition dataset. Here, the ordinal encoding was done on the ordinal features, the continuous features were normalized using standard scaling and categorical variables then encoded with one-hot encoding.

### Ensemble Model

Combining all three algorithms using an ensemble to try to improve predictive performance.

#### Notebook
- [**_voting_classifier_ensemble.ipynb**](https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms/blob/main/notebooks/_voting_classifier_ensemble.ipynb) <br>
Notebook for creating an ensemble model using a Voting Classifier. This notebook combines predictions from XGBoost, CatBoost, and LightGBM for improved performance.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Abdulraqib20/Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms.git
    cd Employee-Attrition-Prediction-Using-Gradient-Boosting-Algorithms
    ```

2. Open and run the notebooks in the specified order to train individual models and create an ensemble.

## Results

The results of each model and the ensemble are evaluated using metrics such as ROC AUC scores. The ensemble model aims to combine the strengths of individual models to achieve better predictive performance.

## Conclusion

This project demonstrates the application of gradient boosting algorithms and ensemble methods for predicting employee attrition/turn-over rates.
