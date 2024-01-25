# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# import libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score, recall_score, precision_score, classification_report, roc_auc_score, confusion_matrix, auc)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand
import random
import sys
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
# import utils

import warnings
warnings.filterwarnings(action='ignore')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def transform_data(data, target_column=None, num_cols=None, cat_cols=None):
    """
    Transforms data by applying Target Encoding to categorical features and preserving numerical features.
    Casts numerical features to int64 or int32, maintaining original floats.
    Handles both training and test sets.

    Args:
        data (pd.DataFrame): The DataFrame to transform (training or test).
        target_column (str, optional): Name of the target column (only for training). Defaults to None.
        num_cols (list, optional): List of numerical feature names. Defaults to None.
        cat_cols (list, optional): List of categorical feature names. Defaults to None.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """

    if target_column is not None:  # Training set
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Fit the transformer on the training set
        full_pipeline = ColumnTransformer(
            transformers=[
                ('numeric', 'passthrough', num_cols),
                ('categorical', Pipeline([
                    ('target encoding', ce.TargetEncoder())
                ]), cat_cols)
            ]
        )
        full_pipeline.fit_transform(X, y)

    else:  # Test set
        X = data
        full_pipeline = ColumnTransformer(
            transformers=[
                ('numeric', 'passthrough', num_cols),
                ('categorical', Pipeline([
                    ('target encoding', ce.TargetEncoder())
                ]), cat_cols)
            ]
        )

    # Transform the data
    X_transformed = full_pipeline.transform(X)

    # Reconstruct DataFrame with correct dtypes
    all_cols = list(num_cols) + list(cat_cols)
    X_transformed = pd.DataFrame(X_transformed, columns=all_cols)

    # Preserve original dtypes for numerical features
    original_dtypes = X[num_cols].dtypes
    X_transformed[num_cols] = X_transformed[num_cols].astype(
        {col: dtype for col, dtype in original_dtypes.items() if dtype != 'float64'}
    ) 

    return X_transformed       


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
skf_seed = 2024
best_k  = 13

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
class ClassifierObjective(object):
    def __init__(self, dtrain, const_params, fold_count, have_gpu):
        self._dtrain = dtrain
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._have_gpu = have_gpu
        self._evaluated_count = 0
        
    # -------------------------------------------------- #
    def _to_xgb_params(self, hyper_params):
        return {
            'n_estimators': int(hyper_params['n_estimators']),
            'max_depth': int(hyper_params['max_depth']),
            'learning_rate': hyper_params['learning_rate'],
            'subsample': hyper_params['subsample'],
            'colsample_bytree': hyper_params['colsample_bytree'],
            'gamma': int(hyper_params['gamma']),
            'min_child_weight': int(hyper_params['min_child_weight']),
            'reg_lambda': hyper_params['reg_lambda'],
            'reg_alpha': hyper_params['reg_alpha'],
            'scale_pos_weight': hyper_params['scale_pos_weight'],
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'tree_method': 'gpu_hist' if self._have_gpu else 'hist',
            'seed': skf_seed,
        }

    # -------------------------------------------------- #
    def __call__(self, hyper_params):
        params = self._to_xgb_params(hyper_params)
        params.update(self._const_params)
        params['tree_method'] = 'gpu_hist' if self._have_gpu else 'hist'

        print(f'Evaluating params={params}', file=sys.stdout)
        mean_aucs = []

        # -------------------------------------------------- #
        # Use a progress bar for feedback
        with tqdm(total=self._fold_count) as pbar:
            for _ in range(self._fold_count):
                cv_result = xgb.cv(
                    params=params,
                    dtrain=self._dtrain,
                    nfold=self._fold_count,
                    seed=skf_seed,
                    metrics='auc',
                    maximize=True,
                    stratified=True
                )
                mean_aucs.append(cv_result['test-auc-mean'].max())
                pbar.update()

        # ----------------------------------------------------------------------- #
        max_mean_auc = np.mean(mean_aucs)
        print(f'Evaluated mean score={max_mean_auc}', file=sys.stdout)

        self._evaluated_count += 1
        print(f'Evaluated {self._evaluated_count} times', file=sys.stdout)

        return {'loss': -max_mean_auc, 'status': STATUS_OK}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def find_best_hyper_params(dtrain, const_params, max_evals=100, have_gpu=False, num_folds=best_k):
    parameter_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 2500, 50),  # Number of boosting rounds
        'max_depth': hp.choice('max_depth', [3,5,7,9,11,13,15,17,20]),  # Maximum depth of a tree, controls the complexity of the model
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1)), # Step size shrinkage used in each boosting update, controls learning rate
        'subsample': hp.uniform('subsample', 0.4, 1.0), # Fraction of samples used for fitting the trees, controls feature randomness
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),  # Fraction of features used for fitting the trees, controls tree randomness
        'gamma': hp.quniform('gamma', 0, 20, 1),   # Minimum loss reduction required to make a further partition on a leaf node
        'min_child_weight': hp.quniform('min_child_weight', 1, 300, 1),   # Minimum sum of instance weight (hessian) needed in a child
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(10.0)),  # L2 regularization term on weights
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(10.0)),  # L1 regularization term on weights
        'scale_pos_weight': hp.uniform('scale_pos_weight', 6.2, 15),  # Controls the balance of positive and negative weights, useful for imbalanced classes
    }

    np.random.seed(seed=skf_seed)

    objective = ClassifierObjective(dtrain=dtrain, const_params=const_params, fold_count=num_folds, have_gpu=have_gpu)
    trials = Trials()
    best = fmin(
        fn=objective,
        space=parameter_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=np.random.seed(seed=skf_seed)
    )

    return best

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
def train_best_model(X, y, const_params, max_evals=100, use_default=False):
    # Create DMatrix for training and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=skf_seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)

    if use_default:
        # Use default hyperparameters
        hyper_params = const_params
    else:
        # Find best hyperparameters
        best = find_best_hyper_params(dtrain, const_params, max_evals=max_evals)
        hyper_params = best.copy()
        hyper_params.update(const_params)
        
        hyper_params['n_estimators'] = int(hyper_params['n_estimators'])
        hyper_params['max_depth'] = int(hyper_params['max_depth'])
        hyper_params['min_child_weight'] = int(hyper_params['min_child_weight'])
        hyper_params['gamma'] = int(hyper_params['gamma'])

        # Print the best hyperparameters
        print("\n\nThe Best Hyperparameters:", hyper_params)

    # Train the model
    model = xgb.train(hyper_params, dtrain, evals=[(dval, 'validation')])

    return model, hyper_params


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Set GPU usage and hyperparameter optimization iterations
have_gpu = False
hyperopt_iterations = 100

# Define constant parameters
const_params = {
    'tree_method': 'gpu_hist' if have_gpu else 'hist', 
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'seed': skf_seed,
}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
-----------# Train, evaluate, and predict with an XGBoost model without cross-validation. #-----------
def train_evaluate_predict_with_xgboost_no_cv(X_train, y_train, X_test, params, early_stopping_rounds=100, random_state=skf_seed):
    """
    Train, evaluate, and predict with an XGBoost model without cross-validation.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target variable.
        X_test (array-like): Test features.
        params (dict): XGBoost model parameters.
        early_stopping_rounds (int, optional): Number of rounds for early stopping. Defaults to 100.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Tuple containing the trained XGBoost model, mean ROC AUC score, standard deviation of ROC AUC score, and predictions for the test set.
    """
    # Function 1: Training and evaluation using learning curves
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=random_state)

    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(**params)

    # Train the model
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    # Retrieve performance metrics
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # Analyze log loss curve to determine optimal early stopping rounds
    logloss_results = results['validation_1']['logloss']
    optimal_rounds_logloss = np.argmin(logloss_results) + 1

    # Analyze AUC curve to determine optimal early stopping rounds
    auc_results = results['validation_1']['auc']
    optimal_rounds_auc = np.argmax(auc_results) + 1

    # Use the minimum of the two optimal rounds as the final early stopping point
    early_stopping_rounds = min(optimal_rounds_logloss, optimal_rounds_auc)

    # Retrain the model with the determined early stopping rounds
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
        early_stopping_rounds=early_stopping_rounds
    )

    # Evaluate with Cross-Validation
    skf = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=random_state)
    cross_val_scores = cross_val_score(xgb_model, X_train, y_train, scoring='roc_auc', cv=skf)
    mean_auc = np.mean(cross_val_scores)
    std_auc = np.std(cross_val_scores)

    print(f"Mean ROC AUC with Cross-Validation: {mean_auc}")
    print(f"Std Dev of ROC AUC with Cross-Validation: {std_auc}")

    # Plot learning Curve
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot log loss
    axs[0].plot(x_axis, results['validation_0']['logloss'], label='Train')
    axs[0].plot(x_axis, results['validation_1']['logloss'], label='Validation')
    axs[0].legend()
    axs[0].set_ylabel('Log Loss')
    axs[0].set_title('XGBoost Log Loss')

    # Indicate optimal early stopping value on the plot
    axs[0].axvline(x=early_stopping_rounds, color='blue', linestyle=':', label=f'Optimal Early Stopping Value: {early_stopping_rounds}')
    axs[0].legend()

    # Plot AUC
    axs[1].plot(x_axis, results['validation_0']['auc'], label='Train')
    axs[1].plot(x_axis, results['validation_1']['auc'], label='Validation')
    axs[1].legend()
    axs[1].set_ylabel('AUC')
    axs[1].set_title('XGBoost AUC')

    # Indicate optimal early stopping value on the plot
    axs[1].axvline(x=early_stopping_rounds, color='blue', linestyle=':', label=f'Optimal Early Stopping Value: {early_stopping_rounds}')
    axs[1].legend()
    plt.show();

    # Function 2: Training and evaluation with metrics calculation
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=early_stopping_rounds)

    # Evaluate on the training set
    train_pred = xgb_model.predict(X_train)
    train_metrics = (
        roc_auc_score(y_train, train_pred),
        recall_score(y_train, train_pred),
        accuracy_score(y_train, train_pred),
        precision_score(y_train, train_pred),
        f1_score(y_train, train_pred),
    )

    # Evaluate on the validation set
    val_pred = xgb_model.predict(X_val)
    val_metrics = (
        roc_auc_score(y_val, val_pred),
        recall_score(y_val, val_pred),
        accuracy_score(y_val, val_pred),
        precision_score(y_val, val_pred),
        f1_score(y_val, val_pred),
    )

    # Display evaluation metrics for Training set
    print("\nTraining Set Metrics:")
    print(f"Mean ROC AUC Train: {train_metrics[0]:.4f}")
    print(f"Mean Recall Train: {train_metrics[1]:.4f}")
    print(f"Mean Accuracy Train: {train_metrics[2]:.4f}")
    print(f"Mean Precision Train: {train_metrics[3]:.4f}")
    print(f"Mean F1 Score Train: {train_metrics[4]:.4f}")

    # Display evaluation metrics for Validation set
    print("\nValidation Set Metrics:")
    print(f"Mean ROC AUC Val: {val_metrics[0]:.4f}")
    print(f"Mean Recall Val: {val_metrics[1]:.4f}")
    print(f"Mean Accuracy Val: {val_metrics[2]:.4f}")
    print(f"Mean Precision Val: {val_metrics[3]:.4f}")
    print(f"Mean F1 Score Val: {val_metrics[4]:.4f}")

    # Function 3: Prediction using the trained model
    
    test_predictions = xgb_model.predict_proba(X_test)[:, 1]

    return xgb_model, mean_auc, std_auc, test_predictions


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
-----------# Trains an XGBoost model, evaluates performance with cross-validation, and predicts on test data. #-----------
def train_evaluate_predict_with_xgboost(X_train, y_train, X_test, params, early_stopping_rounds=100, random_state=skf_seed):
    """
    Trains an XGBoost model, evaluates performance with cross-validation, and predicts on test data.
    Generates validation data from the training set for early stopping.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target variable.
        X_test (array-like): Test features.
        params (dict): XGBoost model parameters.
        early_stopping_rounds (int, optional): Number of rounds for early stopping. Defaults to 100.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Tuple containing the trained XGBoost model, mean ROC AUC score, and standard deviation of ROC AUC score.
        array-like: Predicted probabilities for the test data.
    """
    # Function 1: Training and evaluation with cross-validation
    
    skf = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=random_state)
    mean_auc_list = []
    std_auc_list = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(**params)

        # Train the model
        xgb_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            verbose=False
        )

        # Retrieve performance metrics
        results = xgb_model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)

        # Analyze log loss curve to determine optimal early stopping rounds
        logloss_results = results['validation_1']['logloss']
        optimal_rounds_logloss = np.argmin(logloss_results) + 1

        # Analyze AUC curve to determine optimal early stopping rounds
        auc_results = results['validation_1']['auc']
        optimal_rounds_auc = np.argmax(auc_results) + 1

        # Use the minimum of the two optimal rounds as the final early stopping point
        early_stopping_rounds = min(optimal_rounds_logloss, optimal_rounds_auc)

        # Retrain the model with the determined early stopping rounds
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )

        # Evaluate on the validation set
        y_val_proba = xgb_model.predict_proba(X_val_fold)[:, 1]
        auc_score = roc_auc_score(y_val_fold, y_val_proba)

        mean_auc_list.append(auc_score)

    # Calculate mean and std of AUC scores across folds
    mean_auc = np.mean(mean_auc_list)
    std_auc = np.std(mean_auc_list)

    print(f"Mean ROC AUC with Cross-Validation: {mean_auc}")
    print(f"Std Dev of ROC AUC with Cross-Validation: {std_auc}")

    # Plot learning Curve
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot log loss
    axs[0].plot(x_axis, results['validation_0']['logloss'], label='Train')
    axs[0].plot(x_axis, results['validation_1']['logloss'], label='Validation')
    axs[0].legend()
    axs[0].set_ylabel('Log Loss')
    axs[0].set_title('XGBoost Log Loss')

    # Indicate optimal early stopping value on the plot
    axs[0].axvline(x=early_stopping_rounds, color='blue', linestyle=':', label=f'Optimal Early Stopping Value: {early_stopping_rounds}')
    axs[0].legend()

    # Plot AUC
    axs[1].plot(x_axis, results['validation_0']['auc'], label='Train')
    axs[1].plot(x_axis, results['validation_1']['auc'], label='Validation')
    axs[1].legend()
    axs[1].set_ylabel('AUC')
    axs[1].set_title('XGBoost AUC')

    # Indicate optimal early stopping value on the plot
    axs[1].axvline(x=early_stopping_rounds, color='blue', linestyle=':', label=f'Optimal Early Stopping Value: {early_stopping_rounds}')
    axs[1].legend()
    
    plt.show();

    # Function 2: Training and evaluation with metrics calculation
    
    metrics_train, metrics_val = [], []
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_val, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_val, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=early_stopping_rounds)
        train_pred = xgb_model.predict(X_train_val)  # Train set
        y_pred = xgb_model.predict(X_val)           # Validation set

        # Classification metric calculations
        train_metrics = (
            roc_auc_score(y_train_val, train_pred),
            recall_score(y_train_val, train_pred),
            accuracy_score(y_train_val, train_pred),
            precision_score(y_train_val, train_pred),
            f1_score(y_train_val, train_pred),
        )

        val_metrics = (
            roc_auc_score(y_val, y_pred),
            recall_score(y_val, y_pred),
            accuracy_score(y_val, y_pred),
            precision_score(y_val, y_pred),
            f1_score(y_val, y_pred),
        )

        # Append metrics to lists
        metrics_train.append(train_metrics)
        metrics_val.append(val_metrics)

    # Calculate means of metrics for Training set
    mean_metrics_train = np.mean(metrics_train, axis=0)

    # Calculate means of metrics for Validation set
    mean_metrics_val = np.mean(metrics_val, axis=0)

    # Display evaluation metrics for Training set
    print("\nTraining Set Metrics:")
    print(f"Mean ROC AUC Train: {mean_metrics_train[0]:.4f}")
    print(f"Mean Recall Train: {mean_metrics_train[1]:.4f}")
    print(f"Mean Accuracy Train: {mean_metrics_train[2]:.4f}")
    print(f"Mean Precision Train: {mean_metrics_train[3]:.4f}")
    print(f"Mean F1 Score Train: {mean_metrics_train[4]:.4f}")

    # Display evaluation metrics for Validation set
    print("\nValidation Set Metrics:")
    print(f"Mean ROC AUC Val: {mean_metrics_val[0]:.4f}")
    print(f"Mean Recall Val: {mean_metrics_val[1]:.4f}")
    print(f"Mean Accuracy Val: {mean_metrics_val[2]:.4f}")
    print(f"Mean Precision Val: {mean_metrics_val[3]:.4f}")
    print(f"Mean F1 Score Val: {mean_metrics_val[4]:.4f}")

    # Function 3: Prediction using the trained model
    
    y_pred_list = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Convert data to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
        dtest = xgb.DMatrix(X_test)

        # Create and train the XGBoost model
        xgb_model = xgb.train(
            params,
            dtrain,
            evals=[(dval, 'eval')],
            verbose_eval=False,
            early_stopping_rounds=early_stopping_rounds
        )

        # Predict on the test set
        y_pred_fold = xgb_model.predict(dtest)
        y_pred_list.append(y_pred_fold)

    # Average predictions across folds
    y_pred_avg = np.mean(y_pred_list, axis=0)

    # Return all relevant information
    return xgb_model, mean_auc, std_auc, mean_metrics_train, mean_metrics_val, y_pred_avg