import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def regularized_regression(X_train, y_train, model_type='ridge', lambdas=None, cv=10):
    """
    Perform regularized regression using Ridge or Lasso with cross-validation.

    Parameters:
    - X_train: ndarray, the training features
    - y_train: ndarray, the target variable for training
    - model_type: str, 'ridge' for Ridge regression (L2) or 'lasso' for Lasso regression (L1)
    - lambdas: list or array, range of lambda (alpha) values to test
    - cv: int, the number of folds in cross-validation

    Returns:
    - best_model: the best Ridge or Lasso model fitted with the best lambda
    - best_lambda: the lambda value that gave the best cross-validation score
    """
    if lambdas is None:
        # Define default range for lambda values if none are provided
        lambdas = np.logspace(-4, 4, 10)
    
    best_score = -np.inf  # Initialize to a very low score
    best_lambda = None
    best_model = None
    
    # Select model type
    Model = Ridge if model_type == 'ridge' else Lasso
    
    # Iterate through each lambda value and perform cross-validation
    for lam in lambdas:
        model = Model(alpha=lam)
        # Using negative mean squared error for scoring
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores)
        
        # Update best model if current model's score is better
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam
            best_model = model
    
    # Fit the best model on the full training data
    best_model.fit(X_train, y_train)
    
    print(f"Best lambda for {model_type.capitalize()}: {best_lambda}")
    print(f"Cross-validated score: {best_score}")
    
    return best_model, best_lambda

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

def estimate_generalization_error(X, y, model_type='ridge', lambdas=None, cv=10):
    """
    Estimate the generalization error for different lambda values using K-fold cross-validation.

    Parameters:
    - X: ndarray, the feature matrix.
    - y: ndarray, the target variable.
    - model_type: str, 'ridge' for Ridge regression (L2) or 'lasso' for Lasso regression (L1).
    - lambdas: list or array, range of lambda (alpha) values to test. If None, default range is used.
    - cv: int, number of folds in cross-validation (default is 10).

    Returns:
    - lambda_values: List of lambda values tested.
    - generalization_errors: List of cross-validated errors for each lambda.
    - Plot of the generalization error as a function of lambda.
    """
    if lambdas is None:
        # Default lambda values in a range where generalization error may drop and then increase
        lambdas = np.logspace(-4, 4, 10)
    
    # Select model type
    Model = Ridge if model_type == 'ridge' else Lasso
    
    # Lists to store lambda values and corresponding generalization errors
    generalization_errors = []

    for lam in lambdas:
        model = Model(alpha=lam)
        # Using negative mean squared error for scoring, so we take the negative of the scores to get MSE
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        generalization_errors.append(-np.mean(scores))  # Convert to positive MSE
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, generalization_errors, marker='o', color='orange', label="Estimated Generalization Error")
    plt.xscale('log')
    plt.xlabel("Regularization Parameter λ")
    plt.ylabel("Estimated Generalization Error")
    plt.title("Generalization Error as a Function of λ")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print results
    print(f"Best lambda: {lambdas[np.argmin(generalization_errors)]}")
    print(f"Minimum Generalization Error: {min(generalization_errors)}")

    return lambdas, generalization_errors

def plot_training_vs_generalization_error(X_train, y_train, model_type='ridge', lambdas=None, cv=10):
    """
    Plot training error and estimated generalization error as a function of lambda.

    Parameters:
    - X_train: ndarray, the training feature matrix.
    - y_train: ndarray, the training target variable.
    - model_type: str, 'ridge' for Ridge regression (L2) or 'lasso' for Lasso regression (L1).
    - lambdas: list or array, range of lambda (alpha) values to test. If None, default range is used.
    - cv: int, number of folds in cross-validation (default is 10).

    Returns:
    - lambda_values: List of lambda values tested.
    - training_errors: List of training errors for each lambda.
    - generalization_errors: List of cross-validated errors for each lambda.
    - A plot showing both training and generalization errors as functions of lambda.
    """
    if lambdas is None:
        # Define a reasonable range of lambda values (logarithmic scale)
        lambdas = np.logspace(-4, 4, 10)
    
    # Select model type
    Model = Ridge if model_type == 'ridge' else Lasso
    
    # Lists to store errors for each lambda
    training_errors = []
    generalization_errors = []

    for lam in lambdas:
        model = Model(alpha=lam)
        # Fit the model on the training data
        model.fit(X_train, y_train)
        
        # Compute training error (MSE on training data)
        y_train_pred = model.predict(X_train)
        training_mse = mean_squared_error(y_train, y_train_pred)
        training_errors.append(training_mse)
        
        # Compute generalization error (cross-validated MSE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        generalization_errors.append(-np.mean(cv_scores))  # Convert to positive MSE
    
    # Plotting training and generalization error
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, training_errors, marker='o', label="Training Error", color='blue')
    plt.plot(lambdas, generalization_errors, marker='o', label="Estimated Generalization Error", color='orange')
    plt.xscale('log')
    plt.xlabel("Regularization Parameter λ")
    plt.ylabel("Error")
    plt.title("Training vs Generalization Error as a Function of λ")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print best lambda based on generalization error
    best_lambda = lambdas[np.argmin(generalization_errors)]
    print(f"Best lambda based on generalization error: {best_lambda}")
    
    return lambdas, training_errors, generalization_errors

def cross_validation_model_selection(X, y, model_type='ridge', lambdas=None, K=10):
    """
    Perform K-fold cross-validation to select the best lambda for regularization.

    Parameters:
    - X: DataFrame or ndarray, the feature matrix.
    - y: DataFrame, Series, or ndarray, the target variable.
    - model_type: str, 'ridge' for Ridge regression (L2) or 'lasso' for Lasso regression (L1).
    - lambdas: list or array, range of lambda (alpha) values to test. If None, a default range is used.
    - K: int, the number of folds in cross-validation.

    Returns:
    - lambdas: List of lambda values tested.
    - training_errors: List of average training errors for each lambda.
    - generalization_errors: List of average generalization errors for each lambda.
    - A plot showing both training and generalization errors as functions of lambda.
    """
    if lambdas is None:
        lambdas = np.logspace(-1, 1, 10)
    
    # Choose model class based on type
    Model = Ridge if model_type == 'ridge' else Lasso
    
    # To store training and generalization errors for each lambda
    training_errors = []
    generalization_errors = []
    
    # K-fold cross-validation
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    for lam in lambdas:
        train_error_sum = 0
        gen_error_sum = 0
        
        for train_index, test_index in kf.split(X):
            # Use .iloc if X is a DataFrame, otherwise use regular indexing
            if hasattr(X, 'iloc'):
                X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train_fold, X_test_fold = X[train_index], X[test_index]
            
            # Apply .iloc to y if it's a DataFrame or Series
            if hasattr(y, 'iloc'):
                y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            else:
                y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            # Instantiate and fit the model for this fold
            model = Model(alpha=lam)
            model.fit(X_train_fold, y_train_fold)
            
            # Calculate training error
            y_train_pred = model.predict(X_train_fold)
            train_error = mean_squared_error(y_train_fold, y_train_pred)
            train_error_sum += train_error
            
            # Calculate test (generalization) error
            y_test_pred = model.predict(X_test_fold)
            test_error = mean_squared_error(y_test_fold, y_test_pred)
            gen_error_sum += test_error
        
        # Average errors over all folds
        training_errors.append(train_error_sum / K)
        generalization_errors.append(gen_error_sum / K)
    
    # Plot training and generalization error
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, training_errors, marker='o', color='blue', label="Training Error")
    plt.plot(lambdas, generalization_errors, marker='o', color='orange', label="Estimated Generalization Error")
    plt.xscale('log')
    plt.xlabel("Regularization Parameter λ")
    plt.ylabel("Error")
    plt.title("Training vs Generalization Error as a Function of λ")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Identify best lambda based on minimum generalization error
    best_lambda = lambdas[np.argmin(generalization_errors)]
    print(f"Best lambda based on generalization error: {best_lambda}")
    print(f"Minimum Generalization Error: {min(generalization_errors)}")
    
    return lambdas, training_errors, generalization_errors

def cross_validation_model_selection2(X, y, model_type='ridge', lambdas=None, K=10):
    """
    Perform K-fold cross-validation to select the best lambda for regularization.

    Parameters:
    - X: DataFrame or ndarray, the feature matrix.
    - y: DataFrame, Series, or ndarray, the target variable.
    - model_type: str, 'ridge' for Ridge regression (L2) or 'lasso' for Lasso regression (L1).
    - lambdas: list or array, range of lambda (alpha) values to test. If None, a smaller range is used.
    - K: int, the number of folds in cross-validation.

    Returns:
    - lambdas: List of lambda values tested.
    - training_errors: List of average training errors for each lambda.
    - generalization_errors: List of average generalization errors for each lambda.
    - A plot showing both training and generalization errors as functions of lambda.
    """
    if lambdas is None:
        # Use a narrower range of lambda values, similar to what might be in Fig. 10.9
        lambdas = np.logspace(-3, 1, 3)
    
    Model = Ridge if model_type == 'ridge' else Lasso
    training_errors = []
    generalization_errors = []
    
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    for lam in lambdas:
        train_error_sum = 0
        gen_error_sum = 0
        
        for train_index, test_index in kf.split(X):
            if hasattr(X, 'iloc'):
                X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train_fold, X_test_fold = X[train_index], X[test_index]
            
            if hasattr(y, 'iloc'):
                y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            else:
                y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            model = Model(alpha=lam)
            model.fit(X_train_fold, y_train_fold)
            
            y_train_pred = model.predict(X_train_fold)
            train_error = mean_squared_error(y_train_fold, y_train_pred) / len(y_train_fold)
            train_error_sum += train_error
            
            y_test_pred = model.predict(X_test_fold)
            test_error = mean_squared_error(y_test_fold, y_test_pred) / len(y_test_fold)
            gen_error_sum += test_error
        
        training_errors.append(train_error_sum / K)
        generalization_errors.append(gen_error_sum / K)
    
    # Plot training and generalization error
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(lambdas) + 1), training_errors, marker='o', color='blue', label="Training Error")
    plt.plot(range(1, len(lambdas) + 1), generalization_errors, marker='o', color='orange', label="Estimated Generalization Error")
    plt.xlabel("Model index $M_s$")
    plt.ylabel("Error")
    plt.title("Training vs Generalization Error for Different Models (Lambda Values)")
    plt.xticks(range(1, len(lambdas) + 1), labels=[f"{i+1}" for i in range(len(lambdas))])
    plt.legend()
    plt.grid(True)
    plt.show()
    
    best_lambda = lambdas[np.argmin(generalization_errors)]
    print(f"Best lambda based on generalization error: {best_lambda}")
    print(f"Minimum Generalization Error: {min(generalization_errors)}")
    
    return lambdas, training_errors, generalization_errors





# Baseline Model Class
class BaselineModel:
    def fit(self, X, y):
        self.mean_y = np.mean(y)
        return self
    
    def predict(self, X):
        return np.full(X.shape[0], self.mean_y)

# ANN Model in PyTorch
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(ANNModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Train and evaluate the ANN model
def train_ann(X_train, y_train, X_val, y_val, hidden_units, epochs=100, learning_rate=0.01):
    input_dim = X_train.shape[1]
    model = ANNModel(input_dim, hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor).numpy().flatten()
    return mean_squared_error(y_val, y_val_pred), model

# Two-Level Cross-Validation
def two_level_cross_validation(X, y, lambdas, hidden_units_list, K1=10, K2=10):
    outer_kf = KFold(n_splits=K1, shuffle=True, random_state=42)
    outer_results = {'Baseline': [], 'Ridge': [], 'ANN': []}

    for train_index, test_index in outer_kf.split(X):
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        # Inner Cross-Validation for Ridge
        inner_kf = KFold(n_splits=K2, shuffle=True, random_state=42)
        best_ridge_lambda = None
        best_ridge_mse = float('inf')

        for lam in lambdas:
            ridge_mse = 0
            for inner_train_index, inner_val_index in inner_kf.split(X_train_outer):
                X_train_inner, X_val_inner = X_train_outer[inner_train_index], X_train_outer[inner_val_index]
                y_train_inner, y_val_inner = y_train_outer[inner_train_index], y_train_outer[inner_val_index]
                
                ridge_model = Ridge(alpha=lam)
                ridge_model.fit(X_train_inner, y_train_inner)
                y_val_pred = ridge_model.predict(X_val_inner)
                ridge_mse += mean_squared_error(y_val_inner, y_val_pred)
            
            ridge_mse /= K2
            if ridge_mse < best_ridge_mse:
                best_ridge_mse = ridge_mse
                best_ridge_lambda = lam

        # Train final Ridge model with best lambda on the outer train set and evaluate on the outer test set
        ridge_model = Ridge(alpha=best_ridge_lambda)
        ridge_model.fit(X_train_outer, y_train_outer)
        y_test_pred = ridge_model.predict(X_test_outer)
        outer_results['Ridge'].append(mean_squared_error(y_test_outer, y_test_pred))

        # Baseline Model Evaluation
        baseline_model = BaselineModel()
        baseline_model.fit(X_train_outer, y_train_outer)
        y_test_pred_baseline = baseline_model.predict(X_test_outer)
        outer_results['Baseline'].append(mean_squared_error(y_test_outer, y_test_pred_baseline))

        # Inner Cross-Validation for ANN
        best_ann_units = None
        best_ann_mse = float('inf')
        
        for units in hidden_units_list:
            ann_mse = 0
            for inner_train_index, inner_val_index in inner_kf.split(X_train_outer):
                X_train_inner, X_val_inner = X_train_outer[inner_train_index], X_train_outer[inner_val_index]
                y_train_inner, y_val_inner = y_train_outer[inner_train_index], y_train_outer[inner_val_index]

                mse, _ = train_ann(X_train_inner, y_train_inner, X_val_inner, y_val_inner, hidden_units=units)
                ann_mse += mse
            
            ann_mse /= K2
            if ann_mse < best_ann_mse:
                best_ann_mse = ann_mse
                best_ann_units = units

        # Train final ANN with best hidden units on the outer train set and evaluate on the outer test set
        _, best_ann_model = train_ann(X_train_outer, y_train_outer, X_test_outer, y_test_outer, hidden_units=best_ann_units)
        X_test_tensor = torch.tensor(X_test_outer, dtype=torch.float32)
        with torch.no_grad():
            y_test_pred_ann = best_ann_model(X_test_tensor).numpy().flatten()
        outer_results['ANN'].append(mean_squared_error(y_test_outer, y_test_pred_ann))

    # Average MSE across outer folds
    for model_name, mse_list in outer_results.items():
        print(f"{model_name} Model - Average MSE across outer folds: {np.mean(mse_list)}")

    return outer_results