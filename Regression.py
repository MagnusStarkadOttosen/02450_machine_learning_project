from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import load_data as ld
import pandas as pd
import numpy as np

from preprocess import normalize_continuous_features, one_of_K, validate_mean_and_std

# Loading data
X, y = ld.load_data()

# one-of-K encoding
# X_encoded = one_of_K(X)

X_encoded = one_of_K(X)

# Initialize StandardScaler
# scaler = StandardScaler()

# # Fit and transform the data
# X_normalized = X_encoded
continuous_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
# X_normalized[continuous_features] = scaler.fit_transform(X_normalized[continuous_features])


X_normalized = normalize_continuous_features(X_encoded, continuous_features)

# validate_mean_and_std(X_normalized, continuous_features)


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error

# def estimate_generalization_error(X, y, lambdas, K=10):
#     """
#     Estimates the generalization error using K-fold cross-validation for different values of lambda.
    
#     Parameters:
#     - X: Feature matrix (DataFrame)
#     - y: Target variable (Series)
#     - lambdas: List or array of lambda values for regularization
#     - K: Number of folds for cross-validation
    
#     Returns:
#     - errors: Dictionary with lambda values as keys and generalization errors as values
#     """
#     # Ensure that y is a pandas Series to avoid indexing issues
#     if isinstance(y, pd.DataFrame):
#         y = y.squeeze()  # Convert to Series if it's a single-column DataFrame
    
#     errors = {}
#     kf = KFold(n_splits=K, shuffle=True, random_state=1)

#     for lam in lambdas:
#         model = Ridge(alpha=lam)
#         fold_errors = []

#         for train_index, test_index in kf.split(X):
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
#             # Train model and compute test error for this fold
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             fold_errors.append(mean_squared_error(y_test, y_pred))

#         # Average test error across all folds
#         errors[lam] = np.mean(fold_errors)

#     return errors

# # print(X_normalized)

# continuous_X = X_normalized.drop(columns=["Sex_F", "Sex_I", "Sex_M"])

# # Trying a wider range of lambda values to capture variability
# lambdas = np.logspace(-20, 3, 15)  # Extended range with more points for better resolution

# errors = estimate_generalization_error(continuous_X, y, lambdas)

# # Plotting the results with the updated range
# plt.figure(figsize=(10, 6))
# plt.plot(lambdas, list(errors.values()), marker='o', linestyle='-')
# plt.xscale('log')
# plt.xlabel(r'Regularization Parameter $\lambda$')
# plt.ylabel('Estimated Generalization Error')
# plt.title(r'Generalization Error as a Function of Regularization Parameter $\lambda$')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def calculate_errors(X, y, lambdas, K=10):
    """
    Calculates training and cross-validation errors for different values of lambda.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - lambdas: List or array of lambda values for regularization
    - K: Number of folds for cross-validation
    
    Returns:
    - training_errors: List of training errors for each lambda
    - generalization_errors: List of cross-validation errors for each lambda
    """
    training_errors = []
    generalization_errors = []
    
    # Set up K-Fold cross-validation
    kf = KFold(n_splits=K, shuffle=True, random_state=1)
    
    for lam in lambdas:
        model = Ridge(alpha=lam)
        fold_train_errors = []
        fold_test_errors = []
        
        # Cross-validation loop
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model and calculate errors
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate training and test errors for this fold
            fold_train_errors.append(mean_squared_error(y_train, y_train_pred))
            fold_test_errors.append(mean_squared_error(y_test, y_test_pred))
        
        # Average errors across all folds
        training_errors.append(np.mean(fold_train_errors))
        generalization_errors.append(np.mean(fold_test_errors))
    
    # Plot the training and cross-validation errors
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, training_errors, label="Training Error", marker='o', linestyle='-')
    plt.plot(lambdas, generalization_errors, label="Estimated Generalization Error", marker='o', linestyle='-')
    plt.xscale("log")
    plt.xlabel(r'Regularization Parameter $\lambda$')
    plt.ylabel("Error")
    plt.title(r'Training vs Generalization Error as a Function of $\lambda$')
    plt.legend()
    plt.show()
    
    return training_errors, generalization_errors

# Define a range of lambda values
lambdas = np.logspace(-4, 4, 10)

# Calculate errors and plot
training_errors, generalization_errors = calculate_errors(X_normalized, y, lambdas)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def calculate_errors_vs_model(X, y, lambdas, K=10):
    """
    Calculates training and cross-validation errors for different models defined by lambda values.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - lambdas: List or array of lambda values for regularization (each defines a "model")
    - K: Number of folds for cross-validation
    
    Returns:
    - training_errors: List of training errors for each model (lambda value)
    - generalization_errors: List of cross-validation errors for each model (lambda value)
    """
    training_errors = []
    generalization_errors = []
    
    # Set up K-Fold cross-validation
    kf = KFold(n_splits=K, shuffle=True, random_state=1)
    
    # Loop through each model defined by lambda
    for lam in lambdas:
        model = Ridge(alpha=lam)
        fold_train_errors = []
        fold_test_errors = []
        
        # Cross-validation loop
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train the model and calculate errors
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Store training and test errors for this fold
            fold_train_errors.append(mean_squared_error(y_train, y_train_pred))
            fold_test_errors.append(mean_squared_error(y_test, y_test_pred))
        
        # Average errors across all folds for this model
        training_errors.append(np.mean(fold_train_errors))
        generalization_errors.append(np.mean(fold_test_errors))
    
    # Plotting error vs model index
    model_indices = range(1, len(lambdas) + 1)  # 1, 2, ..., len(lambdas)
    
    plt.figure(figsize=(10, 6))
    plt.plot(model_indices, training_errors, label="Training Error", marker='o', linestyle='-')
    plt.plot(model_indices, generalization_errors, label="Estimated Generalization Error", marker='o', linestyle='-')
    plt.xlabel("Model Index")
    plt.ylabel("Error")
    plt.title("Training vs Generalization Error for Different Models (Lambda Values)")
    plt.legend()
    plt.show()
    
    return training_errors, generalization_errors

# Define a range of lambda values to represent different models
lambdas = np.logspace(-4, 4, 3)  # For example, three models with lambda=0.0001, 1, and 10000

# Calculate errors and plot
training_errors, generalization_errors = calculate_errors_vs_model(X_normalized, y, lambdas)

