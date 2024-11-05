from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from Regression2 import cross_validation_model_selection, cross_validation_model_selection2, estimate_generalization_error, plot_training_vs_generalization_error, regularized_regression
import load_data as ld
import pandas as pd
import numpy as np

from preprocess import normalize_continuous_features, one_of_K, validate_mean_and_std

# Loading data
X, y = ld.load_data()


# Preprocess the data

# One hot encoding to change sex to converts categorical data into a numerical format
X_encoded = one_of_K(X)

# Normalize the data so that mean is 0 and std is 1
continuous_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
X_normalized = normalize_continuous_features(X_encoded, continuous_features)
# validate_mean_and_std(X_normalized, continuous_features)


X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# best_ridge_model, best_ridge_lambda = regularized_regression(X_train, y_train, model_type='ridge')

# print(best_ridge_lambda)
# print(best_ridge_lambda)
# best_lasso_model, best_lasso_lambda = regularized_regression(X_train, y_train, model_type='lasso')

# lambdas, generalization_errors = estimate_generalization_error(X_train, y_train, model_type='ridge')
# lambdas, training_errors, generalization_errors = plot_training_vs_generalization_error(X_train, y_train, model_type='ridge')


lambdas, training_errors, generalization_errors = cross_validation_model_selection(X_train, y_train, model_type='ridge')

print(lambdas)
print(training_errors)
print(generalization_errors)

best_lambda = 0.464
ridge_model = Ridge(alpha=best_lambda)
ridge_model.fit(X_train, y_train)
intercept = ridge_model.intercept_
coefficients = ridge_model.coef_

print("Intercept:", intercept)
print("Coefficients:", coefficients)