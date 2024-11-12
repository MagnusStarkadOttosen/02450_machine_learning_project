from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from Regression2 import compare_regression_models, cross_validation_model_selection, cross_validation_model_selection2, estimate_generalization_error, plot_training_vs_generalization_error, regularized_regression, two_level_cross_validation, two_level_cross_validation_table
import load_data as ld
import pandas as pd
import numpy as np

from preprocess import normalize_continuous_features, one_of_K, validate_mean_and_std

# Loading data
X, y = ld.load_data()


# Preprocess the data

# One hot encoding to change sex to converts categorical data into a numerical format
X_encoded = one_of_K(X)
print(X_encoded)
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



# Convert X_normalized and y to NumPy arrays with consistent numeric types
X_normalized = X_normalized.to_numpy(dtype=np.float32) if isinstance(X_normalized, pd.DataFrame) else X_normalized.astype(np.float32)
y = y.to_numpy(dtype=np.float32) if isinstance(y, (pd.DataFrame, pd.Series)) else y.astype(np.float32)


# Define hyperparameters for cross-validation
lambdas = [0.1, 0.5, 1.0, 2.0]  # Example range for Ridge regularization
hidden_units_list = [1, 5, 10, 20]  # Example range for ANN hidden units

# Run two-level cross-validation
# results = two_level_cross_validation(X_normalized, y, lambdas, hidden_units_list)

# Run two-level cross-validation and generate table
results_df = two_level_cross_validation_table(X_normalized, y, lambdas, hidden_units_list)

# Display the table
print(results_df)

# Extract the test error columns for ANN and Ridge models
loss_ANN = results_df["E_test_i_ANN"].values
loss_Ridge = results_df["E_test_i_Ridge"].values

# Use the compare_regression_models function to compute confidence intervals and p-value
results = compare_regression_models(loss_ANN, loss_Ridge, confidence_level=0.95)

# Print the results
print("Lower bound of confidence interval:", results["z_L"])
print("Upper bound of confidence interval:", results["z_U"])
print("P-value:", results["p_value"])
print("Mean difference:", results["z_mean"])
print("Variance of difference:", results["z_variance"])