import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def one_of_K(X, columns=['Sex']):
    X_encoded = pd.get_dummies(X, columns=columns)
    return X_encoded

def normalize_continuous_features(X, continuous_features):
    scaler = StandardScaler()
    X_normalized = X.copy()  # Copy to avoid modifying the original
    X_normalized[continuous_features] = scaler.fit_transform(X_normalized[continuous_features])
    return X_normalized

def validate_mean_and_std(X, continuous_features):
    # Calculate the mean and standard deviation for continuous features only
    means = X[continuous_features].mean()
    stds = X[continuous_features].std()

    # Print the means and standard deviations
    print("Column means (continuous features):")
    print(means)
    print("\nColumn standard deviations (continuous features):")
    print(stds)

    # Check if mean is close to 0 (within 3 degits after .)
    mean_check = np.allclose(means, 0, atol=1e-3)

    # Check if std is close to 1 (within 3 degits after .)
    std_check = np.allclose(stds, 1, atol=1e-3)

    print(f"\nAll continuous columns have mean 0: {mean_check}")
    print(f"All continuous columns have standard deviation 1: {std_check}")