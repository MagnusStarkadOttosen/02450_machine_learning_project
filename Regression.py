from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import load_data as ld
import pandas as pd
import numpy as np

# Loading data
X, y = ld.load_data()

# one-of-K encoding
X_encoded = pd.get_dummies(X, columns=['Sex'])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_normalized = X_encoded
continuous_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
X_normalized[continuous_features] = scaler.fit_transform(X_normalized[continuous_features])


print(X)
print()
print(X_encoded)
print()
print(X_normalized)


# Calculate the mean and standard deviation for continuous features only
means = X_normalized[continuous_features].mean()
stds = X_normalized[continuous_features].std()

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