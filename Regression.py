from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import load_data as ld
import pandas as pd

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