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

validate_mean_and_std(X_normalized, continuous_features)




