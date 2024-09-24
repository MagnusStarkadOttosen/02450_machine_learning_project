import numpy as np
import load_data as ld
import matplotlib.pyplot as plt
import pandas as pd
import encode_sex as es
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
X = ld.load_data()

X = es.encode_sex(X)

# Select the features you want for PCA (excluding 'Sex' if it's one-hot encoded)
X_features = X.drop(columns=['Sex'])  # Adjust if necessary based on your encoding

# Standardize the data (PCA assumes data is centered and scaled)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Initialize PCA
pca = PCA(n_components=2)  # Change 'n_components' to the number of components you want
X_pca = pca.fit_transform(X_scaled)

# View explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance)

# Assuming X_pca contains the transformed data with two components
plt.figure(figsize=(8,6))

# Plot the two principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X['Sex'], cmap='viridis')  # 'c' represents the color based on the encoded 'Sex'

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')

# Show the plot
plt.colorbar(label='Sex encoding')  # Add color bar to see what the colors represent
plt.show()