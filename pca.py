import numpy as np
import load_data as ld
import matplotlib.pyplot as plt
import pandas as pd
import encode_sex as es
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the data
X, y = ld.load_data()

X = es.encode_sex(X)


features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
X_selected = X[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
print(f'Explained Variance Ratio: {explained_variance}')

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=X['Sex'], palette='Set1')
plt.title('PCA of Selected Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

age = y.to_numpy() + 1.5

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=age, cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Age (Years)')
plt.title('PCA of Selected Features Colored by Age')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()