from matplotlib import pyplot as plt
import pandas as pd
import load_data as ld
import outliers
import plots
import standardize as st
import seaborn as sns
from sklearn.decomposition import PCA

X, y = ld.load_data()

continuous_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']

X = outliers.remove_outliers(X, continuous_features)
# # X = st.scale_standardize(X, continuous_features)

# plots.plot_histograms(X, continuous_features)

# plots.plot_boxplots(X, continuous_features)

# plots.plot_sex_bar(X)

# plots.plot_pairwise_scatter(X, continuous_features)

# plots.plot_boxplots_by_sex(X, continuous_features)

# plots.plot_violin_by_sex(X, continuous_features)

# plots.plot_correlation_heatmap(X, continuous_features)

# plots.plot_pca_2d(X, continuous_features, 'Sex')


all_features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']


summary_stats = X[all_features].describe()
print(summary_stats)

import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder  

def perform_pca(X, plot_variance=False, threshold=0.9):
    # if 'Sex' in X.columns:
    #     le = LabelEncoder()
    #     X['Sex'] = le.fit_transform(X['Sex'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Y = X_scaled
    U, S, Vt = svd(Y, full_matrices=False)
    V = Vt.T

    rho = (S ** 2) / (S ** 2).sum()

    Z = Y @ V

    if plot_variance:
        plt.figure()
        plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual Variance")
        plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative Variance")
        plt.axhline(y=threshold, color='k', linestyle='--', label=f'{threshold * 100}% Threshold')
        plt.title("Variance Explained by Principal Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.legend()
        plt.grid(True)
        plt.show()
  
    return Z, rho, V

if 'Sex' in X.columns:
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])

X_no_sex = X.drop('Sex', axis=1)
Z, rho, V = perform_pca(X_no_sex, plot_variance=True)

print("Principal Components (first 2):")
print(V[:, :2])
print("\nVariance Explained by Each Component:")
print(rho)

plt.figure(figsize=(8,6))
plt.scatter(Z[:, 0], Z[:, 1], c='blue', edgecolor='k', s=50)
plt.title("PCA - First Two Principal Components")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()

# Assuming X['Sex'] is already encoded
# if 'Sex' in X.columns:
#     le = LabelEncoder()
#     X['Sex'] = le.fit_transform(X['Sex'])

# Get the mapping from encoded values to original labels
sex_mapping = le.classes_  # This will return something like ['female', 'infant', 'male']

# Plot the first two principal components, colored by 'Sex'
plt.figure(figsize=(8,6))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=X['Sex'], cmap='viridis', edgecolor='k', s=50)

# Add title and labels
plt.title("PCA - First Two Principal Components Colored by Sex")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)

# Create a legend with the original 'Sex' categories
legend_labels = {i: sex_mapping[i] for i in range(len(sex_mapping))}
handles, _ = scatter.legend_elements()
plt.legend(handles, [legend_labels[i] for i in range(len(legend_labels))], title="Sex")

# Add a color bar (optional)
plt.colorbar(scatter, label='Encoded Sex Values')
plt.show()

