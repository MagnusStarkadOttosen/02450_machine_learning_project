from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def plot_sex_bar(X):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', data=X)
    plt.title('Count of Sex Categories')
    plt.show()
    
def plot_pairwise_scatter(X, feature_list):
    sns.pairplot(X[feature_list])
    plt.show()
    
def plot_boxplots(X, feature_list):
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_list, 1):
        plt.subplot(2, 4, i)
        sns.boxplot(y=X[feature])
        plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    plt.show()
    
def plot_boxplots_by_sex(X, feature_list):
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_list, 1):
        plt.subplot(2, 4, i)
        sns.boxplot(x='Sex', y=feature, data=X)
        plt.title(f'Boxplot of {feature} by Sex')
    plt.tight_layout()
    plt.show()
    
def plot_histograms(X, feature_list):
    X[feature_list].hist(bins=20, figsize=(15, 10), layout=(2, 4))
    plt.tight_layout()
    plt.show()
    
def plot_pairwise_scatter(X, feature_list):
    sns.pairplot(X[feature_list])
    plt.show()
    
def plot_violin_by_sex(X, feature_list):
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_list, 1):
        plt.subplot(2, 4, i)
        sns.violinplot(x='Sex', y=feature, data=X)
        plt.title(f'Violin Plot of {feature} by Sex')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(X, feature_list):
    plt.figure(figsize=(10, 8))
    sns.heatmap(X[feature_list].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_pca_2d(X, feature_list, target_column):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X[feature_list])
    X_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    X_pca[target_column] = X[target_column]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue=target_column, data=X_pca)
    plt.title('2D PCA of Continuous Features')
    plt.show()