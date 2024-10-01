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
# X = st.scale_standardize(X, continuous_features)

plots.plot_histograms(X, continuous_features)

plots.plot_boxplots(X, continuous_features)

plots.plot_sex_bar(X)

plots.plot_pairwise_scatter(X, continuous_features)

plots.plot_boxplots_by_sex(X, continuous_features)

plots.plot_violin_by_sex(X, continuous_features)

plots.plot_correlation_heatmap(X, continuous_features)

plots.plot_pca_2d(X, continuous_features, 'Sex')
