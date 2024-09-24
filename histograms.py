import load_data as ld
import matplotlib.pyplot as plt

# Load the data
X = ld.load_data()

# Create histograms for each column in X
for column in X.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(X[column], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Histogram for {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()