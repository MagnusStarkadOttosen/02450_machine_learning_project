import load_data as ld
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
X = ld.load_data()

pd.plotting.scatter_matrix(X, figsize=(12, 12))
plt.show()