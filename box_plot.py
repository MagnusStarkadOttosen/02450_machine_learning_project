import load_data as ld
import matplotlib.pyplot as plt

# Load the data
X = ld.load_data()

# Create boxplots for each column in X
plt.figure(figsize=(12, 8))
X.boxplot()
plt.title('Boxplot of All Features')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()
