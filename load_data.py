from sklearn.datasets import fetch_openml
import numpy as np
from ucimlrepo import fetch_ucirepo
class Data:
    def __init__(self, X, y, attributeNames, N, M, classNames, C):
        self.X = X  # Data matrix
        self.y = y  # Class/target labels
        self.attributeNames = attributeNames  # Attribute names
        self.N = N  # Number of data objects
        self.M = M  # Number of attributes
        self.classNames = classNames  # Unique class names
        self.C = C  # Number of classes

def load_data():
    # Fetch the latest version of the 'anneal' dataset using its name
    abalone = fetch_ucirepo(id=1) 

     
    # Extract data components
    X = abalone.data.features  # Convert pandas dataframe to numpy array (N x M)
    y = abalone.target.values  # Convert target series to numpy array (N x 1)
    attributeNames = list(abalone.feature_names)  # List of attribute names
    N = X.shape[0]  # Number of data objects
    M = X.shape[1]  # Number of attributes
    classNames = list(np.unique(y))  # Get unique class labels
    C = len(classNames)  # Number of classes
    
    return Data(X, y, attributeNames, N, M, classNames, C)
# Example usage
data_instance = load_data()

# Accessing the data
print(data_instance.X)  # Prints the feature matrix
