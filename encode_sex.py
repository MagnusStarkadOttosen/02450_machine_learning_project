from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_sex(X):
    # Initialize the encoder
    le = LabelEncoder()

    # Encode the 'Sex' column
    X['Sex'] = le.fit_transform(X['Sex'])
    
    return X