

def remove_outliers(X, feature_list):
    X_clean = X.copy()
    
    for feature in feature_list:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the data to remove outliers
        X_clean = X_clean[(X_clean[feature] >= lower_bound) & (X_clean[feature] <= upper_bound)]
    
    return X_clean