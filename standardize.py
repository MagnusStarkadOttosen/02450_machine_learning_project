from sklearn.preprocessing import StandardScaler

def scale_standardize(X, feature_list):
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[feature_list] = scaler.fit_transform(X[feature_list])
    return X_scaled