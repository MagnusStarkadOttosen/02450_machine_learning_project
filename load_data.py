from ucimlrepo import fetch_ucirepo 

def load_data():

    # fetch dataset 
    abalone = fetch_ucirepo(id=1) 
    
    # data (as pandas dataframes) 
    X = abalone.data.features 
    y = abalone.data.targets 

    return X, y