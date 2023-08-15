import numpy as np
import pandas as pd

def transform_data(test_df, col_order, onehot): 
    # Copy dataframe
    X = test_df.copy()
    
    # Onehot encoder 
    encoded_columns = onehot.transform(X.select_dtypes(include='object')).toarray()
    X = X.select_dtypes(exclude='object')
    X[onehot.get_feature_names_out()] = encoded_columns
    
    return X[col_order]