import dask.dataframe as dd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler


X_file = "data/X_matrix.csv"
Y_file = "data/Y_matrix.csv"

print("Loading the data...")
X = pd.read_csv(X_file)
Y = pd.read_csv(Y_file)

# Ensure Yeast_ID is a string type
X['Yeast_ID'] = X['Yeast_ID'].astype(str)
Y['Yeast_ID'] = Y['Yeast_ID'].astype(str)

# Find the intersection of Yeast_ID in both files
common_yeasts = set(X['Yeast_ID']).intersection(set(Y['Yeast_ID']))

# Identify Yeast_IDs that are missing in each file
missing_in_X = set(Y['Yeast_ID']) - set(X['Yeast_ID'])
missing_in_Y = set(X['Yeast_ID']) - set(Y['Yeast_ID'])

def scale_last_columns(data, num_mutations=341901):
    """
    Scales the last N columns (assumed to be CNVs) to a range of 0 to 1.
    
    Parameters:
        data (pd.DataFrame): x_train data
        num_last_columns (int): Number of copy number variation columns from the end to scale .

    Returns:
        pd.DataFrame: Data with the last columns scaled.
    """
    # Select the last N columns
    cnv_columns = data.iloc[:, num_mutations:]
    
    # Scale these columns
    scaler = MinMaxScaler()
    scaled_cnv = scaler.fit_transform(cnv_columns)
    
    # Replace the last N columns with their scaled values
    data.iloc[:, num_mutations:] = scaled_cnv
    
    return data


def remove_low_variance_features_last_columns(data, num_mutations=341901, threshold=0.05):
    """
    Removes features with variance below a specified threshold in the last N columns.
    This has been done already for mutations during our extraction of data, so it is only useful to do it for the copy number variation columns.
    
    Parameters:
        data (pd.DataFrame): Input data.
        num_mutations (int): Number of mutations to not apply variance filtering.
        threshold (float): Minimum variance a feature must have to be retained.

    Returns:
        pd.DataFrame: Data with low-variance features removed in the last N columns.
    """
    # Select the last N columns
    target_columns = data.iloc[:, num_mutations:]
    
    # Apply VarianceThreshold to these columns
    selector = VarianceThreshold(threshold=threshold)
    reduced_data = selector.fit_transform(target_columns)
    
    # Get the selected column indices
    selected_columns = target_columns.columns[selector.get_support()]
    
    # Ensure new DataFrame has the correct index
    reduced_data_df = pd.DataFrame(reduced_data, columns=selected_columns, index=target_columns.index)
    
    # Replace the last N columns with the reduced set
    data = pd.concat([data.iloc[:, :num_mutations], reduced_data_df], axis=1)
    
    return data


def apply_pca_last_columns(data, num_mutations=341901, n_components=0.95, normalize=True):
    """
    Applies PCA to reduce dimensionality of the last N columns of the dataset.
    
    Parameters:
        data (pd.DataFrame): Input data.
        num_last_columns (int): Number of columns from the end to apply PCA.
        n_components (float or int): Number of components to keep or the amount of variance to retain.

    Returns:
        pd.DataFrame: Data with PCA applied to the last N columns.
    """
    # Separate the last N columns and the rest of the dataset
    other_columns = data.iloc[:, :num_mutations]
    target_columns = data.iloc[:, num_mutations:]
    
    # Apply PCA to the last N columns
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(target_columns)

    # Normalize the PCA-transformed features if specified
    if normalize:
        scaler = MinMaxScaler()
        reduced_data = scaler.fit_transform(reduced_data)
    
    # Create a new DataFrame for the reduced PCA data
    reduced_columns = [f'PCA_{i+1}' for i in range(reduced_data.shape[1])]
    reduced_df = pd.DataFrame(reduced_data, columns=reduced_columns, index=data.index)
    
    # Concatenate the other columns with the reduced PCA columns
    result = pd.concat([other_columns, reduced_df], axis=1)
    
    return result


def metrics(x_train, x_train_preprocessed):
    """
    Compute and display various metrics for the original and preprocessed data.
    
    Parameters:
        x_train (pd.DataFrame): Original data.
        x_train_preprocessed (pd.DataFrame): Preprocessed data.
    """
    # Compute and print dimensions
    print("Dimensions of the DataFrame (original):", x_train.shape)
    print("Dimensions of the DataFrame (after preprocessing):", x_train_preprocessed.shape)

    # Exclude the first row and column for calculations
    x_cut = x_train.iloc[1:, 1:]
    x_cut_p = x_train_preprocessed.iloc[1:, 1:]

    # Compute and print mean
    mean_value_x_train = x_cut.values.mean()
    mean_value_x_train_preprocessed = x_cut_p.values.mean()
    print("\nMean of all values (original):", mean_value_x_train)
    print("Mean of all values (after preprocessing):", mean_value_x_train_preprocessed)

    # Compute and print max and min
    max_value_x_train = np.max(x_cut.values)
    min_value_x_train = np.min(x_cut.values)
    max_value_x_train_preprocessed = np.max(x_cut_p.values)
    min_value_x_train_preprocessed = np.min(x_cut_p.values)
    print("\nMax value (original):", max_value_x_train, "Min value (original):", min_value_x_train)
    print("Max value (after preprocessing):", max_value_x_train_preprocessed, "Min value (after preprocessing):", min_value_x_train_preprocessed)

    # Compute and print standard deviation
    std_x_train = x_cut.values.std()
    std_x_train_preprocessed = x_cut_p.values.std()
    print("\nStandard deviation (original):", std_x_train)
    print("Standard deviation (after preprocessing):", std_x_train_preprocessed)

    # Compute and print variance
    var_x_train = x_cut.values.var()
    var_x_train_preprocessed = x_cut_p.values.var()
    print("\nVariance (original):", var_x_train)
    print("Variance (after preprocessing):", var_x_train_preprocessed)

def preprocessed_data (x_df, y_df) :
    x_train_scaled = scale_last_columns(x_df)
    x_train_lv = remove_low_variance_features_last_columns(x_train_scaled)
    x_train = apply_pca_last_columns(x_train_lv)

    return x_train, y_df


#X_data_f, y_data_f = preprocessed_data(X_data, y_data)

