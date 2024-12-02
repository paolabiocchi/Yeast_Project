# %%
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

# %%
'''
# y prends la forme : yeast ID, doubling_time
# Charger les datasets
x_train = pd.read_csv("data/X_matrix.csv")
y_train = pd.read_csv("data/Y_matrix.csv")
print("csv read complete")

#check if the first columns is the same
is_same = x_train.iloc[:, 0].equals(y_train.iloc[:, 0])
print(f"Are the first columns the same? {is_same}")
'''
# %%

def scale_last_columns(data, num_last_columns=7000):
    """
    Scales the last N columns (assumed to be CNVs) to a range of 0 to 1.
    
    Parameters:
        data (pd.DataFrame): x_train data
        num_last_columns (int): Number of copy number variation columns from the end to scale .

    Returns:
        pd.DataFrame: Data with the last columns scaled.
    """
    # Select the last N columns
    cnv_columns = data.iloc[:, -num_last_columns:]
    
    # Scale these columns
    scaler = MinMaxScaler()
    scaled_cnv = scaler.fit_transform(cnv_columns)
    
    # Replace the last N columns with their scaled values
    data.iloc[:, -num_last_columns:] = scaled_cnv
    
    return data


def remove_low_variance_features_last_columns(data, num_last_columns=6051, threshold=0.05):
    """
    Removes features with variance below a specified threshold in the last N columns.
    This has been done already for mutations during our extraction of data, so it is only useful to do it for the copy number variation columns.
    
    Parameters:
        data (pd.DataFrame): Input data.
        num_last_columns (int): Number of columns from the end to apply variance filtering.
        threshold (float): Minimum variance a feature must have to be retained.

    Returns:
        pd.DataFrame: Data with low-variance features removed in the last N columns.
    """
    # Select the last N columns
    target_columns = data.iloc[:, -num_last_columns:]
    
    # Apply VarianceThreshold to these columns
    selector = VarianceThreshold(threshold=threshold)
    reduced_data = selector.fit_transform(target_columns)
    
    # Get the selected column indices
    selected_columns = target_columns.columns[selector.get_support()]
    
    # Replace the last N columns with the reduced set
    data = data.drop(columns=target_columns.columns)  # Drop the original last N columns
    data = pd.concat([data, pd.DataFrame(reduced_data, columns=selected_columns)], axis=1)
    
    return data


def apply_pca_last_columns(data, num_last_columns=7000, n_components=0.95, normalize=True):
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
    other_columns = data.iloc[:, :-num_last_columns]
    target_columns = data.iloc[:, -num_last_columns:]
    
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

# Check for NaN values
def has_nan(df):
    has_nan = df.isnull().values.any()

    if has_nan:
        print("The DataFrame contains NaN values.")
    else:
        print("The DataFrame does not contain any NaN values.")

def preprocessed_data (x_df, y_df) :
    x_train = x_df
    has_nan(x_train)
    x_train = scale_last_columns(x_train)
    has_nan(x_train)
    x_train = remove_low_variance_features_last_columns(x_train)
    has_nan(x_train)
    x_train = apply_pca_last_columns(x_train)

    y_train= y_df

    return x_train, y_train

#x_train_preprocessed, _ = preprocessed_data (x_train, y_train)
#metrics(x_train, x_train_preprocessed)



