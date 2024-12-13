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
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(data, title, bins=50, figsize=(12, 6)):
    """
    Plot histograms of all numerical columns in the DataFrame.
    """
    plt.figure(figsize=figsize)
    data.hist(bins=bins, figsize=figsize)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data, title, figsize=(10, 8)):
    """
    Plot a heatmap of the correlation matrix.
    """
    corr_matrix = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, fmt=".2f")
    plt.title(title, fontsize=16)
    plt.show()

def visualize_pca_variance(pca, title):
    """
    Plot the explained variance ratio of PCA components.
    """
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(title)
    plt.grid()
    plt.show()

def scale_last_columns(data, num_last_columns=6051):
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

    # Plot after scaling
    #plot_histograms(data.iloc[:, -num_last_columns:], title="After Scaling CNVs")
    
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
    
    new_num_last_columns = reduced_data.shape[1]

    # Replace the last N columns with the reduced set
    data = data.drop(columns=target_columns.columns)  # Drop the original last N columns
    data = pd.concat([data, pd.DataFrame(reduced_data, columns=selected_columns)], axis=1)
    print(data.columns[-(new_num_last_columns+1)], data.columns[-new_num_last_columns])
    """
    # Plot variance after filtering
    reduced_variances = pd.DataFrame(reduced_data).var(axis=0)
    plt.figure(figsize=(12, 6))
    plt.hist(reduced_variances, bins=50, color="orange")
    plt.title("Variance of Last Columns (After Filtering)")
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.show()
    """

    return data, new_num_last_columns


def apply_pca_last_columns(data, num_last_columns, n_components=0.95, normalize=True):
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

    # Plot explained variance
    #visualize_pca_variance(pca, title="Explained Variance by PCA Components")

    
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


def shuffle_dataset(data, labels):
    """
    Shuffle the dataset and labels together to randomize the order of examples.
    
    Parameters:
        data (pd.DataFrame): Features dataset.
        labels (pd.DataFrame): Labels dataset.

    Returns:
        Tuple: Shuffled features and labels.
    """
    combined = pd.concat([data, labels], axis=1)
    shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and labels again
    shuffled_data = shuffled.iloc[:, :-labels.shape[1]]
    shuffled_labels = shuffled.iloc[:, -labels.shape[1]:]
    
    return shuffled_data, shuffled_labels


def has_nan(df):
    has_nan = df.isnull().values.any()

    if has_nan:
        print("The DataFrame contains NaN values.")
    else:
        print("The DataFrame does not contain any NaN values.")

import numpy as np
import pandas as pd

# Variables to store scaling parameters
scaling_params = {}

def y_preprocessing(y_df, method="min-max"):
    """
    Preprocesses the y values based on the selected method: "min-max" or "standardization".
    Parameters:
    - y_df: Pandas DataFrame or Series containing the target values.
    - method: Scaling method, either "min-max" or "standardization".

    Returns:
    - Preprocessed y as a NumPy array.
    """
    global scaling_params
    y_array = y_df.values if isinstance(y_df, (pd.Series, pd.DataFrame)) else np.array(y_df)
    
    if method == "min-max":
        min_val = np.min(y_array)
        max_val = np.max(y_array)
        scaling_params["min"] = min_val
        scaling_params["max"] = max_val
        y_scaled = (y_array - min_val) / (max_val - min_val)
    elif method == "standardization":
        mean = np.mean(y_array)
        std = np.std(y_array)
        scaling_params["mean"] = mean
        scaling_params["std"] = std
        y_scaled = (y_array - mean) / std
    else:
        raise ValueError("Invalid method. Choose 'min-max' or 'standardization'.")
    
    scaling_params["method"] = method
    return y_scaled

def y_reverse(y_pred):
    """
    Reverses the preprocessing on the predicted y values.
    Parameters:
    - y_pred: Preprocessed y values as a NumPy array.

    Returns:
    - Original y values as a NumPy array.
    """
    global scaling_params
    method = scaling_params.get("method")
    
    if method == "min-max":
        min_val = scaling_params["min"]
        max_val = scaling_params["max"]
        y_original = y_pred * (max_val - min_val) + min_val
    elif method == "standardization":
        mean = scaling_params["mean"]
        std = scaling_params["std"]
        y_original = y_pred * std + mean
    else:
        raise ValueError("Invalid method. Choose 'min-max' or 'standardization'.")
    
    return y_original


def preprocessed_data (x_df, y_df, y=False, method_chosen="min-max") :

    x_df = scale_last_columns(x_df)
    has_nan(x_df)
    print("1")
    
    x_df, new_num_last_columns = remove_low_variance_features_last_columns(x_df)
    has_nan(x_df)
    print("2")

    x_df = apply_pca_last_columns(x_df, num_last_columns = new_num_last_columns)
    has_nan(x_df)
    print("3")

    x_df, y_df = shuffle_dataset(x_df, y_df)  

    if y==True :
        y_df = y_preprocessing(y_df, method=method_chosen)      
    
    return x_df, y_df


