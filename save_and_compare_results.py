import pandas as pd
import glob
import matplotlib.pyplot as plt

def save_feature_importance(features, importance_scores, method, model_name):
    """
    Save feature importance scores to a CSV file.
    
    Parameters:
        features: list of feature names
        importance_scores: list of corresponding importance scores
        method: str, method used (e.g., "SHAP", "Lasso", "LightGBM")
        model_name: str, name of the model
    """
    df = pd.DataFrame({
        'Feature_ID': features,
        'Importance/Score': importance_scores
    }).sort_values(by='Importance/Score', ascending=False)
    
    df['Rank'] = range(1, len(df) + 1)
    
    output_path = f"../results/{model_name}_{method}_importance.csv"
    df.to_csv(output_path, index=False)
    print(f"Feature importance saved to {output_path}")

def compare_feature_importances(file_pattern, top_n=20):
    """
    Compare feature importances across multiple files.
    
    Parameters:
        file_pattern: str, glob pattern to match files (e.g., "*.csv")
        top_n: int, number of top features to consider from each file
    
    Returns:
        DataFrame summarizing feature frequencies
    """
    files = glob.glob(file_pattern)
    all_features = []

    for file in files:
        df = pd.read_csv(file)
        top_features = df['Feature_ID'].head(top_n).tolist()
        all_features.extend(top_features)

    feature_counts = pd.Series(all_features).value_counts()

    summary_df = feature_counts.reset_index()
    summary_df.columns = ['Feature_ID', 'Frequency']

    return summary_df

def plot_feature_frequencies(summary_df):
    """
    Plots the top 20 most frequent features from a summary DataFrame.

    Parameters:
    - summary_df: Pandas DataFrame containing feature frequencies.
    
    Returns:
    - A horizontal bar plot displaying the top 20 most frequent features.
    """
    summary_df = summary_df.sort_values(by="Frequency", ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(summary_df['Feature_ID'], summary_df['Frequency'], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Feature')
    plt.title('Top Recurrent Features Across Models and Methods')
    plt.gca().invert_yaxis()
    plt.show()
