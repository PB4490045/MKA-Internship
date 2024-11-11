import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import shapiro
import matplotlib.pyplot as plt

def paired_ttest(df1, df2, columns=None):
    """
    Perform paired t-tests between two DataFrames for specified columns.

    Parameters:
    df1 (pd.DataFrame): First DataFrame with measurements.
    df2 (pd.DataFrame): Second DataFrame with measurements.
    columns (list, optional): List of columns to perform the t-test on. If None, tests all columns.

    Returns:
    pd.DataFrame: DataFrame containing t-statistics and p-values for each tested column.
    """
    # If no specific columns are provided, use all common columns
    if columns is None:
        columns = df1.columns.intersection(df2.columns)

    results = {}
    
    for column in columns:
        # Check for matching shapes and NaN handling
        if column not in df1 or column not in df2:
            raise ValueError(f"Column '{column}' not found in both DataFrames.")
        
        # Drop NaN values for both DataFrames
        valid_data = df1[column].notna() & df2[column].notna()
        if valid_data.sum() == 0:
            print(f"No valid data for column '{column}'. Skipping.")
            continue
        
        # Perform paired t-test
        t_stat, p_value = ttest_rel(df1[column][valid_data], df2[column][valid_data])
        results[column] = {'t_statistic': t_stat, 'p_value': p_value}

    # Convert results to a DataFrame to view nicely
    results_df = pd.DataFrame(results).T
    return results_df



def wilcoxon_test(df1, df2, columns=None):
    """
    Perform Wilcoxon signed-rank tests between two DataFrames for specified columns.

    Parameters:
    df1 (pd.DataFrame): First DataFrame with measurements.
    df2 (pd.DataFrame): Second DataFrame with measurements.
    columns (list, optional): List of columns to perform the test on. If None, uses all common columns.

    Returns:
    pd.DataFrame: DataFrame containing test statistics and p-values for each tested column.
    """
    # If no specific columns are provided, use all common columns
    if columns is None:
        columns = df1.columns.intersection(df2.columns)

    results = {}

    for column in columns:
        # Perform the Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(df1[column], df2[column])
        results[column] = {
            'statistic': statistic,
            'p_value': p_value
        }

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results).T
    return results_df

def shapiro_test(df, column):
    """
    Perform the Shapiro-Wilk test for normality on a specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column on which to perform the Shapiro-Wilk test.

    Returns:
    tuple: The test statistic and p-value from the Shapiro-Wilk test.
    """
    if column in df.columns:
        statistic, p_value = shapiro(df[column].dropna())  # Drop NaN values before the test
        return statistic, p_value
    else:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
def plot_histogram(df, column):
    """
    Create a histogram of the specified column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column for which to create the histogram.
    """
    if column in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column].dropna(), bins=30, alpha=0.7, color='blue')  # Drop NaN values
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
    else:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

def calculate_errors(df1, df2, columns=None):
    """
    Calculate systematic and random errors between two DataFrames for specified columns.

    Parameters:
    df1 (pd.DataFrame): First DataFrame (ground truth measurements).
    df2 (pd.DataFrame): Second DataFrame (predicted measurements).
    columns (list, optional): List of columns to calculate errors on. If None, uses all common columns.

    Returns:
    pd.DataFrame: DataFrame containing systematic error (mean difference) and random error (std of differences).
    """
    # If no specific columns are provided, use all common columns
    if columns is None:
        columns = df1.columns.intersection(df2.columns)

    results = {}

    for column in columns:
        # Ensure the column exists in both DataFrames
        if column not in df1 or column not in df2:
            raise ValueError(f"Column '{column}' not found in both DataFrames.")
        
        # Drop NaN values for both DataFrames
        valid_data = df1[column].notna() & df2[column].notna()
        if valid_data.sum() == 0:
            print(f"No valid data for column '{column}'. Skipping.")
            continue

        # Calculate systematic error (mean difference) and random error (std of differences)
        differences = df1[column][valid_data] - df2[column][valid_data]
        systematic_error = differences.mean()
        random_error = differences.std()
        
        results[column] = {
            'systematic_error': systematic_error,
            'random_error': random_error
        }

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results).T
    return results_df

def combined_analysis(df1, df2, columns=None):
    """
    Perform paired t-tests, and calculate systematic and random errors between two DataFrames for specified columns.

    Parameters:
    df1 (pd.DataFrame): First DataFrame (ground truth measurements).
    df2 (pd.DataFrame): Second DataFrame (predicted measurements).
    columns (list, optional): List of columns to analyze. If None, uses all common columns.

    Returns:
    pd.DataFrame: DataFrame containing t-statistic, p-value, systematic error (mean difference),
                  and random error (std of differences) for each specified column.
    """
    # If no specific columns are provided, use all common columns
    if columns is None:
        columns = df1.columns.intersection(df2.columns)

    results = {}

    for column in columns:
        # Ensure the column exists in both DataFrames
        if column not in df1 or column not in df2:
            raise ValueError(f"Column '{column}' not found in both DataFrames.")
        
        # Drop NaN values for both DataFrames
        valid_data = df1[column].notna() & df2[column].notna()
        if valid_data.sum() == 0:
            print(f"No valid data for column '{column}'. Skipping.")
            continue

        # Calculate systematic error (mean difference) and random error (std of differences)
        differences = df1[column][valid_data] - df2[column][valid_data]
        systematic_error = differences.mean()
        random_error = differences.std()

        # Perform paired t-test
        t_stat, p_value = ttest_rel(df1[column][valid_data], df2[column][valid_data])
        
        # Store all results in a dictionary
        results[column] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'systematic_error': systematic_error,
            'random_error': random_error
        }

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results).T
    return results_df