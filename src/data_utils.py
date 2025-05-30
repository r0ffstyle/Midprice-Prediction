"""
Data utilities module for cryptocurrency high-frequency trading models.

This module handles data loading, preprocessing, and dataset creation
for limit order book (LOB) data.

Author: Noah Trägårdh
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
import json

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def load_parquet_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a parquet file.
    
    Parameters:
    ----------
    file_path : str
        Path to the parquet file
        
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the loaded data
    """
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded data from {file_path}: {df.shape} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def load_multiple_parquet_files(directory: str, pattern: str = "*.parquet*", limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from multiple parquet files in a directory.
    
    Parameters:
    ----------
    directory : str
        Directory containing parquet files
    pattern : str, default="*.parquet*"
        Glob pattern to match file names
    limit : int, optional
        Maximum number of files to load
        
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the combined data from all files
    """
    # Find all matching files
    file_paths = sorted(glob.glob(os.path.join(directory, pattern)))
    
    if limit is not None:
        file_paths = file_paths[:limit]
    
    logger.info(f"Found {len(file_paths)} parquet files to load")
    
    # Load each file
    dfs = []
    for file_path in file_paths:
        try:
            df = load_parquet_data(file_path)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping file {file_path} due to error: {e}")
    
    # Combine all dataframes
    if not dfs:
        raise ValueError("No data loaded from any files")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape} rows x {combined_df.shape[1]} columns")
    
    return combined_df


def preprocess_lob_data(
    df: pd.DataFrame, 
    levels: int = 10,
    convert_timestamps: bool = True,
    timestamp_col: str = 'time',
    add_mid_price: bool = True,
    winsorize: bool = True,
    winsorize_quantiles: Tuple[float, float] = (0.001, 0.999),
    clean_outliers: bool = True,
    max_spread_factor: float = 10.0,
    sort_by_time: bool = True
) -> pd.DataFrame:
    """
    Preprocess limit order book data.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Raw LOB data
    levels : int, default=10
        Number of price levels to consider
    convert_timestamps : bool, default=True
        Whether to convert timestamp strings to datetime objects
    timestamp_col : str, default='time'
        Name of the timestamp column
    add_mid_price : bool, default=True
        Whether to add mid-price column if not present
    winsorize : bool, default=False
        Whether to winsorize data to remove extreme values
    winsorize_quantiles : Tuple[float, float], default=(0.001, 0.999)
        Quantiles for winsorizing
    clean_outliers : bool, default=True
        Whether to remove outliers based on spread
    max_spread_factor : float, default=10.0
        Maximum allowed spread as a factor of median spread
    sort_by_time : bool, default=True
        Whether to sort data by timestamp
        
    Returns:
    -------
    pd.DataFrame
        Preprocessed LOB data
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # 1. Convert timestamps if needed
    if convert_timestamps and timestamp_col in processed_df.columns:
        if isinstance(processed_df[timestamp_col].iloc[0], str):
            processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col])
            logger.info(f"Converted {timestamp_col} to datetime")
    
    # 2. Sort by timestamp if requested
    if sort_by_time and timestamp_col in processed_df.columns:
        processed_df = processed_df.sort_values(timestamp_col).reset_index(drop=True)
        logger.info("Sorted data by timestamp")
    
    # 3. Add mid-price if needed and not present
    if add_mid_price and 'MidPrice' not in processed_df.columns:
        if 'AskPrice_1' in processed_df.columns and 'BidPrice_1' in processed_df.columns:
            processed_df['MidPrice'] = (processed_df['AskPrice_1'] + processed_df['BidPrice_1']) / 2
            logger.info("Added MidPrice column")
    
    # 4. Check and ensure required columns are present
    required_columns = []
    for i in range(1, levels + 1):
        required_columns.extend([
            f'BidPrice_{i}', f'BidVolume_{i}', 
            f'AskPrice_{i}', f'AskVolume_{i}'
        ])
    
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
    
    # 5. Add spread column
    if 'AskPrice_1' in processed_df.columns and 'BidPrice_1' in processed_df.columns:
        processed_df['Spread'] = processed_df['AskPrice_1'] - processed_df['BidPrice_1']
        logger.info("Added spread column")
    
    # 6. Clean outliers if requested
    if clean_outliers and 'Spread' in processed_df.columns:
        # Calculate median spread
        median_spread = processed_df['Spread'].median()
        
        # Identify outliers
        outliers = processed_df['Spread'] > median_spread * max_spread_factor
        
        # Remove outliers
        n_outliers = outliers.sum()
        if n_outliers > 0:
            processed_df = processed_df[~outliers].reset_index(drop=True)
            logger.info(f"Removed {n_outliers} outliers with excessive spread")
    
    # 7. Winsorize data if requested
    if winsorize:
        # Identify numeric columns to winsorize
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude certain columns
        exclude_cols = [timestamp_col] if timestamp_col in numeric_cols else []
        winsorize_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Winsorize each column
        for col in winsorize_cols:
            lower_bound = processed_df[col].quantile(winsorize_quantiles[0])
            upper_bound = processed_df[col].quantile(winsorize_quantiles[1])
            
            # Apply winsorizing
            processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Winsorized {len(winsorize_cols)} numeric columns")
    
    # 8. Check for NaN values
    nan_counts = processed_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if not nan_cols.empty:
        logger.warning(f"NaN values found in columns: {nan_cols.to_dict()}")
        
        # Handle NaN values - forward fill for time series data
        processed_df = processed_df.fillna(method='ffill')
        
        # If still have NaNs, fill with column mean for numeric columns
        for col in processed_df.columns:
            if processed_df[col].isna().any() and np.issubdtype(processed_df[col].dtype, np.number):
                col_mean = processed_df[col].mean()
                processed_df[col] = processed_df[col].fillna(col_mean)
        
        # Drop rows that still have NaNs
        processed_df = processed_df.dropna()
        
        logger.info(f"After handling NaNs, data shape: {processed_df.shape}")
    
    return processed_df


def create_training_validation_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = 'time',
    shuffle: bool = False,
    random_state: int = 666
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test sets.
    
    For time series data, the splits are typically in chronological order.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Preprocessed LOB data
    train_ratio : float, default=0.7
        Ratio of data to use for training
    val_ratio : float, default=0.15
        Ratio of data to use for validation
    test_ratio : float, default=0.15
        Ratio of data to use for testing
    timestamp_col : str, default='time'
        Name of the timestamp column
    shuffle : bool, default=False
        Whether to shuffle the data (typically False for time series)
    random_state : int, default=42
        Random seed for reproducibility if shuffling
        
    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and test DataFrames
    """
    # Check ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    
    # Sort by timestamp if available
    if timestamp_col in df.columns and not shuffle:
        sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
    else:
        sorted_df = df.copy()
    
    # Calculate split indices
    n_samples = len(sorted_df)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split the data
    if shuffle:
        # Shuffle the data if requested
        shuffled_df = sorted_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        train_df = shuffled_df.iloc[:train_end]
        val_df = shuffled_df.iloc[train_end:val_end]
        test_df = shuffled_df.iloc[val_end:]
    else:
        # Chronological split
        train_df = sorted_df.iloc[:train_end]
        val_df = sorted_df.iloc[train_end:val_end]
        test_df = sorted_df.iloc[val_end:]
    
    logger.info(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def scale_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = 'standard',
    exclude_cols: List[str] = ['time']
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Scale numeric features using the specified method.
    
    Scalers are fit only on the training data to prevent data leakage.
    
    Parameters:
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    method : str, default='standard'
        Scaling method ('standard', 'robust', or 'minmax')
    exclude_cols : List[str], default=['time']
        Columns to exclude from scaling
        
    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]
        Scaled training, validation, and test DataFrames, plus the fitted scaler
    """
    # Identify numeric columns to scale
    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Scaling {len(scale_cols)} numeric columns using {method} scaling")
    
    # Initialize scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit scaler on training data
    scaler.fit(train_df[scale_cols])
    
    # Apply scaler to all datasets
    scaled_train = train_df.copy()
    scaled_val = val_df.copy()
    scaled_test = test_df.copy()
    
    # Transform each dataset
    scaled_train[scale_cols] = scaler.transform(train_df[scale_cols])
    scaled_val[scale_cols] = scaler.transform(val_df[scale_cols])
    scaled_test[scale_cols] = scaler.transform(test_df[scale_cols])
    
    return scaled_train, scaled_val, scaled_test, scaler


def explore_lob_data(
    df: pd.DataFrame,
    levels: int = 5,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Explore and visualize limit order book data.
    
    Parameters:
    ----------
    df : pd.DataFrame
        LOB data
    levels : int, default=5
        Number of price levels to visualize
    output_dir : str, optional
        Directory to save visualizations
        
    Returns:
    -------
    Dict[str, Any]
        Dictionary containing summary statistics and properties
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'summary': df.describe(),
        'missing_values': df.isna().sum().to_dict()
    }
    
    # 1. Basic LOB statistics
    if all(f'BidPrice_{i}' in df.columns and f'AskPrice_{i}' in df.columns for i in range(1, levels+1)):
        # Calculate statistics
        results['avg_spread'] = (df['AskPrice_1'] - df['BidPrice_1']).mean()
        results['median_spread'] = (df['AskPrice_1'] - df['BidPrice_1']).median()
        
        logger.info(f"Average spread: {results['avg_spread']:.6f}")
        logger.info(f"Median spread: {results['median_spread']:.6f}")
        
        # Visualize spread distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df['AskPrice_1'] - df['BidPrice_1'], kde=True)
        plt.title('Bid-Ask Spread Distribution')
        plt.xlabel('Spread')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'spread_distribution.png'))
            plt.close()
        else:
            plt.show()
        
        # Visualize LOB depth for a sample
        sample_idx = min(1000, len(df) - 1)  # Take a sample from the first 1000 rows
        
        # Bid side
        bid_prices = [df.loc[sample_idx, f'BidPrice_{i}'] for i in range(1, levels+1)]
        bid_volumes = [df.loc[sample_idx, f'BidVolume_{i}'] for i in range(1, levels+1)]
        
        # Ask side
        ask_prices = [df.loc[sample_idx, f'AskPrice_{i}'] for i in range(1, levels+1)]
        ask_volumes = [df.loc[sample_idx, f'AskVolume_{i}'] for i in range(1, levels+1)]
        
        plt.figure(figsize=(14, 7))
        
        # Plot bid side
        plt.barh(range(levels), bid_volumes, color='green', alpha=0.6, label='Bid')
        
        # Plot ask side
        plt.barh(range(levels), [-v for v in ask_volumes], color='red', alpha=0.6, label='Ask')
        
        # Add price labels
        for i, (bid_price, bid_vol, ask_price, ask_vol) in enumerate(zip(bid_prices, bid_volumes, ask_prices, ask_volumes)):
            plt.text(bid_vol + 0.1, i, f'{bid_price:.2f}', va='center')
            plt.text(-ask_vol - 0.1, i, f'{ask_price:.2f}', va='center', ha='right')
        
        plt.yticks(range(levels), [f'Level {i+1}' for i in range(levels)])
        plt.xlabel('Volume')
        plt.title(f'Limit Order Book Depth (Sample at index {sample_idx})')
        plt.axvline(0, color='black', linestyle='-', alpha=0.7)
        plt.grid(alpha=0.3)
        plt.legend()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'lob_depth_sample.png'))
            plt.close()
        else:
            plt.show()
    
    # 2. Time series visualizations
    if 'MidPrice' in df.columns and 'time' in df.columns:
        # Sample the data to avoid overcrowding the plot
        sample_size = min(10000, len(df))
        sample_idx = np.linspace(0, len(df)-1, sample_size, dtype=int)
        sample_df = df.iloc[sample_idx]
        
        plt.figure(figsize=(14, 7))
        plt.plot(sample_df['time'], sample_df['MidPrice'], color='blue', alpha=0.8)
        plt.title('Mid-Price Time Series')
        plt.xlabel('Time')
        plt.ylabel('Mid-Price')
        plt.grid(alpha=0.3)
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'midprice_timeseries.png'))
            plt.close()
        else:
            plt.show()
        
        # Calculate returns
        if len(df) > 1:
            df_with_returns = df.copy()
            df_with_returns['returns'] = df_with_returns['MidPrice'].pct_change()
            
            # Plot return distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(df_with_returns['returns'].dropna(), kde=True, bins=100)
            plt.title('Mid-Price Returns Distribution')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, 'returns_distribution.png'))
                plt.close()
            else:
                plt.show()
            
            # Calculate return statistics
            results['return_mean'] = df_with_returns['returns'].mean()
            results['return_std'] = df_with_returns['returns'].std()
            results['return_skew'] = df_with_returns['returns'].skew()
            results['return_kurtosis'] = df_with_returns['returns'].kurtosis()
            
            logger.info(f"Return statistics - Mean: {results['return_mean']:.6f}, Std: {results['return_std']:.6f}")
            logger.info(f"Return statistics - Skew: {results['return_skew']:.6f}, Kurtosis: {results['return_kurtosis']:.6f}")
    
    # 3. Correlation analysis
    # Select a subset of important features to avoid overcrowding
    important_cols = ['MidPrice', 'BidPrice_1', 'AskPrice_1', 'BidVolume_1', 'AskVolume_1', 'Spread']
    # Add OFI columns if available
    important_cols.extend([f'OFI_{i}' for i in range(1, levels+1) if f'OFI_{i}' in df.columns])
    
    # Filter to columns that actually exist in the dataframe
    existing_cols = [col for col in important_cols if col in df.columns]
    
    if len(existing_cols) > 1:
        # Calculate correlation matrix
        corr = df[existing_cols].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
            plt.close()
        else:
            plt.show()
        
        results['correlation_matrix'] = corr.to_dict()
    
    # 4. Save results to JSON if output_dir is specified
    if output_dir is not None:
        # Convert non-serializable objects to strings
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(os.path.join(output_dir, 'exploration_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    return results


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    keep_original: bool = True
) -> pd.DataFrame:
    """
    Create lagged features for time series data.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        List of lag values (e.g., [1, 2, 3])
    keep_original : bool, default=True
        Whether to keep the original non-lagged columns
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with added lagged features
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        for lag in lags:
            result[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop original columns if not keeping them
    if not keep_original:
        result = result.drop(columns=columns)
    
    # Drop rows with NaN values created by lagging
    result = result.dropna()
    
    logger.info(f"Created lagged features with lags {lags} for columns {columns}")
    logger.info(f"New DataFrame shape: {result.shape}")
    
    return result


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    stats: List[str] = ['mean', 'std', 'min', 'max'],
    keep_original: bool = True
) -> pd.DataFrame:
    """
    Create rolling window features for time series data.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing time series data
    columns : List[str]
        Columns to create rolling features for
    windows : List[int]
        List of window sizes (e.g., [5, 10, 20])
    stats : List[str], default=['mean', 'std', 'min', 'max']
        Statistics to compute for each window
    keep_original : bool, default=True
        Whether to keep the original columns
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with added rolling features
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        for window in windows:
            rolling = df[col].rolling(window=window)
            
            for stat in stats:
                if stat == 'mean':
                    result[f'{col}_rolling_{window}_mean'] = rolling.mean()
                elif stat == 'std':
                    result[f'{col}_rolling_{window}_std'] = rolling.std()
                elif stat == 'min':
                    result[f'{col}_rolling_{window}_min'] = rolling.min()
                elif stat == 'max':
                    result[f'{col}_rolling_{window}_max'] = rolling.max()
                elif stat == 'median':
                    result[f'{col}_rolling_{window}_median'] = rolling.median()
                elif stat == 'sum':
                    result[f'{col}_rolling_{window}_sum'] = rolling.sum()
                else:
                    logger.warning(f"Unknown statistic {stat}, skipping")
    
    # Drop original columns if not keeping them
    if not keep_original:
        result = result.drop(columns=columns)
    
    # Drop rows with NaN values created by rolling windows
    result = result.dropna()
    
    logger.info(f"Created rolling features with windows {windows} and stats {stats} for columns {columns}")
    logger.info(f"New DataFrame shape: {result.shape}")
    
    return result