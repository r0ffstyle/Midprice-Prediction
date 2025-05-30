"""
Visualization module for cryptocurrency high-frequency trading models.

This module provides functions for creating various visualizations
to help understand the data, models, and results.

Author: Noah Trägårdh
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import itertools
import torch
import torch.nn as nn

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def plot_lob_state(
    df: pd.DataFrame,
    index: int,
    levels: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the state of the limit order book at a specific time.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing LOB data
    index : int
        Row index to visualize
    levels : int, default=5
        Number of price levels to show
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if index is valid
    if index < 0 or index >= len(df):
        raise ValueError(f"Index {index} is out of bounds for DataFrame with {len(df)} rows")
    
    # Extract bid and ask data at the specified index
    bid_prices = []
    bid_volumes = []
    ask_prices = []
    ask_volumes = []
    
    for i in range(1, levels + 1):
        if f'BidPrice_{i}' in df.columns and f'BidVolume_{i}' in df.columns:
            bid_prices.append(df.loc[index, f'BidPrice_{i}'])
            bid_volumes.append(df.loc[index, f'BidVolume_{i}'])
        
        if f'AskPrice_{i}' in df.columns and f'AskVolume_{i}' in df.columns:
            ask_prices.append(df.loc[index, f'AskPrice_{i}'])
            ask_volumes.append(df.loc[index, f'AskVolume_{i}'])
    
    # Get timestamp if available
    timestamp = df.loc[index, 'time'] if 'time' in df.columns else f"Index {index}"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bid side (green)
    bid_bars = ax.barh(
        range(len(bid_volumes)), 
        bid_volumes, 
        color='green', 
        alpha=0.6, 
        label='Bid'
    )
    
    # Plot ask side (red)
    ask_bars = ax.barh(
        range(len(ask_volumes)), 
        [-v for v in ask_volumes], 
        color='red', 
        alpha=0.6, 
        label='Ask'
    )
    
    # Add price labels
    for i, (bid_price, bid_vol, ask_price, ask_vol) in enumerate(zip(bid_prices, bid_volumes, ask_prices, ask_volumes)):
        ax.text(bid_vol + max(bid_volumes) * 0.02, i, f'{bid_price:.2f}', va='center')
        ax.text(-ask_vol - max(ask_volumes) * 0.02, i, f'{ask_price:.2f}', va='center', ha='right')
    
    # Set plot properties
    ax.set_yticks(range(len(bid_prices)))
    ax.set_yticklabels([f'Level {i+1}' for i in range(len(bid_prices))])
    ax.set_xlabel('Volume')
    ax.set_title(f'Limit Order Book State at {timestamp}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.7)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Set x-axis limits with some padding
    max_vol = max(max(bid_volumes), max(ask_volumes))
    ax.set_xlim(-max_vol*1.2, max_vol*1.2)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_lob_time_evolution(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    level: int = 1,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the evolution of the limit order book over time.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing LOB data
    start_index : int
        Starting row index
    end_index : int
        Ending row index
    level : int, default=1
        Order book level to visualize
    figsize : Tuple[int, int], default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Validate indices
    if start_index < 0 or start_index >= len(df):
        raise ValueError(f"Start index {start_index} is out of bounds")
    if end_index < start_index or end_index >= len(df):
        raise ValueError(f"End index {end_index} is out of bounds")
    
    # Check if required columns exist
    required_cols = [f'BidPrice_{level}', f'BidVolume_{level}', f'AskPrice_{level}', f'AskVolume_{level}']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract slice of data
    slice_df = df.iloc[start_index:end_index+1].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Get x-axis values
    if 'time' in slice_df.columns:
        x = slice_df['time']
        x_label = 'Time'
    else:
        x = slice_df.index
        x_label = 'Index'
    
    # Plot price evolution
    axes[0].plot(x, slice_df[f'BidPrice_{level}'], color='green', label=f'Bid Price {level}')
    axes[0].plot(x, slice_df[f'AskPrice_{level}'], color='red', label=f'Ask Price {level}')
    
    if 'MidPrice' in slice_df.columns:
        axes[0].plot(x, slice_df['MidPrice'], color='blue', label='Mid Price', alpha=0.7)
    
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'LOB Price Evolution (Level {level})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot volume evolution
    axes[1].plot(x, slice_df[f'BidVolume_{level}'], color='green', label=f'Bid Volume {level}')
    axes[1].plot(x, slice_df[f'AskVolume_{level}'], color='red', label=f'Ask Volume {level}')
    axes[1].set_ylabel('Volume')
    axes[1].set_title(f'LOB Volume Evolution (Level {level})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot spread if we have top-of-book data
    if all(col in slice_df.columns for col in ['BidPrice_1', 'AskPrice_1']):
        slice_df['spread'] = slice_df['AskPrice_1'] - slice_df['BidPrice_1']
        axes[2].plot(x, slice_df['spread'], color='purple')
        axes[2].set_ylabel('Spread')
        axes[2].set_title('Bid-Ask Spread Evolution')
        axes[2].grid(alpha=0.3)
    
    # Set x-axis properties
    axes[2].set_xlabel(x_label)
    
    if 'time' in slice_df.columns:
        # Format x-axis for datetime
        date_format = mdates.DateFormatter('%H:%M:%S')
        axes[2].xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_ofi_features(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    levels: List[int] = [1, 2, 3],
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the Order Flow Imbalance (OFI) features.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing LOB data with OFI features
    start_index : int
        Starting row index
    end_index : int
        Ending row index
    levels : List[int], default=[1, 2, 3]
        Order book levels to visualize
    figsize : Tuple[int, int], default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Validate indices
    if start_index < 0 or start_index >= len(df):
        raise ValueError(f"Start index {start_index} is out of bounds")
    if end_index < start_index or end_index >= len(df):
        raise ValueError(f"End index {end_index} is out of bounds")
    
    # Check if OFI columns exist
    for level in levels:
        if not all(col in df.columns for col in [f'OFI_{level}', f'bOF_{level}', f'aOF_{level}']):
            logger.warning(f"Missing OFI columns for level {level}")
            levels.remove(level)
    
    if not levels:
        raise ValueError("No valid OFI levels found in the data")
    
    # Extract slice of data
    slice_df = df.iloc[start_index:end_index+1].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(levels) + 1, 1, figsize=figsize, sharex=True)
    
    # Get x-axis values
    if 'time' in slice_df.columns:
        x = slice_df['time']
        x_label = 'Time'
    else:
        x = slice_df.index
        x_label = 'Index'
    
    # Plot mid-price
    if 'MidPrice' in slice_df.columns:
        ax_price = axes[0]
        ax_price.plot(x, slice_df['MidPrice'], color='steelblue', label='Mid Price')
        ax_price.set_ylabel('Price')
        ax_price.set_title('Mid-Price Evolution')
        ax_price.grid(alpha=0.3)
    
    # Plot OFI for each level
    for i, level in enumerate(levels):
        ax = axes[i + 1]
        
        # Plot bid and ask order flow
        ax.plot(x, slice_df[f'bOF_{level}'], color='green', label=f'Bid OF {level}', alpha=0.5)
        ax.plot(x, slice_df[f'aOF_{level}'], color='red', label=f'Ask OF {level}', alpha=0.5)
        
        # Plot OFI
        ax.plot(x, slice_df[f'OFI_{level}'], color='purple', label=f'OFI {level}', alpha=0.8)
        
        ax.set_ylabel('Order Flow')
        ax.set_title(f'Order Flow Imbalance - Level {level}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Set x-axis properties
    axes[-1].set_xlabel(x_label)
    
    if 'time' in slice_df.columns:
        # Format x-axis for datetime
        date_format = mdates.DateFormatter('%H:%M:%S')
        axes[-1].xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_alpha_term_structure(
    df: pd.DataFrame,
    num_horizons: int = 10,
    figsize: Tuple[int, int] = (12, 10),
    sample_size: int = 10000,
    random_seed: int = 666,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the alpha term structure (target variables for different horizons).
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing target variables
    num_horizons : int, default=10
        Number of horizons to visualize
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    sample_size : int, default=10000
        Number of samples to use for visualization
    random_seed : int, default=666
        Random seed for reproducible sampling
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if target columns exist
    target_cols = [f'target_horizon_{k}' for k in range(1, num_horizons + 1)]
    existing_cols = [col for col in target_cols if col in df.columns]
    
    if not existing_cols:
        raise ValueError("No target horizon columns found in the data")
    
    # Sample the data if it's large
    if len(df) > sample_size:
        np.random.seed(random_seed)
        sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
        sample_df = df.iloc[sample_indices]
    else:
        sample_df = df
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        (len(existing_cols) + 1) // 2, 2, 
        figsize=figsize, 
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()
    
    # Plot distribution for each horizon
    for i, col in enumerate(existing_cols):
        ax = axes[i]
        
        # Calculate statistics
        mean = sample_df[col].mean()
        std = sample_df[col].std()
        
        # Create histogram
        ax.hist(
            sample_df[col], 
            bins=50, 
            color='purple', 
            alpha=0.6,
            # ax=ax
        )
        
        # Add vertical lines for mean and standard deviations
        ax.axvline(mean, color='red', linestyle='-', label=f'Mean: {mean:.6f}')
        ax.axvline(mean + std, color='blue', linestyle='--', alpha=0.7, label=f'Std: {std:.6f}')
        ax.axvline(mean - std, color='blue', linestyle='--', alpha=0.7)
        
        # Add horizontal line at y=0
        ax.axhline(0, color='black', linestyle='-', alpha=0.2)
        
        # Set title and labels
        horizon_num = int(col.split('_')[-1])
        ax.set_title(f'Horizon {horizon_num}')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        
        # Add legend to first plot only
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(existing_cols), len(axes)):
        axes[i].set_visible(False)
    
    # Set overall title
    plt.suptitle('Alpha Term Structure - Return Distributions', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the training history of a model.
    
    Parameters:
    ----------
    history : Dict[str, List[float]]
        Dictionary containing training metrics
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if history contains required metrics
    required_metrics = ['train_loss', 'val_loss']
    if not all(metric in history for metric in required_metrics):
        missing = [metric for metric in required_metrics if metric not in history]
        raise ValueError(f"Missing required metrics in history: {missing}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training and validation loss
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot learning rate if available
    if 'learning_rate' in history:
        axes[1].plot(epochs, history['learning_rate'], 'g-')
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(alpha=0.3)
        # Use log scale for learning rate
        axes[1].set_yscale('log')
    else:
        # If no learning rate, plot the loss on a different scale
        axes[1].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[1].set_title('Loss (Log Scale)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_prediction_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    horizon_idx: int = 0,
    figsize: Tuple[int, int] = (14, 10),
    sample_size: int = 2000,
    random_seed: int = 666,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize model predictions versus actual values.
    
    Parameters:
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Actual target values
    horizon_idx : int, default=0
        Index of the horizon to visualize
    figsize : Tuple[int, int], default=(14, 10)
        Figure size
    sample_size : int, default=2000
        Number of samples to use for visualization
    random_seed : int, default=666
        Random seed for reproducible sampling
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if inputs have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
    
    # Check if horizon_idx is valid
    if horizon_idx < 0 or horizon_idx >= predictions.shape[1]:
        raise ValueError(f"Invalid horizon index {horizon_idx}, must be between 0 and {predictions.shape[1]-1}")
    
    # Sample the data if it's large
    if len(predictions) > sample_size:
        np.random.seed(random_seed)
        sample_indices = np.random.choice(len(predictions), size=sample_size, replace=False)
        pred_sample = predictions[sample_indices, horizon_idx]
        target_sample = targets[sample_indices, horizon_idx]
    else:
        pred_sample = predictions[:, horizon_idx]
        target_sample = targets[:, horizon_idx]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot scatter of predictions vs actual
    axes[0, 0].scatter(target_sample, pred_sample, alpha=0.5, color='blue')
    
    # Add diagonal line for perfect predictions
    min_val = min(np.min(pred_sample), np.min(target_sample))
    max_val = max(np.max(pred_sample), np.max(target_sample))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    axes[0, 0].set_title(f'Predicted vs Actual (Horizon {horizon_idx+1})')
    axes[0, 0].set_xlabel('Actual Return')
    axes[0, 0].set_ylabel('Predicted Return')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot histograms of predictions and actual values
    axes[0, 1].hist(
        target_sample, 
        bins=50, 
        alpha=0.5, 
        color='blue', 
        label='Actual'
    )
    axes[0, 1].hist(
        pred_sample, 
        bins=50, 
        alpha=0.5, 
        color='red', 
        label='Predicted'
    )
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot prediction error histogram
    error = pred_sample - target_sample
    axes[1, 0].hist(error, bins=50, color='green', alpha=0.7)
    axes[1, 0].axvline(0, color='r', linestyle='-')
    axes[1, 0].set_title('Prediction Error Distribution')
    axes[1, 0].set_xlabel('Error (Predicted - Actual)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot error statistics
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    mean_error = np.mean(error)
    std_error = np.std(error)
    
    # Calculate R^2
    ss_res = np.sum((target_sample - pred_sample) ** 2)
    ss_tot = np.sum((target_sample - np.mean(target_sample)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate sign accuracy
    sign_match = np.sign(pred_sample) == np.sign(target_sample)
    non_zero_mask = target_sample != 0
    sign_accuracy = np.mean(sign_match[non_zero_mask])
    
    # Display statistics
    stats_text = (
        f"MSE: {mse:.6f}\n"
        f"MAE: {mae:.6f}\n"
        f"Mean Error: {mean_error:.6f}\n"
        f"Std Error: {std_error:.6f}\n"
        f"R²: {r2:.6f}\n"
        f"Sign Accuracy: {sign_accuracy:.4f}"
    )
    
    axes[1, 1].text(
        0.5, 0.5, 
        stats_text, 
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[1, 1].transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    axes[1, 1].set_title('Error Statistics')
    axes[1, 1].set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_r2_by_horizon(
    r2_values: List[float],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "R² by Horizon",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize R² values for different prediction horizons.
    
    Parameters:
    ----------
    r2_values : List[float]
        R² values for each horizon
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : str, default="R² by Horizon"
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot R² values
    horizons = range(1, len(r2_values) + 1)
    
    # Create bar plot
    bars = ax.bar(horizons, r2_values, color='royalblue', alpha=0.7)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    
    # Set plot properties
    ax.set_xlabel('Horizon')
    ax.set_ylabel('R²')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    # Ensure x-axis shows integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01 if height > 0 else height - 0.05,
            f'{r2_values[i]:.4f}',
            ha='center', 
            va='bottom' if height > 0 else 'top',
            fontsize=9
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize feature importance.
    
    Parameters:
    ----------
    feature_names : List[str]
        Names of features
    importance_values : np.ndarray
        Importance values for each feature
    top_n : int, default=20
        Number of top features to show
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    title : str, default="Feature Importance"
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if inputs have the same length
    if len(feature_names) != len(importance_values):
        raise ValueError(f"Length mismatch: feature_names {len(feature_names)}, importance_values {len(importance_values)}")
    
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1]
    
    # Select top features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance_values[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(
        range(len(top_features)), 
        top_importance, 
        color='royalblue', 
        alpha=0.7
    )
    
    # Set plot properties
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # Show most important at the top
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    labels: List[str],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a correlation matrix.
    
    Parameters:
    ----------
    corr_matrix : np.ndarray
        Correlation matrix
    labels : List[str]
        Labels for the matrix
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    title : str, default="Correlation Matrix"
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if inputs have matching dimensions
    if corr_matrix.shape[0] != len(labels) or corr_matrix.shape[1] != len(labels):
        raise ValueError(f"Shape mismatch: corr_matrix {corr_matrix.shape}, labels {len(labels)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        annot=True, 
        fmt=".2f",
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": .8},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    # Set plot properties
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_trading_results(
    results: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 12),
    title: str = "Trading Strategy Results",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize trading strategy results.
    
    Parameters:
    ----------
    results : pd.DataFrame
        DataFrame containing trading results
    figsize : Tuple[int, int], default=(15, 12)
        Figure size
    title : str, default="Trading Strategy Results"
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if required columns exist
    required_cols = ['price', 'cumulative_returns', 'drawdown', 'signal', 'position']
    if not all(col in results.columns for col in required_cols):
        missing = [col for col in required_cols if col not in results.columns]
        raise ValueError(f"Missing required columns in results: {missing}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Price and signals
    ax1 = axes[0]
    
    # Plot price
    ax1.plot(results.index, results['price'], color='black', linewidth=1, label='Price')
    
    # Plot buy and sell signals
    buy_signals = results[results['signal'] == 1].index
    sell_signals = results[results['signal'] == -1].index
    
    if len(buy_signals) > 0:
        ax1.scatter(
            buy_signals, 
            results.loc[buy_signals, 'price'], 
            marker='^', 
            color='green', 
            s=100, 
            label='Buy Signal'
        )
    
    if len(sell_signals) > 0:
        ax1.scatter(
            sell_signals, 
            results.loc[sell_signals, 'price'], 
            marker='v', 
            color='red', 
            s=100, 
            label='Sell Signal'
        )
    
    # Shade regions based on position
    long_regions = results['position'] == 1
    if long_regions.any():
        ax1.fill_between(
            results.index, 
            results['price'].min(), 
            results['price'].max(), 
            where=long_regions, 
            color='green', 
            alpha=0.1
        )
    
    short_regions = results['position'] == -1
    if short_regions.any():
        ax1.fill_between(
            results.index, 
            results['price'].min(), 
            results['price'].max(), 
            where=short_regions, 
            color='red', 
            alpha=0.1
        )
    
    ax1.set_title(f'{title} - Price and Signals')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # 2. Cumulative returns
    ax2 = axes[1]
    ax2.plot(results.index, results['cumulative_returns'], color='blue', linewidth=2)
    ax2.set_title('Cumulative Returns')
    ax2.set_ylabel('Return')
    ax2.grid(alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    ax3.fill_between(
        results.index, 
        results['drawdown'], 
        0, 
        color='red', 
        alpha=0.3
    )
    ax3.set_title('Drawdown')
    ax3.set_ylabel('Drawdown')
    ax3.set_xlabel('Time')
    ax3.grid(alpha=0.3)
    
    # Calculate and display performance metrics
    total_return = results['cumulative_returns'].iloc[-1]
    max_drawdown = results['drawdown'].min()
    
    # Calculate Sharpe ratio (assumes daily returns)
    returns = results['returns'].dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    
    # Count trades
    n_trades = len(buy_signals) + len(sell_signals)
    
    metrics_text = (
        f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)\n"
        f"Sharpe Ratio: {sharpe_ratio:.4f}\n"
        f"Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)\n"
        f"Number of Trades: {n_trades}"
    )
    
    # Add text box with metrics
    ax2.text(
        0.02, 0.05, 
        metrics_text, 
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_model_comparison(
    models: List[str],
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a comparison of multiple models.
    
    Parameters:
    ----------
    models : List[str]
        List of model names
    metrics : Dict[str, List[float]]
        Dictionary of metrics for each model
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    title : str, default="Model Comparison"
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if metrics have the same length as models
    for metric_name, values in metrics.items():
        if len(values) != len(models):
            raise ValueError(f"Length mismatch for {metric_name}: {len(values)} != {len(models)}")
    
    # Create figure with subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # If only one metric, wrap axes in a list
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Create bar plot
        bars = ax.bar(
            range(len(models)), 
            values, 
            color='royalblue', 
            alpha=0.7
        )
        
        # Add values on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01 if height > 0 else height - 0.05,
                f'{values[j]:.4f}',
                ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=9
            )
        
        # Set plot properties
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title(metric_name)
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at y=0
        if min(values) < 0:
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def plot_rolling_window_results(
    window_results: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Rolling Window Results",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize results from rolling window evaluation.
    
    Parameters:
    ----------
    window_results : Dict[str, List[float]]
        Dictionary containing results for each window
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    title : str, default="Rolling Window Results"
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if required keys exist
    required_keys = ['window', 'r2_oos']
    if not all(key in window_results for key in required_keys):
        missing = [key for key in required_keys if key not in window_results]
        raise ValueError(f"Missing required keys in window_results: {missing}")
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot R² OOS values
    windows = window_results['window']
    r2_values = window_results['r2_oos']
    
    # Create line plot with markers
    ax.plot(
        windows, 
        r2_values, 
        'o-', 
        color='blue', 
        markersize=8, 
        linewidth=2
    )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Set plot properties
    ax.set_xlabel('Window')
    ax.set_ylabel('R² Out-of-Sample')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    # Ensure x-axis shows integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add average R² value
    avg_r2 = np.nanmean(r2_values)
    ax.axhline(y=avg_r2, color='g', linestyle='-', alpha=0.7, label=f'Average: {avg_r2:.4f}')
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def visualize_cnn_filters(
    model: nn.Module,
    layer_idx: int = 0,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize filters from a CNN model.
    
    Parameters:
    ----------
    model : nn.Module
        Trained CNN model
    layer_idx : int, default=0
        Index of the convolutional layer to visualize
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if model has convolutional layers
    conv_layers = [module for module in model.modules() 
                   if isinstance(module, nn.Conv2d)]
    
    if not conv_layers:
        raise ValueError("Model does not contain any Conv2d layers")
    
    if layer_idx >= len(conv_layers):
        raise ValueError(f"Layer index {layer_idx} is out of bounds (max: {len(conv_layers)-1})")
    
    # Get the specified convolutional layer
    conv_layer = conv_layers[layer_idx]
    
    # Extract filters
    filters = conv_layer.weight.data.cpu().numpy()
    
    # Get filter dimensions
    n_filters, n_channels, filter_height, filter_width = filters.shape
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    # Normalize filter values for visualization
    vmin = filters.min()
    vmax = filters.max()
    
    # Plot each filter
    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        ax = axes[row, col] if grid_size > 1 else axes
        
        if i < n_filters:
            # If filter has multiple channels, average them
            if n_channels > 1:
                filter_img = np.mean(filters[i], axis=0)
            else:
                filter_img = filters[i, 0]
            
            # Plot filter
            im = ax.imshow(
                filter_img, 
                cmap='viridis', 
                vmin=vmin, 
                vmax=vmax
            )
            
            ax.set_title(f'Filter {i+1}')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Set overall title
    plt.suptitle(f'CNN Filters - Layer {layer_idx+1} ({n_channels} channels, {filter_height}x{filter_width})', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.8)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig


def visualize_lstm_states(
    df: pd.DataFrame,
    model: nn.Module,
    sequence_indices: List[int],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize LSTM hidden states for selected sequences.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Data used for model input
    model : nn.Module
        Trained LSTM model
    sequence_indices : List[int]
        Indices of sequences to visualize
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Check if model contains an LSTM layer
    lstm_layers = [module for module in model.modules() 
                  if isinstance(module, nn.LSTM)]
    
    if not lstm_layers:
        raise ValueError("Model does not contain any LSTM layers")
    
    # Get the first LSTM layer
    lstm_layer = lstm_layers[0]
    
    # Create figure
    fig, axes = plt.subplots(len(sequence_indices), 2, figsize=figsize)
    
    # Ensure axes is 2D even with a single sequence
    if len(sequence_indices) == 1:
        axes = axes.reshape(1, -1)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process each sequence
    for i, seq_idx in enumerate(sequence_indices):
        # Extract sequence data
        if hasattr(model, 'seq_length'):
            seq_length = model.seq_length
        else:
            seq_length = 100  # Default sequence length
        
        if seq_idx + seq_length > len(df):
            raise ValueError(f"Sequence at index {seq_idx} exceeds data length")
        
        sequence = df.iloc[seq_idx:seq_idx+seq_length].values
        
        # Prepare input
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Forward pass with hooks to capture LSTM states
        hidden_states = []
        cell_states = []
        
        def hook_fn(module, input, output):
            # output[0] is the output, output[1] is a tuple of (h_n, c_n)
            hidden_states.append(output[1][0].detach().cpu().numpy())
            cell_states.append(output[1][1].detach().cpu().numpy())
        
        # Register hook
        hook = lstm_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hook
        hook.remove()
        
        # Extract states
        h_n = hidden_states[0].squeeze()  # Shape: [num_layers, hidden_size]
        c_n = cell_states[0].squeeze()    # Shape: [num_layers, hidden_size]
        
        # If h_n is 1D (single layer), reshape to 2D
        if h_n.ndim == 1:
            h_n = h_n.reshape(1, -1)
            c_n = c_n.reshape(1, -1)
        
        # Plot hidden states
        ax1 = axes[i, 0]
        im1 = ax1.imshow(h_n, cmap='viridis', aspect='auto')
        ax1.set_title(f'Hidden States (Sequence {seq_idx})')
        ax1.set_xlabel('Hidden Units')
        ax1.set_ylabel('Layer')
        fig.colorbar(im1, ax=ax1)
        
        # Plot cell states
        ax2 = axes[i, 1]
        im2 = ax2.imshow(c_n, cmap='viridis', aspect='auto')
        ax2.set_title(f'Cell States (Sequence {seq_idx})')
        ax2.set_xlabel('Hidden Units')
        ax2.set_ylabel('Layer')
        fig.colorbar(im2, ax=ax2)
    
    # Set overall title
    plt.suptitle('LSTM States Visualization', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    return fig

def plot_activation_heatmap(
    model: nn.Module,
    input_data: torch.Tensor,
    layer_name: str,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a heatmap of activations for a specific layer.
    
    Parameters:
    ----------
    model : nn.Module
        Trained model
    input_data : torch.Tensor
        Input data to feed through the model
    layer_name : str
        Name of the layer to visualize
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    -------
    plt.Figure
        Matplotlib figure
    """
    # Set model to evaluation mode
    model.eval()
    
    # Dictionary to store activations
    activations = {}
    
    # Hook function to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hook for the specified layer
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(name))
            break
    else:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_data)
    
    # Check if activations were captured
    if layer_name not in activations:
        raise ValueError(f"No activations captured for layer '{layer_name}'")
    
    # Get activations
    layer_activations = activations[layer_name]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If activations are 4D (batch, channels, height, width), use first example and average across channels
    if layer_activations.ndim == 4:
        activations_map = np.mean(layer_activations[0], axis=0)
    # If activations are 3D (batch, sequence, features), use first example
    elif layer_activations.ndim == 3:
        activations_map = layer_activations[0]
    # If activations are 2D (batch, features), use first example
    elif layer_activations.ndim == 2:
        activations_map = layer_activations[0].reshape(1, -1)
    else:
        raise ValueError(f"Unexpected activation shape: {layer_activations.shape}")
    
    # Plot heatmap
    im = ax.imshow(activations_map, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Activation Value')
    
    # Set plot properties
    ax.set_title(f'Activations for Layer: {layer_name}')
    
    # If activations are for a sequential model, add appropriate axis labels
    if activations_map.ndim == 2:
        ax.set_xlabel('Feature')
        ax.set_ylabel('Sequence Position')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig