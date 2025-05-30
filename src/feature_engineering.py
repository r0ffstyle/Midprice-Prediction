"""
Feature engineering module for cryptocurrency high-frequency trading.

This module implements the feature engineering techniques described in:
Kolm et al. (2023) - "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book"

The main features implemented are:
1. Order Flow Imbalance (OFI)
2. Alpha term structure for multi-horizon prediction
3. Additional LOB-derived features

Author: Noah Trägårdh
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging
import warnings
import yaml
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy.stats import zscore, linregress
import seaborn as sns
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numba
from numba import prange

# CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
    print("GPU acceleration enabled with CuPy!")
except ImportError:
    HAS_GPU = False
    print("CuPy not found. Running in CPU-only mode.")

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

PRICE_TOL = 1e-9          # ≈ one‑hundred‑millionth of BTC price, safe for micro‑caps too
RETURN_SCALE = 10_000     # basis‑point scaling

@numba.jit(nopython=True, parallel=True)
def _calculate_ofi_numba(bid_prices, ask_prices,
                          bid_volumes, ask_volumes,
                          levels, price_tol=PRICE_TOL):
    num_samples = bid_prices.shape[0]

    bof = np.zeros((num_samples, levels), dtype=np.float64)
    aof = np.zeros((num_samples, levels), dtype=np.float64)
    ofi = np.zeros((num_samples, levels), dtype=np.float64)

    for i in range(levels):
        for t in prange(1, num_samples):
            # ------------------------------------------------------------------
            # Bid side
            diff_bid = bid_prices[t, i] - bid_prices[t-1, i]

            if   diff_bid >  price_tol:                      # ↑ price
                bof[t, i] = bid_volumes[t, i]
            elif abs(diff_bid) <= price_tol:                 # ≈ unchanged
                bof[t, i] = bid_volumes[t, i] - bid_volumes[t-1, i]
            else:                                            # ↓ price
                bof[t, i] = -bid_volumes[t-1, i]

            # Ask side
            diff_ask = ask_prices[t, i] - ask_prices[t-1, i]

            if   diff_ask < -price_tol:                      # ask improves
                aof[t, i] = -ask_volumes[t, i]
            elif abs(diff_ask) <= price_tol:                 # ≈ unchanged
                aof[t, i] = ask_volumes[t, i] - ask_volumes[t-1, i]
            else:                                            # ask widens
                aof[t, i] = ask_volumes[t-1, i]
            # ------------------------------------------------------------------
            ofi[t, i] = bof[t, i] - aof[t, i]

    return bof, aof, ofi

def _calculate_ofi_gpu(bid_prices, ask_prices, bid_volumes, ask_volumes, levels, price_tol=PRICE_TOL):
    """GPU-accelerated OFI calculation using CuPy."""
    # Move data to GPU
    bid_prices_gpu = cp.asarray(bid_prices)
    ask_prices_gpu = cp.asarray(ask_prices)
    bid_volumes_gpu = cp.asarray(bid_volumes)
    ask_volumes_gpu = cp.asarray(ask_volumes)
    
    num_samples = bid_prices.shape[0]
    
    # Initialize result arrays on GPU
    bof_gpu = cp.zeros((num_samples, levels), dtype=np.float64)
    aof_gpu = cp.zeros((num_samples, levels), dtype=np.float64)
    ofi_gpu = cp.zeros((num_samples, levels), dtype=np.float64)
    
    # Define GPU kernel using CuPy's elementwise
    for i in range(levels):
        # ------------------------------------------------------------------
        diff_bid = bid_prices_gpu[1:, i] - bid_prices_gpu[:-1, i]
        diff_ask = ask_prices_gpu[1:, i] - ask_prices_gpu[:-1, i]

        bid_greater = diff_bid >  price_tol
        bid_equal   = cp.abs(diff_bid) <= price_tol
        bid_less    = diff_bid < -price_tol

        ask_less    = diff_ask < -price_tol      # ask price improves
        ask_equal   = cp.abs(diff_ask) <= price_tol
        ask_greater = diff_ask >  price_tol
        
        # Apply conditions - bid side
        bof_gpu[1:, i] = (
            bid_greater * bid_volumes_gpu[1:, i] +
            bid_equal * (bid_volumes_gpu[1:, i] - bid_volumes_gpu[:-1, i]) +
            bid_less * (-bid_volumes_gpu[:-1, i])
        )
        
        # Apply conditions - ask side
        aof_gpu[1:, i] = (
            ask_less * (-ask_volumes_gpu[1:, i]) +
            ask_equal * (ask_volumes_gpu[1:, i] - ask_volumes_gpu[:-1, i]) +
            ask_greater * ask_volumes_gpu[:-1, i]
        )
        
        # Calculate OFI
        ofi_gpu[:, i] = bof_gpu[:, i] - aof_gpu[:, i]
    
    # Move results back to CPU
    return cp.asnumpy(bof_gpu), cp.asnumpy(aof_gpu), cp.asnumpy(ofi_gpu)

def calculate_ofi(data: pd.DataFrame, levels: int = 10, use_gpu: bool = None, price_tol: float = PRICE_TOL) -> pd.DataFrame:
    """
    Optimized version of calculate_ofi with option for GPU acceleration.
    """
    start_time = time.time()
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = HAS_GPU
    
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    # Extract data into arrays for better performance
    num_samples = len(result)
    bid_prices = np.zeros((num_samples, levels))
    ask_prices = np.zeros((num_samples, levels))
    bid_volumes = np.zeros((num_samples, levels))
    ask_volumes = np.zeros((num_samples, levels))
    
    # Fill arrays
    for i in range(1, levels + 1):
        bid_prices[:, i-1] = result[f'BidPrice_{i}'].values
        ask_prices[:, i-1] = result[f'AskPrice_{i}'].values
        bid_volumes[:, i-1] = result[f'BidVolume_{i}'].values
        ask_volumes[:, i-1] = result[f'AskVolume_{i}'].values
    
    # Calculate OFI using GPU or CPU implementation
    if use_gpu and HAS_GPU:
        bof, aof, ofi = _calculate_ofi_gpu(
            bid_prices, ask_prices, bid_volumes, ask_volumes,
            levels, price_tol  )               # pass tol
    else:
        bof, aof, ofi = _calculate_ofi_numba(
            bid_prices, ask_prices, bid_volumes, ask_volumes,
            levels, price_tol  )               # pass tol
    
    # Add results back to DataFrame
    for i in range(1, levels + 1):
        result[f'bOF_{i}'] = bof[:, i-1]
        result[f'aOF_{i}'] = aof[:, i-1]
        result[f'OFI_{i}'] = ofi[:, i-1]
    
    # Fill NA values for the first row
    ofi_cols = []
    for i in range(1, levels + 1):
        ofi_cols.extend([f'bOF_{i}', f'aOF_{i}', f'OFI_{i}'])
    
    result.loc[0, ofi_cols] = 0
    
    elapsed = time.time() - start_time
    logger.info(f"OFI calculation completed for {levels} levels in {elapsed:.2f} seconds")
    return result
    
def _find_idx_ge(arr, value):
    """
    Return the index of the first element >= value.
    If value is past the end of the array, clamp to the last index.
    """
    idx = np.searchsorted(arr, value, side="left")
    return min(idx, len(arr)-1)

def _process_horizon_chunk(args):
    """Process a single horizon for a batch of timestamps."""
    k, timestamps, prices, latency_ms, delta_t = args
    
    # Calculate horizon time
    h_k = (k / 5) * delta_t
    
    # Sort timestamps and corresponding prices for faster lookup
    sort_indices = np.argsort(timestamps)
    sorted_timestamps = timestamps[sort_indices]
    sorted_prices = prices[sort_indices]
    
    # Initialize return array
    returns = np.full(len(timestamps), np.nan)
    
    # For each row, find future prices efficiently
    for i in range(len(timestamps)):
        # Calculate target timestamps in microseconds
        latency_ts = timestamps[i] + (latency_ms * 1000)  # Convert ms to μs
        target_ts = timestamps[i] + ((latency_ms + h_k) * 1000)  # Convert ms to μs
        
        # Find closest timestamps
        latency_idx = min(len(sorted_timestamps)-1, max(0, _find_idx_ge(sorted_timestamps, latency_ts)))
        target_idx = min(len(sorted_timestamps)-1, max(0, _find_idx_ge(sorted_timestamps, target_ts)))
        
        # Calculate return if both timestamps are valid
        if latency_idx < len(sorted_timestamps) and target_idx < len(sorted_timestamps):
            latency_price = sorted_prices[latency_idx]
            target_price = sorted_prices[target_idx]
            epsilon = 1e-12
            returns[i] = RETURN_SCALE * np.log((target_price + epsilon) / (latency_price + epsilon))
    
    return k, returns

def alpha_term_structure(
    data: pd.DataFrame, 
    latency_ms: int = 50,
    num_horizons: int = 10,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Optimized version of alpha_term_structure using multiprocessing.
    """
    start_time = time.time()
    
    # Make a copy and prepare data
    result = data.copy()
    if isinstance(result['time'].iloc[0], str):
        result['time'] = pd.to_datetime(result['time'])
    
    result['timestamp_us'] = result['time'].astype(np.int64) // 1000
    result['mid_price_change'] = result['MidPrice'].diff() != 0
    N = result['mid_price_change'].sum()
    time_span_ms = (result['timestamp_us'].max() - result['timestamp_us'].min()) / 1000
    delta_t = time_span_ms / N if N > 0 else time_span_ms
    
    # Log info
    logger.info(f"Time span of data: {time_span_ms/1000/60:.2f} minutes ({time_span_ms:.2f} ms)")
    logger.info(f"Number of price changes: {N}")
    logger.info(f"Delta t (avg time between price changes): {delta_t:.2f} ms")
    
    # Extract arrays for processing
    timestamps = result['timestamp_us'].values
    prices = result['MidPrice'].values
    
    # Determine number of processes
    if n_jobs == -1:
        n_jobs = min(multiprocessing.cpu_count(), num_horizons)
    else:
        n_jobs = min(n_jobs, num_horizons)
    
    logger.info(f"Using {n_jobs} processes for parallel computation")
    
    # Process each horizon in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Prepare arguments for each horizon
        horizon_args = [(k, timestamps, prices, latency_ms, delta_t) for k in range(1, num_horizons + 1)]
        
        # Submit tasks
        futures = {executor.submit(_process_horizon_chunk, args): args[0] for args in horizon_args}
        
        # Process results as they complete
        for i, future in enumerate(futures):
            k, returns = future.result()
            logger.info(f"Horizon {k}: {(k/5)*delta_t:.2f} ms - ({i+1}/{num_horizons} completed)")
            result[f'target_horizon_{k}'] = returns
    
    # Clean up temporary columns
    result = result.drop(['timestamp_us', 'mid_price_change'], axis=1)
    
    elapsed = time.time() - start_time
    logger.info(f"Alpha term structure calculation completed for {num_horizons} horizons in {elapsed:.2f} seconds")
    return result

def alpha_term_structure_gpu(
    data: pd.DataFrame, 
    latency_ms: int = 50, # Based on the paper "Latency Arbitrage in Cryptocurrency Markets"
    num_horizons: int = 10,
    batch_size: int = 100_000
) -> pd.DataFrame:
    """
    GPU-accelerated version of alpha_term_structure.
    
    This implementation uses CuPy to parallelize the computation across all horizons
    simultaneously on the GPU. For very large datasets, it processes in batches to
    avoid GPU memory issues.
    
    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing LOB data with 'MidPrice' and 'time' columns.
    latency_ms : int, default=10
        Latency buffer in milliseconds to account for execution delay.
    num_horizons : int, default=10
        Number of prediction horizons to generate.
    batch_size : int, default=100000
        Number of rows to process in each batch (for large datasets).
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with target variables for each horizon.
    """
    start_time = time.time()
    
    # Make a copy and prepare data
    result = data.copy()
    if isinstance(result['time'].iloc[0], str):
        result['time'] = pd.to_datetime(result['time'])
    
    # Calculate timestamps in microseconds and identify price changes
    result['timestamp_us'] = result['time'].astype(np.int64) // 1000
    result['mid_price_change'] = result['MidPrice'].diff() != 0
    N = result['mid_price_change'].sum()
    time_span_ms = (result['timestamp_us'].max() - result['timestamp_us'].min()) / 1000
    delta_t = time_span_ms / N if N > 0 else time_span_ms
    
    # Log diagnostic information
    logger.info(f"Time span of data: {time_span_ms/1000/60:.2f} minutes ({time_span_ms:.2f} ms)")
    logger.info(f"Number of price changes: {N}")
    logger.info(f"Delta t (avg time between price changes): {delta_t:.2f} ms")
    logger.info(f"Using GPU acceleration for all {num_horizons} horizons")
    
    # Extract arrays for processing
    timestamps = result['timestamp_us'].values
    prices = result['MidPrice'].values
    
    # Initialize results dictionary for all horizons
    returns_dict = {}
    
    # Process data in batches if necessary to avoid GPU memory issues
    total_rows = len(timestamps)
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    if num_batches > 1:
        logger.info(f"Processing data in {num_batches} batches to manage GPU memory")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rows)
        
        if num_batches > 1:
            logger.info(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Process this batch
        batch_timestamps = timestamps[start_idx:end_idx]
        batch_returns = _calculate_horizons_gpu(
            batch_timestamps, 
            timestamps,  # We still need all timestamps for reference
            prices, 
            latency_ms, 
            delta_t, 
            num_horizons
        )
        
        # Store batch results
        for k in range(1, num_horizons + 1):
            if k not in returns_dict:
                returns_dict[k] = np.full(total_rows, np.nan)
            returns_dict[k][start_idx:end_idx] = batch_returns[k-1]
    
    # Add results to the DataFrame
    for k in range(1, num_horizons + 1):
        result[f'target_horizon_{k}'] = returns_dict[k]
        logger.info(f"Horizon {k}: {(k/5)*delta_t:.2f} ms - completed")
    
    # Clean up temporary columns
    result = result.drop(['timestamp_us', 'mid_price_change'], axis=1)
    
    elapsed = time.time() - start_time
    logger.info(f"Alpha term structure GPU calculation completed for {num_horizons} horizons in {elapsed:.2f} seconds")
    return result

def _calculate_horizons_gpu(
    batch_timestamps: np.ndarray,
    all_timestamps:   np.ndarray,
    all_prices:       np.ndarray,
    latency_ms:       int,
    delta_t:          float,
    num_horizons:     int,
    clip_bp:          int = 10,   # ±2 % guard‑rail (basis‑points)
) -> np.ndarray:
    """GPU-accelerated horizon returns **without end-of-file clamping**.

    Any row whose *target* timestamp falls beyond the last available
    quote is marked as *NaN* (so the caller can drop it or impute). An
    optional *clip_bp* caps extreme returns at ±clip_bp bps, preventing
    outliers from dominating the loss while preserving signal direction.
    """
    # ── Move static data to GPU only once
    batch_ts_gpu  = cp.asarray(batch_timestamps)
    all_ts_gpu    = cp.asarray(all_timestamps)
    all_px_gpu    = cp.asarray(all_prices)

    # ── Pre‑sort reference arrays for fast index lookup
    sort_idx_gpu  = cp.argsort(all_ts_gpu)
    sorted_ts_gpu = all_ts_gpu[sort_idx_gpu]
    sorted_px_gpu = all_px_gpu[sort_idx_gpu]

    # ── Pre‑compute 50 ms latency offsets (µs)
    latency_ts_gpu = batch_ts_gpu + (latency_ms * 1000)

    batch_n   = len(batch_timestamps)
    out       = np.full((num_horizons, batch_n), np.nan, dtype=np.float64)
    epsilon   = 1e-12  # avoid log(0)

    for k in range(1, num_horizons + 1):
        # Time offset for this horizon (ms) ---------------------------------
        h_k_ms = (k / 5.0) * delta_t
        target_ts_gpu = batch_ts_gpu + ((latency_ms + h_k_ms) * 1000)

        # Valid rows: target timestamp ≤ last quote -------------------------
        valid_mask = target_ts_gpu <= sorted_ts_gpu[-1]
        if not cp.any(valid_mask):
            # No valid rows for this horizon in this batch → leave NaNs
            continue

        # Compute indices only where valid to avoid OOB bias ---------------
        latency_idx = _find_ge_indices_gpu(sorted_ts_gpu, latency_ts_gpu[valid_mask])
        target_idx  = _find_ge_indices_gpu(sorted_ts_gpu, target_ts_gpu [valid_mask])

        lat_px  = sorted_px_gpu[latency_idx]
        tgt_px  = sorted_px_gpu[target_idx]

        ret_gpu = RETURN_SCALE * cp.log((tgt_px + epsilon) / (lat_px + epsilon))

        # Optional clipping (on‑GPU, cheap) --------------------------------
        if clip_bp is not None:
            ret_gpu = cp.clip(ret_gpu, -clip_bp, clip_bp)

        # Copy into output --------------------------------------------------
        out[k - 1, cp.asnumpy(valid_mask)] = cp.asnumpy(ret_gpu)

    return out

def _find_ge_indices_gpu(sorted_array_gpu: cp.ndarray,
                         values_gpu:       cp.ndarray) -> cp.ndarray:
    """
    Return the index of the first element in `sorted_array_gpu` that is
    greater than or equal to the corresponding element in `values_gpu`.
    If the insertion point would be past the end of the array, clamp to
    len(array)-1.

    This guarantees a non-negative time difference.
    """
    idx = cp.searchsorted(sorted_array_gpu, values_gpu, side='left')
    return cp.minimum(idx, len(sorted_array_gpu) - 1)   # clamp right edge


def create_lob_features(data: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Create additional LOB-derived features for the model.
    
    These features capture various aspects of the limit order book structure
    and dynamics that may be predictive of future price movements.
    
    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing basic LOB data.
    levels : int, default=10
        Number of price levels to consider.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with additional features.
    """
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    # 1. Bid-Ask Spread
    result['Spread'] = result['AskPrice_1'] - result['BidPrice_1']
    
    # 2. Mid-price volatility (rolling window)
    result['midprice_volatility_10'] = result['MidPrice'].rolling(window=10).std().fillna(0)
    result['midprice_volatility_50'] = result['MidPrice'].rolling(window=50).std().fillna(0)
    
    # 3. Volume imbalance at top levels
    for i in range(1, min(levels, 6)):
        result[f'volume_imbalance_{i}'] = result[f'BidVolume_{i}'] - result[f'AskVolume_{i}']
    
    # 4. Price differences between levels (market depth)
    for i in range(1, levels):
        result[f'bid_price_diff_{i}'] = result[f'BidPrice_{i}'] - result[f'BidPrice_{i+1}']
        result[f'ask_price_diff_{i}'] = result[f'AskPrice_{i+1}'] - result[f'AskPrice_{i}']
    
    # 5. Accumulated OFI features
    if f'OFI_1' in result.columns:
        result['accumulated_OFI'] = sum(result[f'OFI_{i}'] for i in range(1, levels + 1) if f'OFI_{i}' in result.columns)
        result['accumulated_OFI_top5'] = sum(result[f'OFI_{i}'] for i in range(1, min(levels, 6)) if f'OFI_{i}' in result.columns)
    
    # 6. Bid and ask side liquidity
    result['bid_liquidity'] = sum(result[f'BidVolume_{i}'] for i in range(1, levels + 1))
    result['ask_liquidity'] = sum(result[f'AskVolume_{i}'] for i in range(1, levels + 1))
    result['liquidity_imbalance'] = result['bid_liquidity'] - result['ask_liquidity']
    
    # 7. Weighted price levels (volume-weighted)
    bid_weighted_price = 0
    ask_weighted_price = 0
    total_bid_volume = 0
    total_ask_volume = 0
    
    for i in range(1, levels + 1):
        bid_weighted_price += result[f'BidPrice_{i}'] * result[f'BidVolume_{i}']
        ask_weighted_price += result[f'AskPrice_{i}'] * result[f'AskVolume_{i}']
        total_bid_volume += result[f'BidVolume_{i}']
        total_ask_volume += result[f'AskVolume_{i}']
    
    result['weighted_bid_price'] = bid_weighted_price / total_bid_volume
    result['weighted_ask_price'] = ask_weighted_price / total_ask_volume
    result['weighted_price_imbalance'] = result['weighted_bid_price'] - result['weighted_ask_price']
    
    # 8. Recent price trends
    result['price_trend_5'] = result['MidPrice'].diff(5)
    result['price_trend_10'] = result['MidPrice'].diff(10)
    result['price_trend_20'] = result['MidPrice'].diff(20)
    
    logger.info(f"Created {len(result.columns) - len(data.columns)} additional LOB features")
    return result


def compute_stationary_features(data: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Compute stationary features from raw LOB data.
    """
    start_time = time.time()
    
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    # Get reference price (mid price)
    if 'MidPrice' not in result.columns:
        if 'AskPrice_1' in result.columns and 'BidPrice_1' in result.columns:
            result['MidPrice'] = (result['AskPrice_1'] + result['BidPrice_1']) / 2
        else:
            raise ValueError("Cannot compute mid price, required columns missing")
    
    # Prepare arrays for vectorized operations
    if HAS_GPU:
        # Use GPU for faster array operations
        try:
            # Move mid price to GPU
            mid_price_gpu = cp.asarray(result['MidPrice'].values)
            
            for i in range(1, levels + 1):
                if f'BidPrice_{i}' in result.columns:
                    # Calculate price differences on GPU
                    bid_price_gpu = cp.asarray(result[f'BidPrice_{i}'].values)
                    bid_diff_gpu = mid_price_gpu - bid_price_gpu
                    result[f'BidPriceDiff_{i}'] = cp.asnumpy(bid_diff_gpu)
                
                if f'AskPrice_{i}' in result.columns:
                    # Calculate price differences on GPU
                    ask_price_gpu = cp.asarray(result[f'AskPrice_{i}'].values)
                    ask_diff_gpu = ask_price_gpu - mid_price_gpu
                    result[f'AskPriceDiff_{i}'] = cp.asnumpy(ask_diff_gpu)
            
            # Compute volume ratios using GPU
            total_volume_gpu = cp.zeros_like(mid_price_gpu)
            for i in range(1, levels + 1):
                if f'BidVolume_{i}' in result.columns and f'AskVolume_{i}' in result.columns:
                    bid_vol_gpu = cp.asarray(result[f'BidVolume_{i}'].values)
                    ask_vol_gpu = cp.asarray(result[f'AskVolume_{i}'].values)
                    total_volume_gpu += bid_vol_gpu + ask_vol_gpu
            
            # Avoid division by zero
            total_volume_gpu = cp.maximum(total_volume_gpu, 1e-10)
            
            for i in range(1, levels + 1):
                if f'BidVolume_{i}' in result.columns:
                    bid_vol_gpu = cp.asarray(result[f'BidVolume_{i}'].values)
                    result[f'BidVolumeRatio_{i}'] = cp.asnumpy(bid_vol_gpu / total_volume_gpu)
                
                if f'AskVolume_{i}' in result.columns:
                    ask_vol_gpu = cp.asarray(result[f'AskVolume_{i}'].values)
                    result[f'AskVolumeRatio_{i}'] = cp.asnumpy(ask_vol_gpu / total_volume_gpu)
        
        except Exception as e:
            logger.warning(f"GPU calculation failed: {e}. Falling back to CPU implementation.")
            # Fall back to CPU implementation
            _compute_stationary_features_cpu(result, levels)
    else:
        # Use CPU implementation
        _compute_stationary_features_cpu(result, levels)
    
    elapsed = time.time() - start_time
    logger.info(f"Computed stationary features in {elapsed:.2f} seconds")
    return result

def _compute_stationary_features_cpu(result: pd.DataFrame, levels: int):
    """CPU implementation of stationary features calculation."""
    # Compute price differences relative to mid price
    for i in range(1, levels + 1):
        if f'BidPrice_{i}' in result.columns:
            result[f'BidPriceDiff_{i}'] = result['MidPrice'] - result[f'BidPrice_{i}']
        
        if f'AskPrice_{i}' in result.columns:
            result[f'AskPriceDiff_{i}'] = result[f'AskPrice_{i}'] - result['MidPrice']
    
    # Compute volume ratios
    total_volume = 0
    for i in range(1, levels + 1):
        if f'BidVolume_{i}' in result.columns and f'AskVolume_{i}' in result.columns:
            total_volume += result[f'BidVolume_{i}'] + result[f'AskVolume_{i}']
    
    # Avoid division by zero
    total_volume = total_volume.replace(0, 1e-10)
    
    for i in range(1, levels + 1):
        if f'BidVolume_{i}' in result.columns:
            result[f'BidVolumeRatio_{i}'] = result[f'BidVolume_{i}'] / total_volume
        
        if f'AskVolume_{i}' in result.columns:
            result[f'AskVolumeRatio_{i}'] = result[f'AskVolume_{i}'] / total_volume

def prepare_features(
    data: pd.DataFrame, 
    latency_ms: int = 50,
    levels: int = 10,
    normalize: bool = True,
    visualize: bool = False,
    n_jobs: int = -1,
    use_gpu: bool = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimized version of prepare_features with GPU acceleration option.
    """
    start_time = time.time()
    logger.info("Starting optimized feature preparation...")
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = HAS_GPU
    
    # Step 1: Calculate Order Flow Imbalance
    logger.info("Calculating Order Flow Imbalance with optimization...")
    df = calculate_ofi(data, levels=levels, use_gpu=use_gpu)
    
    # Step 2: Calculate the targets (alpha term structure)
    logger.info("Calculating alpha term structure with optimization...")
    df_with_targets = alpha_term_structure(
        df, latency_ms=latency_ms, num_horizons=10, n_jobs=n_jobs
    )
    
    # Step 3: Split into features and targets
    target_cols = [f'target_horizon_{k}' for k in range(1, 11)]
    
    features_df = df_with_targets.drop(target_cols, axis=1, errors='ignore')
    targets_df = df_with_targets[target_cols]
    
    # Step 4: Normalize features if requested
    if normalize:
        logger.info("Normalizing features...")
        # Skip non-numeric columns and certain columns that shouldn't be normalized
        skip_cols = ['time', 'instrument']
        numeric_cols = [col for col in features_df.columns 
                        if col not in skip_cols and np.issubdtype(features_df[col].dtype, np.number)]
        
        # Use GPU for normalization if available and requested
        if use_gpu and HAS_GPU:
            logger.info("Using GPU for feature normalization")
            # Process in chunks to avoid GPU memory issues
            chunk_size = 100  # Number of columns to process at once
            for i in range(0, len(numeric_cols), chunk_size):
                chunk_cols = numeric_cols[i:i+chunk_size]
                chunk_data = features_df[chunk_cols].values
                
                # Move to GPU, normalize, and move back
                chunk_data_gpu = cp.asarray(chunk_data)
                mean = cp.mean(chunk_data_gpu, axis=0)
                std = cp.std(chunk_data_gpu, axis=0)
                std = cp.maximum(std, 1e-10)  # Avoid division by zero
                chunk_data_normalized = cp.asnumpy((chunk_data_gpu - mean) / std)
                
                # Update DataFrame
                for j, col in enumerate(chunk_cols):
                    features_df[col] = chunk_data_normalized[:, j]
        else:
            features_df[numeric_cols] = features_df[numeric_cols].apply(zscore, nan_policy='omit')
    
    # Step 5: Drop rows with missing values in targets
    valid_rows = ~targets_df.isna().any(axis=1)
    features_df = features_df[valid_rows].reset_index(drop=True)
    targets_df = targets_df[valid_rows].reset_index(drop=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Feature preparation completed in {elapsed:.2f} seconds")
    logger.info(f"Final dataset shape: {features_df.shape} features, {targets_df.shape} targets")
    
    return features_df, targets_df

def hybrid_alpha_term_structure(
    data: pd.DataFrame, 
    latency_ms: int = 50,
    num_horizons: int = 10,
    n_jobs: int = -1,
    use_gpu: bool = None,
    batch_size: int = 100000
) -> pd.DataFrame:
    """
    Hybrid implementation of alpha_term_structure that can use either CPU or GPU.
    
    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing LOB data
    latency_ms : int, default=10
        Latency buffer in milliseconds
    num_horizons : int, default=10
        Number of prediction horizons
    n_jobs : int, default=-1
        Number of CPU processes to use (for CPU implementation)
    use_gpu : bool, default=None
        Whether to use GPU (None = auto-detect)
    batch_size : int, default=100000
        Batch size for GPU processing
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with target variables
    """
    import sys
    
    # Try to import GPU dependencies
    try:
        import cupy as cp
        has_gpu = True
    except ImportError:
        has_gpu = False
    
    # Determine whether to use GPU
    if use_gpu is None:
        use_gpu = has_gpu
    elif use_gpu and not has_gpu:
        logger.warning("GPU acceleration requested but CuPy not available. Falling back to CPU.")
        use_gpu = False
    
    # Use GPU implementation if available and requested
    if use_gpu and has_gpu:
        try:
            return alpha_term_structure_gpu(data, latency_ms, num_horizons, batch_size)
        except Exception as e:
            logger.warning(f"GPU implementation failed with error: {e}. Falling back to CPU.")
            return alpha_term_structure(data, latency_ms, num_horizons, n_jobs)
    else:
        return alpha_term_structure(data, latency_ms, num_horizons, n_jobs)

def prepare_features_with_gpu(
    data: pd.DataFrame, 
    latency_ms: int = 50,
    levels: int = 10,
    normalize: bool = True,
    visualize: bool = False,
    n_jobs: int = -1,
    use_gpu: bool = None,
    batch_size: int = 100_000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features with GPU-accelerated alpha term structure.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Raw LOB data
    latency_ms : int, default=10
        Latency buffer in milliseconds
    levels : int, default=10
        Number of price levels to consider
    normalize : bool, default=True
        Whether to normalize features using z-score
    visualize : bool, default=False
        Whether to generate visualizations
    n_jobs : int, default=-1
        Number of processes to use (-1 means use all available cores)
    use_gpu : bool, default=None
        Whether to use GPU acceleration (None = auto-detect)
    batch_size : int, default=100000
        Batch size for GPU processing of large datasets
        
    Returns:
    -------
    tuple: (features_df, targets_df)
        Processed feature and target DataFrames
    """
    start_time = time.time()
    logger.info("Starting optimized feature preparation with GPU alpha term structure...")
    
    # Try to import GPU dependencies
    try:
        import cupy as cp
        has_gpu = True
        if use_gpu is None:
            use_gpu = True
        logger.info("CuPy is available for GPU acceleration")
    except ImportError:
        has_gpu = False
        use_gpu = False
        logger.info("CuPy not available, using CPU-only mode")
    
    # Step 1: Calculate Order Flow Imbalance
    logger.info("Calculating Order Flow Imbalance with optimization...")
    df = calculate_ofi(data, levels=levels, use_gpu=use_gpu)
    
    # Step 2: Calculate the targets (alpha term structure) using hybrid implementation
    logger.info("Calculating alpha term structure with GPU acceleration...")
    df_with_targets = hybrid_alpha_term_structure(
        df, 
        latency_ms=latency_ms, 
        num_horizons=10, 
        n_jobs=n_jobs,
        use_gpu=use_gpu,
        batch_size=batch_size
    )
    
    # Step 3: Split into features and targets
    target_cols = [f'target_horizon_{k}' for k in range(1, 11)]
    
    features_df = df_with_targets.drop(target_cols, axis=1, errors='ignore')
    targets_df = df_with_targets[target_cols]
    
    # Step 4: Normalize features if requested
    if normalize:
        logger.info("Normalizing features...")
        # Skip non-numeric columns and certain columns that shouldn't be normalized
        skip_cols = ['time', 'instrument']
        numeric_cols = [col for col in features_df.columns 
                        if col not in skip_cols and np.issubdtype(features_df[col].dtype, np.number)]
        
        # Use GPU for normalization if available and requested
        if use_gpu and has_gpu:
            logger.info("Using GPU for feature normalization")
            try:
                # Process in chunks to avoid GPU memory issues
                chunk_size = 100  # Number of columns to process at once
                for i in range(0, len(numeric_cols), chunk_size):
                    chunk_cols = numeric_cols[i:i+chunk_size]
                    chunk_data = features_df[chunk_cols].values
                    
                    # Move to GPU, normalize, and move back
                    chunk_data_gpu = cp.asarray(chunk_data)
                    mean = cp.mean(chunk_data_gpu, axis=0)
                    std = cp.std(chunk_data_gpu, axis=0)
                    std = cp.maximum(std, 1e-10)  # Avoid division by zero
                    chunk_data_normalized = cp.asnumpy((chunk_data_gpu - mean) / std)
                    
                    # Update DataFrame
                    for j, col in enumerate(chunk_cols):
                        features_df[col] = chunk_data_normalized[:, j]
            except Exception as e:
                logger.warning(f"GPU normalization failed: {e}. Falling back to CPU implementation.")
                features_df[numeric_cols] = features_df[numeric_cols].apply(zscore, nan_policy='omit')
        else:
            features_df[numeric_cols] = features_df[numeric_cols].apply(zscore, nan_policy='omit')
    
    # Step 5: Drop rows with missing values in targets
    valid_rows = ~targets_df.isna().any(axis=1)
    features_df = features_df[valid_rows].reset_index(drop=True)
    targets_df = targets_df[valid_rows].reset_index(drop=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Feature preparation completed in {elapsed:.2f} seconds")
    logger.info(f"Final dataset shape: {features_df.shape} features, {targets_df.shape} targets")
    
    return features_df, targets_df