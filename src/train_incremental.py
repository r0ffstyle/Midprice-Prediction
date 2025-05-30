"""
Unified training module for cryptocurrency high-frequency trading.

This module provides memory-optimized training with checkpoint/resume capability.

Based on methodology from:
Kolm et al. (2023) - "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book"
"""

import os
import yaml
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gc
import pyarrow.parquet as pq
from datetime import datetime
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model_architectures import get_model, OptimizedLOBDataset
from src.evaluation import evaluate_model
from src.visualization import plot_r2_by_horizon, plot_training_history

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def convert_numpy_to_python(obj):
    """Convert NumPy arrays and types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    This class monitors a validation metric and stops training when the metric
    stops improving for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Initialize the early stopping handler.
        
        Parameters:
        ----------
        patience : int, default=10
            Number of epochs to wait after last improvement
        min_delta : float, default=0.0
            Minimum change to qualify as improvement
        mode : str, default='min'
            'min' for metrics that should decrease, 'max' for metrics that should increase
        verbose : bool, default=True
            Whether to print messages
        save_path : str, optional
            Path to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda score, best: score <= (best - min_delta)
        elif mode == 'max':
            self.is_better = lambda score, best: score >= (best + min_delta)
        else:
            raise ValueError(f"Mode {mode} not recognized; use 'min' or 'max'")
            
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped and save the model if it's the best so far.
        
        Parameters:
        ----------
        score : float
            Current validation metric value
        model : nn.Module
            Model to save if it's the best
            
        Returns:
        -------
        bool
            True if training should be stopped, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(score, model)
        elif self.is_better(score, self.best_score):
            # Improvement found
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def save_checkpoint(self, score: float, model: nn.Module):
        """Save the model checkpoint if it's the best so far."""
        if self.save_path is not None:
            if self.verbose:
                logger.info(f"Validation score improved to {score:.6f}. Saving model to {self.save_path}")
            torch.save(model.state_dict(), self.save_path, _use_new_zipfile_serialization=True)
            
    def load_best_model(self, model: nn.Module):
        """Load the best model checkpoint."""
        if self.save_path is not None and os.path.exists(self.save_path):
            model.load_state_dict(torch.load(self.save_path, weights_only=True))
            return model
        else:
            logger.warning("No checkpoint found, returning current model")
            return model


def set_gpu_limits(memory_limit_percentage=0.9, utilization_target=0.8):
    """
    Set limits on GPU usage to prevent system crashes.
    
    Parameters:
    ----------
    memory_limit_percentage : float, default=0.9
        Percentage of GPU memory to use (0.0 to 1.0)
    utilization_target : float, default=0.8
        Target GPU utilization percentage (0.0 to 1.0)
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU limit settings")
        return

    # Set memory growth to limit memory usage
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        try:
            # Get total memory
            total_memory = torch.cuda.get_device_properties(i).total_memory
            
            # Calculate memory limit
            max_memory = int(total_memory * memory_limit_percentage)
            
            # Limit memory usage by allocating a smaller tensor first
            # This reserves a portion of memory but doesn't use it
            torch.cuda.set_per_process_memory_fraction(memory_limit_percentage, i)
            
            print(f"GPU {i}: Memory limited to {memory_limit_percentage*100:.1f}% of total ({max_memory/(1024**2):.1f} MB)")
        except Exception as e:
            print(f"Could not set memory limit for GPU {i}: {e}")

    # Try to set environment variables to control utilization
    try:
        # Set environment variables for CUDA to limit utilization
        # This doesn't work for all GPUs but might help on some systems
        import os
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        # For some NVIDIA GPUs, this might help limit kernel execution time
        if torch.cuda.get_device_name(0).find('NVIDIA') != -1:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        print("Set environment variables to help control GPU utilization")
    except Exception as e:
        print(f"Could not set environment variables: {e}")


def save_plot_data(data, directory, filename, format='json'):
    """
    Save data for later plotting in organized directories.
    
    Parameters:
    ----------
    data : dict or array-like
        Data to save (history dict, r2 values, etc.)
    directory : str
        Directory to save the data in
    filename : str
        Name of the file without extension
    format : str, default='json'
        Format to save the data in ('json' or 'npy')
    """
    import os
    import json
    import numpy as np
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save based on format
    if format == 'json':
        # Handle NumPy arrays before saving to JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
                
        # Convert NumPy types to Python native types
        data_converted = convert_numpy(data)
        
        with open(os.path.join(directory, f"{filename}.json"), 'w') as f:
            json.dump(data_converted, f, indent=4)
    elif format == 'npy':
        # Convert to numpy array if it's a list
        if isinstance(data, list):
            data = np.array(data)
        np.save(os.path.join(directory, f"{filename}.npy"), data)
    else:
        raise ValueError(f"Unsupported format: {format}")

def fit_arx_optimized(
    y_train: pd.Series,
    X_ofi_train: pd.DataFrame,
    y_val: pd.Series,
    X_ofi_val: pd.DataFrame,
    y_test: pd.Series,
    X_ofi_test: pd.DataFrame,
    ny: int = 10,
    nx: int = 10,
    max_rows: int = 100000
) -> Dict[str, np.ndarray]:
    """
    Optimized ARX model fitting that avoids DataFrame fragmentation.
    Uses pd.concat for better performance and reduced memory usage.
    Calculates R²OOS against mean return baseline for comparison with neural networks.
    """
    # Sample all datasets, not just training
    def sample_data(y_series, X_df, max_size):
        if len(y_series) > max_size:
            print(f"Sampling {len(y_series):,} rows to {max_size:,} rows")
            sample_idx = np.random.choice(len(y_series), max_size, replace=False)
            return y_series.iloc[sample_idx], X_df.iloc[sample_idx]
        return y_series, X_df

    # Sample all datasets
    y_train, X_ofi_train = sample_data(y_train, X_ofi_train, max_rows)
    y_val, X_ofi_val = sample_data(y_val, X_ofi_val, max_rows)
    y_test, X_ofi_test = sample_data(y_test, X_ofi_test, max_rows)
    
    print(f"After sampling - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} rows")
    
    # Better lag creation that avoids fragmentation
    def make_lags_optimized(y, X, ny, nx):
        print("  Creating lag features...")
        # List to hold individual DataFrames
        dfs_to_concat = []
        
        # Create y lags in a single DataFrame
        y_lags = pd.DataFrame(index=y.index)
        for i in range(1, ny+1):
            y_lags[f"y_lag{i}"] = y.shift(i)
        dfs_to_concat.append(y_lags)
        del y_lags
        
        # Create feature lags one feature at a time
        for col_idx, col in enumerate(X.columns):
            print(f"    Processing feature {col_idx+1}/{len(X.columns)}: {col}")
            # Create a separate DataFrame for this feature's lags
            feature_lags = pd.DataFrame(index=X.index)
            for j in range(1, nx+1):
                feature_lags[f"{col}_lag{j}"] = X[col].shift(j)
            
            # Add to list of DataFrames to concatenate
            dfs_to_concat.append(feature_lags)
            # Free memory
            del feature_lags
            gc.collect()
        
        # Concatenate all lag DataFrames at once
        print("  Combining all lag features...")
        result = pd.concat(dfs_to_concat, axis=1)
        del dfs_to_concat
        gc.collect()
        return result
    
    # Process datasets one at a time to save memory
    print("Processing training data...")
    train_lags = make_lags_optimized(y_train, X_ofi_train, ny, nx)
    train_df = pd.concat([y_train.rename("y"), train_lags], axis=1).dropna()
    Xtr, ytr = train_df.drop("y", axis=1), train_df["y"]
    # Free memory
    del train_lags, train_df, y_train, X_ofi_train
    gc.collect()
    
    print("Processing validation data...")
    val_lags = make_lags_optimized(y_val, X_ofi_val, ny, nx)
    val_df = pd.concat([y_val.rename("y"), val_lags], axis=1).dropna()
    Xv, yv = val_df.drop("y", axis=1), val_df["y"]
    # Free memory
    del val_lags, val_df, y_val, X_ofi_val
    gc.collect()
    
    print("Processing test data...")
    test_lags = make_lags_optimized(y_test, X_ofi_test, ny, nx)
    test_df = pd.concat([y_test.rename("y"), test_lags], axis=1).dropna()
    Xt, yt = test_df.drop("y", axis=1), test_df["y"]
    # Free memory - keep only what we need
    del test_lags, y_test, X_ofi_test
    gc.collect()
    
    # Fit model
    print(f"Fitting linear regression model on {len(ytr)} samples with {Xtr.shape[1]} features...")
    arx = LinearRegression(fit_intercept=True)
    arx.fit(Xtr, ytr)
    
    # Make predictions
    print("Making predictions...")
    pred_train = arx.predict(Xtr)
    del Xtr
    gc.collect()
    
    pred_val = arx.predict(Xv)
    del Xv
    gc.collect()
    
    pred_test = arx.predict(Xt)
    
    # Calculate standard metrics
    r2_train = r2_score(ytr, pred_train)
    r2_val = r2_score(yv, pred_val)
    r2_test = r2_score(yt, pred_test)
    
    # Calculate R^2 Out-of-Sample vs mean return (same as neural networks)
    print("Calculating R^2 OOS against mean return baseline...")
    
    # MSE for model predictions
    model_mse = np.mean((yt.values - pred_test) ** 2)
    
    # MSE for mean return baseline (same as used in neural network evaluation)
    mean_return = np.mean(yt.values)
    mean_baseline_mse = np.mean((yt.values - mean_return) ** 2)
    
    # Calculate R^2 OOS
    r2_oos = 1 - (model_mse / mean_baseline_mse)
    
    # Calculate sign accuracy (directional accuracy)
    matching_signs = np.sign(pred_test) == np.sign(yt.values)
    non_zero_mask = yt.values != 0
    sign_accuracy = np.mean(matching_signs[non_zero_mask]) if np.any(non_zero_mask) else np.nan
    
    # Calculate MSE for reporting
    # mse = mean_squared_error(yt.values, pred_test)
    
    print(f"Results: Train R²={r2_train:.4f}, Val R²={r2_val:.4f}, Test R²={r2_test:.4f}")
    print(f"R² OOS vs Mean Return: {r2_oos:.4f} (comparable to neural networks)")
    # print(f"MSE: {mse:.6f}, Sign Accuracy: {sign_accuracy:.4f}")
    
    # Clean up remaining memory
    del test_df, pred_train, pred_val
    gc.collect()
    
    return {
        "model": arx,
        "r2_train": r2_train,
        "r2_val": r2_val,
        "r2_test": r2_test,
        "r2_oos": r2_oos,         # vs mean return (same as NN evaluation)
        # "mse": mse,
        "sign_accuracy": sign_accuracy,
        "y_test": yt.values,
        "y_pred": pred_test,
    }

def modified_train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
    use_amp: bool = True,  # Enable mixed precision by default
    grad_accumulation: int = 1,  # Number of batches to accumulate gradients
    sleep_time: float = 0.01  # Add small sleep between batches to reduce GPU load
) -> Dict[str, List[float]]:
    """
    Train a model with resource-limited settings to prevent crashes.
    
    Added features:
    - Gradient accumulation to reduce memory pressure
    - Small sleeps between batches to give the GPU time to cool down
    - More careful memory management
    
    Parameters:
    ----------
    (Same as train_model with these additions)
    grad_accumulation : int, default=1
        Number of batches to accumulate gradients before updating weights
    sleep_time : float, default=0.01
        Time to sleep between batches (seconds) to reduce GPU pressure
    """
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize scaler for mixed precision training
    scaler = torch.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Use tqdm for progress bar if verbose
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)") if verbose else train_loader
        
        # For gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        for inputs, targets in train_iter:
            # Move data to device
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    # Scale loss by accumulation steps
                    loss = criterion(outputs, targets) / grad_accumulation
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                
                # Only update weights after accumulating gradients
                if (batch_count + 1) % grad_accumulation == 0 or (batch_count + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                # Standard precision (fallback)
                outputs = model(inputs)
                # Scale loss by accumulation steps
                loss = criterion(outputs, targets) / grad_accumulation
                loss.backward()
                
                # Only update weights after accumulating gradients
                if (batch_count + 1) % grad_accumulation == 0 or (batch_count + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            # Update statistics (scale back to get true loss)
            train_loss += (loss.item() * grad_accumulation) * inputs.size(0)
            batch_count += 1
            
            # Explicitly clear some memory
            del outputs, loss
            
            # Small sleep to reduce GPU pressure
            if sleep_time > 0:
                import time
                time.sleep(sleep_time)
                
            # Periodically force CUDA synchronization to prevent memory buildup
            if batch_count % 10 == 0 and device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Free memory before validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Use tqdm for progress bar if verbose
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)") if verbose else val_loader
        
        with torch.no_grad():
            for inputs, targets in val_iter:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Use mixed precision for validation as well
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item() * inputs.size(0)
                
                # Explicitly clear some memory
                del outputs, loss
                
                # Small sleep to reduce GPU pressure
                if sleep_time > 0:
                    import time
                    time.sleep(sleep_time)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        if scheduler is not None:
            # Different scheduler types need different parameters
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print epoch results
        if verbose:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.8f}"
            )
            
            # Log if learning rate was changed by scheduler
            if epoch > 0 and scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                prev_lr = history['learning_rate'][-2]
                if current_lr < prev_lr:
                    logger.info(f"Learning rate reduced from {prev_lr:.8f} to {current_lr:.8f}")
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                if verbose:
                    logger.info("Early stopping triggered")
                break
        
        # Force memory cleanup after each epoch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Load the best model if early stopping was used
    if early_stopping is not None:
        model = early_stopping.load_best_model(model)
    
    return history


def train_symbol_months_time_based_limited(symbol, year, months, model_type="cnn", config_path="config/model_config.yaml", 
                             data_config_path="config/data_config.yaml", resume=False, chunk_size=1_000_000,
                             min_window_days=30, gpu_memory_limit=0.8, gpu_util_target=0.8, 
                             batch_size_factor=0.75, grad_accumulation=2, sleep_time=0.005, ofi_only=False):
    """
    Train a model with resource limits to prevent system crashes.
    
    This version limits GPU memory usage, adds gradient accumulation,
    and uses smaller batch sizes to reduce system load.
    
    Additional Parameters:
    ----------
    gpu_memory_limit : float, default=0.8
        Percentage of GPU memory to use (0.0 to 1.0)
    gpu_util_target : float, default=0.8
        Target GPU utilization percentage (0.0 to 1.0)
    batch_size_factor : float, default=0.75
        Factor to reduce batch size by (relative to config)
    grad_accumulation : int, default=2
        Number of batches to accumulate gradients before updating weights
    sleep_time : float, default=0.005
        Time to sleep between batches (seconds) to reduce GPU pressure
    ofi_only : bool, default=False
        If True, use only OFI features. If False, use only raw LOB data.
    """
    
    # Set GPU limits to prevent crashes
    set_gpu_limits(memory_limit_percentage=gpu_memory_limit, utilization_target=gpu_util_target)
    
    print("Starting training process with resource limits...")
    print(f"Feature set: {'OFI only' if ofi_only else 'LOB only'}")

    # Load configurations
    print("Loading configurations...")
    with open(config_path, 'r') as file:
        model_config = yaml.safe_load(file)
    
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
    print("Configurations loaded successfully")
    
    # Format months for output directory naming
    months_str = f"{min(months):02d}-{max(months):02d}" if len(months) > 1 else f"{months[0]:02d}"
    
    # Create model output directory with feature mode in the path
    feature_mode = "ofi_only" if ofi_only else "lob_only"
    model_output_dir = os.path.join(model_config['general']['models_dir'], symbol, f"{year}-{months_str}", model_type, feature_mode)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Create results directory
    results_dir = os.path.join(model_config['general']['output_dir'], symbol, f"{year}-{months_str}", model_type, feature_mode)
    os.makedirs(results_dir, exist_ok=True)
    
    # Checkpoint file for saving/resuming training state
    checkpoint_file = os.path.join(model_output_dir, "training_state.json")
    
    # Check if we're resuming
    start_window = 0
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            training_state = json.load(f)
            start_window = training_state.get('last_completed_window', 0) + 1
            print(f"Resuming training from window {start_window}")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up training parameters
    model_params = model_config[model_type]
    seq_length = data_config['sequence']['length']
    
    # Reduce batch size to prevent out of memory errors
    orig_batch_size = model_params['training']['batch_size']
    batch_size = max(1, int(orig_batch_size * batch_size_factor))
    print(f"Reducing batch size from {orig_batch_size} to {batch_size} (factor: {batch_size_factor})")
    print(f"Using gradient accumulation of {grad_accumulation} steps (effective batch: {batch_size * grad_accumulation})")
    
    num_epochs = model_params['training']['num_epochs']
    learning_rate = model_params['training']['learning_rate']
    weight_decay = model_params['training']['weight_decay']
    
    # Get split parameters from config
    timestamp_col = model_config['rolling_window'].get('timestamp_col', 'time')
    train_days = model_config['rolling_window'].get('train_days', 21)
    val_days = model_config['rolling_window'].get('val_days', 7)
    test_days = model_config['rolling_window'].get('test_days', 7)
    
    # Verify that our minimum window requirement is sufficient
    required_days = train_days + val_days + test_days
    if min_window_days < required_days:
        min_window_days = required_days
        print(f"Warning: Adjusted min_window_days to {min_window_days} to accommodate train/val/test periods")
    
    # Store model construction parameters based on model type
    if model_type == 'cnn':
        model_kwargs = {
            'conv_channels': model_params['architecture']['conv_channels'],
            'fc_units': model_params['architecture']['fc_units'],
            'dropout': model_params['architecture']['dropout']
        }
    elif model_type == 'lstm':
        model_kwargs = {
            'hidden_dim': model_params['architecture']['hidden_dim'],
            'num_layers': model_params['architecture']['num_layers'],
            'dropout': model_params['architecture']['dropout'],
            'bidirectional': model_params['architecture']['bidirectional']
        }
    elif model_type == 'cnn_lstm':
        model_kwargs = {
            'conv_channels': model_params['architecture']['conv_channels'],
            'lstm_hidden_dim': model_params['architecture']['lstm_hidden_dim'],
            'lstm_layers': model_params['architecture']['lstm_layers'],
            'fc_units': model_params['architecture']['fc_units'],
            'dropout': model_params['architecture']['dropout']
        }
    
    # Results tracking
    all_results = {}
    
    # Prepare to track each month's data info
    month_data_info = []
    
    # First pass: scan all months to gather basic info
    print("Scanning months to gather information...")
    for month in months:
        # Check if the parquet file exists for this month
        processed_data_dir = os.path.join(data_config['data']['output_dir'], symbol, f"{year}-{month:02d}")
        processed_data_path = os.path.join(processed_data_dir, f"{symbol}_{year}-{month:02d}_processed.parquet")
        
        if not os.path.exists(processed_data_path):
            print(f"Warning: Processed data not found at {processed_data_path}")
            print(f"Skipping month {month:02d}")
            continue
        
        # Read metadata
        print(f"Reading metadata for month {month:02d}...")
        parquet_file = pq.ParquetFile(processed_data_path)
        total_rows = parquet_file.metadata.num_rows
        
        # Verify the timestamp column exists
        has_timestamp = timestamp_col in parquet_file.schema.names
        if not has_timestamp:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in {os.path.basename(processed_data_path)}. Cannot use time-based splitting.")
        
        print(f"Month {month:02d}: {total_rows:,} rows, timestamp column verified")
        
        # Now read a small sample of rows to get timestamp information
        print(f"Reading timestamp samples for month {month:02d}...")
        
        # Read the first batch to check timestamps
        batch = next(parquet_file.iter_batches(batch_size=min(10000, total_rows)))
        sample_df = batch.to_pandas()
        
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(sample_df[timestamp_col])
        
        # Get min timestamp from this sample
        min_time_sample = timestamps.min()
        
        # Read the last batch to get max timestamp
        # First calculate how many rows to skip
        rows_to_skip = max(0, total_rows - 10000)
        last_batch = None
        
        # Use a counter to track batches
        row_count = 0
        for batch in parquet_file.iter_batches(batch_size=10000):
            row_count += batch.num_rows
            if row_count > rows_to_skip:
                last_batch = batch
                break
        
        if last_batch is not None:
            last_sample_df = last_batch.to_pandas()
            last_timestamps = pd.to_datetime(last_sample_df[timestamp_col])
            max_time_sample = last_timestamps.max()
        else:
            # If we couldn't get the last batch, use the max from first batch
            max_time_sample = timestamps.max()
        
        # Calculate approximate time span in days
        time_span_days = (max_time_sample - min_time_sample).total_seconds() / (60 * 60 * 24)
        
        print(f"Month {month:02d}: Time span: {time_span_days:.2f} days " 
              f"({min_time_sample.strftime('%Y-%m-%d %H:%M:%S')} to {max_time_sample.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Store month info
        month_data_info.append({
            'month': month,
            'total_rows': total_rows,
            'file_path': processed_data_path,
            'min_time': min_time_sample,
            'max_time': max_time_sample,
            'time_span_days': time_span_days
        })
    
    if not month_data_info:
        raise ValueError("No valid data found for any month.")
    
    # Calculate global time range
    all_min_times = [info['min_time'] for info in month_data_info]
    all_max_times = [info['max_time'] for info in month_data_info]
    earliest_time = min(all_min_times)
    latest_time = max(all_max_times)
    
    # Calculate total time span in days
    total_time_span_days = (latest_time - earliest_time).total_seconds() / (60 * 60 * 24)
    
    print(f"Total time span across all months: {total_time_span_days:.2f} days")
    
    # Check if we have enough days for at least one window
    if total_time_span_days < min_window_days:
        raise ValueError(f"Total time span ({total_time_span_days:.2f} days) is less than the minimum required window size ({min_window_days} days)")
    
    # Create time-based windows
    windows = []
    
    # Calculate how many full windows we can create
    possible_windows = int(total_time_span_days / (min_window_days / 2))  # 50% overlap
    max_windows = 10  # Limit the number of windows to avoid excessive computation
    num_windows = min(possible_windows, max_windows)
    num_windows = max(1, num_windows)  # Ensure at least one window
    
    print(f"Creating {num_windows} time-based windows spanning {min_window_days} days each")
    
    # Create windows with proper spacing
    for i in range(num_windows):
        # For evenly spaced windows
        if num_windows > 1:
            # Calculate position within the time span (0 to 1)
            position = i / (num_windows - 1)
            # Calculate start time based on position
            window_start_time = earliest_time + pd.Timedelta(days=(total_time_span_days - min_window_days) * position)
        else:
            # For a single window, start at the beginning
            window_start_time = earliest_time
        
        window_end_time = window_start_time + pd.Timedelta(days=min_window_days)
        
        # Ensure we don't go beyond the latest time
        if window_end_time > latest_time:
            window_end_time = latest_time
            window_start_time = max(earliest_time, window_end_time - pd.Timedelta(days=min_window_days))
        
        # Calculate training, validation, and test boundaries
        train_end_time = window_start_time + pd.Timedelta(days=train_days)
        val_end_time = train_end_time + pd.Timedelta(days=val_days)
        
        # Only create the window if it has enough days
        if window_end_time >= val_end_time + pd.Timedelta(days=test_days/2):  # Allow some flexibility
            windows.append({
                'window_id': len(windows) + 1,
                'start_time': window_start_time,
                'end_time': window_end_time,
                'train_end_time': train_end_time,
                'val_end_time': val_end_time,
                'days': min_window_days
            })
            
            print(f"Window {len(windows)}: {window_start_time.strftime('%Y-%m-%d')} to "
                  f"{window_end_time.strftime('%Y-%m-%d')} ({min_window_days:.0f} days)")
    
    if not windows:
        raise ValueError("Could not create any valid time-based windows.")
    
    print(f"\nCreated {len(windows)} windows total")
    
    # Skip windows we've already processed if resuming
    if start_window > 0:
        windows = windows[start_window:]
        print(f"Skipping {start_window} already processed windows")
    
    # Process each window
    for window_idx, window in enumerate(windows):
        window_id = window['window_id']
        print(f"\n===== Processing Window {window_id}/{len(windows) + start_window} =====")
        
        # Create window output directory
        window_dir = os.path.join(model_output_dir, f"window_{window_id}")
        os.makedirs(window_dir, exist_ok=True)
        
        # Print window details
        print(f"Time window from {window['start_time'].strftime('%Y-%m-%d %H:%M:%S')} "
              f"to {window['end_time'].strftime('%Y-%m-%d %H:%M:%S')} ({window['days']:.1f} days)")
        print(f"Train period ends at: {window['train_end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Validation period ends at: {window['val_end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Select relevant months for this window
        window_months = []
        for info in month_data_info:
            # Check if this month overlaps with the window
            if not (info['max_time'] < window['start_time'] or info['min_time'] > window['end_time']):
                window_months.append(info)
        
        print(f"Window spans data from {len(window_months)} months: "
              f"{[info['month'] for info in window_months]}")
        
        # Initialize data containers
        try:
            # Use optimized data loading to reduce memory pressure
            # Load directly into pandas DataFrames but do the time filtering after loading
            
            print("\nLoading data for time-based filtering...")
            all_train_dfs = []
            all_val_dfs = []
            all_test_dfs = []
            
            # Process each month's data for this window
            for month_info in window_months:
                month = month_info['month']
                parquet_path = month_info['file_path']
                print(f"\nProcessing {os.path.basename(parquet_path)} for window {window_id}...")
                
                # Load the full month data in chunks to avoid memory issues
                parquet_file = pq.ParquetFile(parquet_path)
                for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                    print(f"Processing chunk {chunk_idx + 1}...")
                    # Convert batch to pandas DataFrame
                    chunk_df = batch.to_pandas()
                    
                    # Convert timestamp to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(chunk_df[timestamp_col]):
                        chunk_df[timestamp_col] = pd.to_datetime(chunk_df[timestamp_col])
                    
                    # -------------------------------------------------------------
                    # time–based masks
                    train_mask = ((chunk_df[timestamp_col] >= window['start_time']) &
                                (chunk_df[timestamp_col] <  window['train_end_time']))
                    val_mask   = ((chunk_df[timestamp_col] >= window['train_end_time']) &
                                (chunk_df[timestamp_col] <  window['val_end_time']))
                    test_mask  = ((chunk_df[timestamp_col] >= window['val_end_time']) &
                                (chunk_df[timestamp_col] <= window['end_time']))

                    train_df = chunk_df[train_mask]
                    val_df   = chunk_df[val_mask]
                    test_df  = chunk_df[test_mask]

                    train_count = len(train_df)
                    val_count   = len(val_df)
                    test_count  = len(test_df)

                    # ------- 1 ▸ completely empty chunk?  just move on -------------
                    if train_count == 0 and val_count == 0 and test_count == 0:
                        del chunk_df      # still free memory
                        continue

                    # ------- 2 ▸ diagnostics only if we *have* training rows -------
                    print(f"Chunk {chunk_idx+1}: Train={train_count}, "
                        f"Val={val_count}, Test={test_count} rows")

                    # ------- 3 ▸ append each split only if it has data -------------
                    if train_count > 0:
                        all_train_dfs.append(train_df)
                    if val_count   > 0:
                        all_val_dfs.append(val_df)
                    if test_count  > 0:
                        all_test_dfs.append(test_df)

                    # free memory (leave masks out; they vanish with chunk_df)
                    del chunk_df, train_df, val_df, test_df
                    gc.collect()
            
            # Concatenate data from all months for each split
            print("\nCombining data from all months...")
            
            # Check for missing data in any split
            if not all_train_dfs or not all_val_dfs or not all_test_dfs:
                print(f"Missing data for one or more splits: Train={len(all_train_dfs)}, Val={len(all_val_dfs)}, Test={len(all_test_dfs)}")
                print("Skipping this window due to insufficient data")
                continue
            
            # Combine dataframes
            print("Concatenating training data...")
            train_df = pd.concat(all_train_dfs, ignore_index=True)
            
            print("Concatenating validation data...")
            val_df = pd.concat(all_val_dfs, ignore_index=True)
            
            print("Concatenating test data...")
            test_df = pd.concat(all_test_dfs, ignore_index=True)
            
            # Clean up individual dataframes
            del all_train_dfs, all_val_dfs, all_test_dfs
            gc.collect()
            
            # Detect columns
            all_columns = train_df.columns
            target_cols = [col for col in all_columns if col.startswith('target_horizon_')]
            feature_cols = [col for col in all_columns if col not in target_cols]

            # Split into features and targets
            X_train = train_df.drop(target_cols, axis=1)
            y_train = train_df[target_cols]

            X_val = val_df.drop(target_cols, axis=1)
            y_val = val_df[target_cols]

            X_test = test_df.drop(target_cols, axis=1)
            y_test = test_df[target_cols]

            # Add to train_incremental.py after y_train, y_val splitting
            print("\nDiagnostic Target Statistics:")
            print(f"y_train mean: {y_train.mean().mean()}")
            print(f"y_train min: {y_train.min().min()}, max: {y_train.max().max()}")
            print(f"y_train non-zero values: {(y_train != 0).sum().sum()} out of {y_train.size}")

            # Now apply feature selection based on mode
            print("\nSelecting features based on mode...")

            if ofi_only:
                # Select only the engineered imbalance features
                selected_features = [col for col in X_train.columns
                                    if col.startswith('OFI_')]
                print(f"Using OFI-only mode with {len(selected_features)} features")
            else:
                # Select only raw LOB data (prices and volumes)
                selected_features = [col for col in X_train.columns if (
                    col.startswith('BidPrice_') or 
                    col.startswith('BidVolume_') or 
                    col.startswith('AskPrice_') or 
                    col.startswith('AskVolume_')
                )]
                print(f"Using LOB-only mode with {len(selected_features)} features")

            # Print selected features
            if len(selected_features) < 20:
                print(f"Selected features: {selected_features}")
            else:
                print(f"Selected features (first 10): {selected_features[:10]}...")

            # Filter dataframes to only include selected features
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            X_test = X_test[selected_features]

            # Report final dataset sizes
            print(f"\nFinal dataset sizes:")
            print(f"Training: {len(train_df):,} rows")
            print(f"Validation: {len(val_df):,} rows")
            print(f"Testing: {len(test_df):,} rows")

            # Clean up full dataframes
            del train_df, val_df, test_df
            gc.collect()

            # Normalize features
            print("\nNormalizing features...")

            # Skip columns that shouldn't be normalized
            skip_cols = ['time', 'instrument'] 
            numeric_cols = [col for col in selected_features 
                        if col not in skip_cols and col in X_train.columns and 
                        np.issubdtype(X_train[col].dtype, np.number)]

            # Fit scaler on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[numeric_cols])

            # Create new DataFrames with the scaled values
            X_train_numeric = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)

            # Add non-numeric columns back
            for col in X_train.columns:
                if col not in numeric_cols:
                    X_train_numeric[col] = X_train[col]

            # Transform validation and test sets
            X_val_scaled = scaler.transform(X_val[numeric_cols])
            X_test_scaled = scaler.transform(X_test[numeric_cols])

            # Create new DataFrames
            X_val_numeric = pd.DataFrame(X_val_scaled, columns=numeric_cols, index=X_val.index)
            X_test_numeric = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)

            # Add non-numeric columns back
            for col in X_val.columns:
                if col not in numeric_cols:
                    X_val_numeric[col] = X_val[col]
                    
            for col in X_test.columns:
                if col not in numeric_cols:
                    X_test_numeric[col] = X_test[col]

            # Clean up unscaled data
            del X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled
            gc.collect()

            # Check if we're using ARX model
            # Replace the current ARX block with this more comprehensive version
            if model_type.lower() == "arx":
                print("\nFitting ARX(100,100) models for all prediction horizons...")
                
                # Initialize lists to store results for each horizon
                r2_train_list = []
                r2_val_list = []
                r2_test_list = []
                mse_list = []
                y_test_all = []
                y_pred_all = []
                
                # Create directory for ARX results
                coef_dir = os.path.join(window_dir, "arx_results")
                os.makedirs(coef_dir, exist_ok=True)
                
                # Train a separate ARX model for each target horizon
                for i, target_col in enumerate(target_cols):
                    print(f"\nTraining ARX model for horizon: {target_col}")
                    
                    # Extract Series for this horizon
                    y_train_series = y_train[target_col]
                    y_val_series = y_val[target_col]
                    y_test_series = y_test[target_col]
                    
                    # Fit ARX model
                    arx_results = fit_arx_optimized(
                        y_train=y_train_series,
                        X_ofi_train=X_train_numeric,
                        y_val=y_val_series,
                        X_ofi_val=X_val_numeric,
                        y_test=y_test_series,
                        X_ofi_test=X_test_numeric,
                        ny=args.arx_lags,  # Use the command-line arg
                        nx=args.arx_lags,  # Use the command-line arg
                        max_rows=500000  # Limit sample size
                    )
                    
                    # Store results for this horizon
                    r2_train_list.append(arx_results['r2_train'])
                    r2_val_list.append(arx_results['r2_val'])
                    r2_test_list.append(arx_results['r2_test'])
                    
                    # Calculate and store MSE
                    mse = np.mean((arx_results["y_test"] - arx_results["y_pred"])**2)
                    mse_list.append(mse)
                    
                    # Store predictions for this horizon
                    y_test_all.append(arx_results["y_test"])
                    y_pred_all.append(arx_results["y_pred"])
                    
                    # Save the coefficients for this horizon
                    feature_names = arx_results.get("model").__dict__.get("feature_names_in_", None)
                    if feature_names is None:
                        feature_names = [f"feature_{i}" for i in range(len(arx_results["model"].coef_))]
                    
                    coef_df = pd.DataFrame({
                        'feature': feature_names,
                        'coefficient': arx_results["model"].coef_
                    })
                    coef_df.to_csv(os.path.join(coef_dir, f"arx_coefs_horizon_{i+1}.csv"), index=False)
                    
                    # Save predictions for this horizon
                    np.save(os.path.join(coef_dir, f"arx_y_test_horizon_{i+1}.npy"), arx_results["y_test"])
                    np.save(os.path.join(coef_dir, f"arx_y_pred_horizon_{i+1}.npy"), arx_results["y_pred"])
                    
                    # Log results for this horizon
                    logger.info(f"ARX Horizon {i+1} - Train R²: {arx_results['r2_train']:.4f}, "
                            f"Val R²: {arx_results['r2_val']:.4f}, "
                            f"Test R²: {arx_results['r2_test']:.4f}")
                
                # Calculate sign accuracy across all horizons
                print("Calculating sign accuracy...")
                sign_acc_values = []

                for i in range(len(target_cols)):
                    y_true = y_test_all[i]
                    y_pred = y_pred_all[i]
                    
                    # Calculate sign accuracy for this horizon
                    correct_signs = np.sign(y_pred) == np.sign(y_true)
                    non_zero_mask = y_true != 0
                    
                    if np.sum(non_zero_mask) > 0:
                        horizon_sign_acc = np.mean(correct_signs[non_zero_mask])
                    else:
                        horizon_sign_acc = np.nan
                    
                    sign_acc_values.append(horizon_sign_acc)
                    logger.info(f"ARX Horizon {i+1} - Sign Accuracy: {horizon_sign_acc:.4f}")

                # Calculate average sign accuracy across all horizons
                sign_accuracy = np.nanmean(sign_acc_values)
                logger.info(f"ARX Average Sign Accuracy: {sign_accuracy:.4f}")
                
                # Generate a plot similar to the deep learning models
                try:
                    plt.figure(figsize=(10, 6))
                    horizons = [float(h.replace('target_horizon_', '')) for h in target_cols]
                    plt.plot(horizons, r2_test_list, marker='o', label='ARX')
                    plt.xlabel('Prediction Horizon')
                    plt.ylabel('R²OOS (%)')
                    plt.title(f'ARX Model R² by Horizon - {symbol}')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(coef_dir, "arx_r2_by_horizon.png"))
                    plt.close()
                except Exception as e:
                    print(f"Error creating ARX R² plot: {e}")
                
                # Store window details for results
                window_details = {
                    'start_time': window['start_time'].strftime('%Y-%m-%d'),
                    'end_time': window['end_time'].strftime('%Y-%m-%d'),
                    'train_end_time': window['train_end_time'].strftime('%Y-%m-%d'),
                    'val_end_time': window['val_end_time'].strftime('%Y-%m-%d'),
                    'days': window['days'],
                    'feature_mode': "ofi_only" if ofi_only else "lob_only"
                }
                
                # Save R2 values in the format expected by all_results
                all_results[f"window_{window_id}"] = {
                    'window_details': window_details,
                    'train_size': len(X_train_numeric),
                    'val_size': len(X_val_numeric),
                    'test_size': len(X_test_numeric),
                    'r2_oos': r2_test_list,  # List of R² values for all horizons
                    'mse': mse_list,
                    'sign_accuracy': sign_accuracy
                }
                
                # Save the full results separately for easier plotting
                arx_full_results = {
                    'horizons': horizons,
                    'r2_train': r2_train_list,
                    'r2_val': r2_val_list, 
                    'r2_oos': r2_test_list,
                    'mse': mse_list
                }
                
                with open(os.path.join(coef_dir, "arx_full_results.json"), 'w') as f:
                    json.dump(convert_numpy_to_python(arx_full_results), f, indent=4)
                
                # Update checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'last_completed_window': window_id - 1,
                        'total_windows': len(windows) + start_window,
                        'model_type': model_type,
                        'symbol': symbol,
                        'year': year,
                        'months': months,
                        'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=4)
                
                # Skip the deep learning training
                print("\nARX models fitting and evaluation complete for all horizons!")
                continue  # Skip to the next window

            # Initialize the model
            print("\nInitializing model...")
            model_kwargs_full = {
                'input_dim': len(numeric_cols),
                'num_horizons': len(target_cols),
            }
            if model_type.lower() in ['cnn', 'cnn_lstm']:
                model_kwargs_full['seq_length'] = seq_length

            # Add the rest of the model parameters
            model_kwargs_full.update(model_kwargs)

            model = get_model(model_type, **model_kwargs_full).to(device)

            # Create datasets
            print("Creating datasets...")
            # Reduced preload threshold to save memory
            preload = len(X_train_numeric) < 50000  # Only preload for smaller datasets

            # Select only numeric columns for the datasets
            X_train_model = X_train_numeric.select_dtypes(include=['number'])
            X_val_model = X_val_numeric.select_dtypes(include=['number'])
            X_test_model = X_test_numeric.select_dtypes(include=['number'])

            train_dataset = OptimizedLOBDataset(X_train_model, y_train, seq_length=seq_length, preload=preload)
            val_dataset = OptimizedLOBDataset(X_val_model, y_val, seq_length=seq_length, preload=preload)
            test_dataset = OptimizedLOBDataset(X_test_model, y_test, seq_length=seq_length, preload=preload)
            
            # Create data loaders with reduced num_workers
            print("Creating data loaders...")
            # Disable multiprocessing by setting num_workers=0 to avoid Windows-specific pickling issues
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,  # Disable multiprocessing to avoid Windows pickling issues
                pin_memory=device.type == 'cuda'
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,  # Same as training batch size to be conservative
                shuffle=False,
                num_workers=0,  # Disable multiprocessing to avoid Windows pickling issues
                pin_memory=device.type == 'cuda'
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size,  # Same as training batch size to be conservative
                shuffle=False,
                num_workers=0,  # Disable multiprocessing to avoid Windows pickling issues
                pin_memory=device.type == 'cuda'
            )
            
            # Initialize optimizer and criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            
            # Create scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            
            # Log learning rate changes manually instead of using verbose=True
            def lr_lambda(epoch):
                if epoch > 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Learning rate: {current_lr:.8f}")
                return 1.0  # No modification to lr
            
            # Create early stopping
            early_stopping = EarlyStopping(
                patience=model_params['training']['early_stopping']['patience'],
                mode='min',
                verbose=True,
                save_path=os.path.join(window_dir, "best_model.pt")
            )
            
            # Use modified training function
            print(f"\nTraining model for window {window_id} with resource limits...")
            history = modified_train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_dir=window_dir,
                verbose=True,
                use_amp=True,
                grad_accumulation=grad_accumulation,
                sleep_time=sleep_time  # Small sleep between batches
            )
            
            # Save training history plot
            print("Generating training history plot...")
            plot_training_history(
                history,
                save_path=os.path.join(window_dir, "training_history.png")
            )
            
            # Evaluate on test set
            print("Evaluating model on test set...")
            all_train_targets = []
            with torch.no_grad():
                for _, t in train_loader:
                    all_train_targets.append(t.cpu().numpy())
            train_targets = np.concatenate(all_train_targets, axis=0)  # shape (N_train, num_horizons)
            benchmark_means = train_targets.mean(axis=0)

            model.load_state_dict(torch.load(os.path.join(window_dir, "best_model.pt"), weights_only=True))
            eval_results = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                output_dir=window_dir,
                model_name=f"{model_type}_window_{window_id}",
                benchmark_means=benchmark_means,    # ← now uses train‐set means
            )
            
            # Save R2 by horizon plot
            print("Generating R2 by horizon plot...")
            plot_r2_by_horizon(
                eval_results['r2_oos'],
                save_path=os.path.join(window_dir, "r2_by_horizon.png")
            )
            
            # Store window details for results
            window_details = {
                'start_time': window['start_time'].strftime('%Y-%m-%d'),
                'end_time': window['end_time'].strftime('%Y-%m-%d'),
                'train_end_time': window['train_end_time'].strftime('%Y-%m-%d'),
                'val_end_time': window['val_end_time'].strftime('%Y-%m-%d'),
                'days': window['days'],
                'feature_mode': "ofi_only" if ofi_only else "lob_only"
            }
            
            # Store results
            all_results[f"window_{window_id}"] = {
                'window_details': window_details,
                'train_size': len(X_train_model),
                'val_size': len(X_val_model),
                'test_size': len(X_test_model),
                'r2_oos': eval_results['r2_oos'],
                'mse': eval_results['mse'].tolist(),
                'sign_accuracy': eval_results['sign_accuracy']
            }
            
            # Save training history data for later plotting
            months_range = f"{min(months):02d}-{max(months):02d}" if len(months) > 1 else f"{months[0]:02d}"
            plot_data_dir = os.path.join(
                results_dir, 
                'plot_data', 
                model_type, 
                f"{symbol}_{year}-{months_range}", 
                f"window_{window_id}"
            )
            save_plot_data(history, plot_data_dir, "training_history")
            save_plot_data(eval_results['r2_oos'], plot_data_dir, "r2_values", format='npy')
            # save_plot_data(eval_results, plot_data_dir, "full_eval_results") # Saves a ~3GB file
            
            # Update checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'last_completed_window': window_id - 1,  # Convert back to 0-based index
                    'total_windows': len(windows) + start_window,
                    'model_type': model_type,
                    'symbol': symbol,
                    'year': year,
                    'months': months,
                    'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=4)
            
            # Thoroughly clean up after each window
            del X_train_numeric, X_val_numeric, X_test_numeric, y_train, y_val, y_test
            del X_train_model, X_val_model, X_test_model
            del train_dataset, val_dataset, test_dataset
            del train_loader, val_loader, test_loader
            
            # Extra cleanup for GPU memory
            model = model.cpu()  # Move model to CPU before deleting
            del model, optimizer, criterion, scheduler, early_stopping
            
            # Force aggressive garbage collection
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"Completed window {window_id}/{len(windows) + start_window}")
            if 'r2_oos' in eval_results:
                print(f"Avg R2 OOS: {np.mean(eval_results['r2_oos']):.4f}")
                
            # Optional pause between windows to let system cool down
            import time
            print("Pausing for 5 seconds to let system cool down...")
            time.sleep(5)
                
        except Exception as e:
            print(f"Error processing window {window_id}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next window
            continue
    
    # Save all results together
    try:
        # Convert NumPy arrays to Python native types
        converted_results = convert_numpy_to_python(all_results)
        with open(os.path.join(model_output_dir, "all_results.json"), 'w') as f:
            json.dump(converted_results, f, indent=4)
        print(f"\nResults successfully saved to {os.path.join(model_output_dir, 'all_results.json')}")
    except Exception as e:
        print(f"Error saving final results: {e}")
    
    print("\nTraining completed for all windows!")
    print(f"Results saved to {model_output_dir}")
    
    return True

def run_multiple_symbols_and_modes(args):
    """
    Run training for multiple symbols and feature modes automatically.
    
    This function will iterate through:
    - BTCUSDT and ETHUSDT
    - Both OFI-only and LOB-only feature modes
    
    Parameters:
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    """
    symbols = ['BTCUSDT', 'ETHUSDT']
    feature_modes = [True, False]  # True for OFI-only, False for LOB-only
    
    print("\n" + "="*100)
    print("AUTOMATED MULTI-SYMBOL AND MULTI-MODE TRAINING")
    print(f"Symbols: {symbols}")
    print(f"Feature modes: OFI-only and LOB-only")
    print(f"Year: {args.year}, Months: {args.months}, Model: {args.model}")
    print("="*100 + "\n")
    
    results_summary = {}
    total_runs = len(symbols) * len(feature_modes)
    current_run = 0
    
    for symbol in symbols:
        for ofi_only in feature_modes:
            current_run += 1
            feature_mode_name = "OFI-only" if ofi_only else "LOB-only"
            
            print("\n" + "="*80)
            print(f"RUN {current_run}/{total_runs}: {symbol} - {feature_mode_name}")
            print("="*80)
            
            try:
                # Set the symbol and ofi_only flag for this run
                current_symbol = symbol
                current_ofi_only = ofi_only
                
                # Call the training function
                result = train_symbol_months_time_based_limited(
                    symbol=current_symbol,
                    year=args.year,
                    months=args.months,
                    model_type=args.model,
                    config_path=args.model_config,
                    data_config_path=args.data_config,
                    resume=args.resume,
                    chunk_size=args.chunk_size,
                    min_window_days=args.min_window_days,
                    gpu_memory_limit=args.gpu_memory_limit,
                    gpu_util_target=args.gpu_memory_limit,  # Using same value
                    batch_size_factor=args.batch_size_factor,
                    grad_accumulation=args.grad_accumulation,
                    sleep_time=args.sleep_time,
                    ofi_only=current_ofi_only
                )
                
                # Store result
                run_key = f"{current_symbol}_{feature_mode_name.lower().replace('-', '_')}"
                results_summary[run_key] = {
                    'status': 'SUCCESS' if result else 'FAILED',
                    'symbol': current_symbol,
                    'feature_mode': feature_mode_name,
                    'completed': True
                }
                
                print(f"\n✅ COMPLETED: {current_symbol} - {feature_mode_name}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ ERROR in {current_symbol} - {feature_mode_name}: {error_msg}")
                
                # Store error result
                run_key = f"{current_symbol}_{feature_mode_name.lower().replace('-', '_')}"
                results_summary[run_key] = {
                    'status': 'ERROR',
                    'symbol': current_symbol,
                    'feature_mode': feature_mode_name,
                    'error': error_msg,
                    'completed': False
                }
                
                # Print traceback for debugging but continue with next run
                import traceback
                traceback.print_exc()
                
            finally:
                # Force cleanup between runs
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Pause between runs
                if current_run < total_runs:
                    print(f"\nPausing for 10 seconds before next run...")
                    import time
                    time.sleep(10)
    
    # Print final summary
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    
    successful_runs = 0
    failed_runs = 0
    error_runs = 0
    
    for run_key, result in results_summary.items():
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌" if result['status'] == 'ERROR' else "⚠️"
        print(f"{status_emoji} {result['symbol']} - {result['feature_mode']}: {result['status']}")
        
        if result['status'] == 'SUCCESS':
            successful_runs += 1
        elif result['status'] == 'ERROR':
            error_runs += 1
        else:
            failed_runs += 1
    
    print("-" * 100)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Errors: {error_runs}")
    print("="*100 + "\n")
    
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on specific symbol across multiple months using time-based splitting with resource limits')
    parser.add_argument('--symbol', type=str, required=True, choices=['ADAUSDT', 'BTCUSDT', 'ETHUSDT'], 
                        help='Symbol to train on')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2024)')
    parser.add_argument('--months', type=int, nargs='+', required=True, help='Months to include (e.g., 1 2 3 for Jan-Mar)')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'lstm', 'cnn_lstm', 'arx'], help='Model type')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml', help='Path to model config')
    parser.add_argument('--data-config', type=str, default='config/data_config.yaml', help='Path to data config')
    parser.add_argument('--chunk-size', type=int, default=1000000, help='Size of batches to process at a time')
    parser.add_argument('--min-window-days', type=int, default=35, 
                      help='Minimum number of days required per window (default: 35)')
    parser.add_argument('--gpu-memory-limit', type=float, default=0.8,
                      help='Percentage of GPU memory to use (0.0 to 1.0)')
    parser.add_argument('--batch-size-factor', type=float, default=0.75,
                      help='Factor to reduce batch size by')
    parser.add_argument('--grad-accumulation', type=int, default=2,
                      help='Number of batches to accumulate gradients')
    parser.add_argument('--sleep-time', type=float, default=0.005,
                      help='Time to sleep between batches (seconds)')
    parser.add_argument('--ofi-only', action='store_true', 
                      help='Use only OFI features instead of raw LOB data')
    parser.add_argument('--arx-lags', type=int, default=100, 
                    help='Number of lags to use for ARX model (default: 100)')
    parser.add_argument('--max-window-days', type=int, default=14,
                    help='Maximum window size in days for backtesting periods (default: 14)')
    
    # NEW: Add argument for automated multi-symbol training
    parser.add_argument('--auto-multi', action='store_true',
                      help='Automatically run for BTCUSDT and ETHUSDT with both OFI-only and LOB-only modes')
    
    args = parser.parse_args()
    
    # Check if we should run automated multi-symbol training
    if args.auto_multi:
        print("Running automated multi-symbol and multi-mode training...")
        results = run_multiple_symbols_and_modes(args)
    else:
        # Original single-run logic
        print("\n" + "="*80)
        print(f"Resource-Limited Multi-Month Training - Symbol: {args.symbol}, Year: {args.year}, Months: {args.months}")
        print(f"Model: {args.model}, Feature mode: {'OFI-only' if args.ofi_only else 'LOB-only'}")
        print(f"GPU Memory Limit: {args.gpu_memory_limit*100:.0f}%, Batch Size Factor: {args.batch_size_factor}")
        print(f"Gradient Accumulation: {args.grad_accumulation}, Sleep Time: {args.sleep_time}s")
        print("="*80 + "\n")
        
        try:
            result = train_symbol_months_time_based_limited(
                args.symbol, 
                args.year, 
                args.months, 
                args.model, 
                args.model_config, 
                args.data_config, 
                args.resume,
                args.chunk_size,
                args.min_window_days,
                args.gpu_memory_limit,
                args.gpu_memory_limit,  # Using same value for utilization target
                args.batch_size_factor,
                args.grad_accumulation,
                args.sleep_time,
                args.ofi_only
            )
            
            print("="*80)
            print(f"Training completed with status: {'Success' if result else 'Failed'}")
            print("="*80 + "\n")
        except Exception as e:
            print("="*80)
            print(f"ERROR: {str(e)}")
            print("Training failed.")
            print("="*80 + "\n")
            import traceback
            traceback.print_exc()

        # python -m src.train_incremental --symbol BTCUSDT --year 2024 --months 1 2 3 --model cnn --gpu-memory-limit 0.9 --batch-size-factor 1.0 --grad-accumulation 2 --ofi-only
        # python -m src.train_incremental --symbol BTCUSDT --year 2024 --months 1 2 3 --model cnn --gpu-memory-limit 0.9 --batch-size-factor 1.0 --grad-accumulation 2

        # python -m src.train_incremental --symbol BTCUSDT --year 2024 --months 1 2 3 --model arx --ofi-only
        # python -m src.train_incremental --symbol BTCUSDT --year 2024 --months 1 2 3 --model arx