"""
Evaluation module for cryptocurrency high-frequency trading models.

This module implements evaluation metrics and procedures following:
Kolm et al. (2023) - "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book"

The main metrics include:
1. R^2 out-of-sample
2. Mean squared error (MSE)
3. Model comparison statistics

Author: Noah Trägårdh
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union, Optional
import logging
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import gc

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Optional[str] = None,
    model_name: str = "model",
    use_amp: bool = True,
    generate_plots: bool = True,
    save_predictions: bool = True,  # Set to False by default to save memory
    benchmark_means: Optional[np.ndarray] = None  # Added parameter for benchmark means
) -> Dict[str, Union[np.ndarray, List[float]]]:
    """
    Memory-efficient evaluation of a trained model.
    
    Parameters:
    ----------
    model : nn.Module
        The trained neural network model
    data_loader : DataLoader
        DataLoader containing the test data
    device : torch.device
        Device to run evaluation on
    output_dir : Optional[str]
        Directory to save evaluation results
    model_name : str
        Name of the model for saving results
    use_amp : bool
        Whether to use automatic mixed precision
    generate_plots : bool
        Whether to generate evaluation plots
    save_predictions : bool
        Whether to save predictions to disk
    benchmark_means : Optional[np.ndarray]
        Average returns from training data to use as benchmark.
        If None, the mean of the test data will be used.
        
    Returns:
    -------
    Dict[str, Union[np.ndarray, List[float]]]
        Dictionary containing evaluation metrics
    """
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Pre-allocate arrays
    all_predictions = []
    all_targets = []
    
    logger.info("Evaluating model...")
    
    # Clean GPU memory before starting
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc=f"Evaluating {model_name}"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Use mixed precision for inference
                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Immediately move to CPU and detach to free GPU memory
                all_predictions.append(outputs.cpu().detach())
                all_targets.append(targets.cpu().detach())
                
                # Free memory immediately
                del outputs, inputs, targets
                
                # Periodically clear CUDA cache if needed
                if len(all_predictions) % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate in smaller chunks if needed
        logger.info("Processing evaluation results...")
        
        # Process predictions and calculate metrics incrementally to save memory
        num_batches = len(all_predictions)
        batch_size = all_predictions[0].shape[0]
        num_horizons = all_predictions[0].shape[1]
        
        # If benchmark_means not provided, calculate means from targets
        if benchmark_means is None:
            logger.info("Calculating benchmark means from test data...")
            benchmark_means = []
            # Calculate means from all targets
            for h in range(num_horizons):
                h_targets = [batch[:, h].numpy() for batch in all_targets]
                h_targets_flat = np.concatenate(h_targets)
                benchmark_means.append(np.mean(h_targets_flat))
            benchmark_means = np.array(benchmark_means)
        
        # Initialize metric arrays
        mse_sums = np.zeros(num_horizons)
        mae_sums = np.zeros(num_horizons)
        r2_numerators = np.zeros(num_horizons)
        r2_denominators = np.zeros(num_horizons)
        benchmark_mse_sums = np.zeros(num_horizons)  # Changed from naive_mse_sums
        correct_signs = np.zeros(num_horizons)
        total_non_zero = np.zeros(num_horizons)
        total_samples = 0
        
        # If we need to save predictions, prepare arrays
        predictions = None
        targets = None
        if save_predictions:
            try:
                # Estimate memory needed
                total_rows = len(data_loader.dataset) - data_loader.dataset.seq_length + 1
                mem_needed_gb = total_rows * num_horizons * 2 * 4 / (1024**3)  # 4 bytes per float32
                logger.info(f"Allocating ~{mem_needed_gb:.2f} GB for predictions and targets...")
                
                predictions = torch.cat(all_predictions, dim=0).numpy()
                targets = torch.cat(all_targets, dim=0).numpy()
            except Exception as e:
                logger.warning(f"Could not allocate memory for full predictions: {e}")
                save_predictions = False
        
        # Process each batch individually for metrics if not saving predictions
        if not save_predictions:
            for pred_batch, target_batch in zip(all_predictions, all_targets):
                pred_np = pred_batch.numpy()
                target_np = target_batch.numpy()
                batch_size = pred_np.shape[0]
                
                # Update running sums for each metric
                for h in range(num_horizons):
                    # MSE calculation
                    batch_mse = np.mean((pred_np[:, h] - target_np[:, h]) ** 2)
                    mse_sums[h] += batch_mse * batch_size
                    
                    # MAE calculation
                    batch_mae = np.mean(np.abs(pred_np[:, h] - target_np[:, h]))
                    mae_sums[h] += batch_mae * batch_size
                    
                    # For R2 calculation 
                    batch_r2_num = np.sum((pred_np[:, h] - target_np[:, h]) ** 2)
                    batch_r2_denom = np.sum((target_np[:, h] - np.mean(target_np[:, h])) ** 2)
                    r2_numerators[h] += batch_r2_num
                    r2_denominators[h] += batch_r2_denom
                    
                    # For benchmark MSE (R2 OOS)
                    # Use the benchmark mean for this horizon
                    avg_return = benchmark_means[h]
                    batch_benchmark_mse = np.mean((target_np[:, h] - avg_return) ** 2)
                    benchmark_mse_sums[h] += batch_benchmark_mse * batch_size
                    
                    # Sign accuracy
                    batch_correct = np.sum((np.sign(pred_np[:, h]) == np.sign(target_np[:, h])) 
                                         & (target_np[:, h] != 0))
                    batch_non_zero = np.sum(target_np[:, h] != 0)
                    correct_signs[h] += batch_correct
                    total_non_zero[h] += batch_non_zero
                
                total_samples += batch_size
                
                # Clean up memory
                del pred_np, target_np
            
            # Free up memory after processing
            del all_predictions, all_targets
            
            # Calculate final metrics
            mse_values = mse_sums / total_samples
            mae_values = mae_sums / total_samples
            
            r2_values = []
            r2_oos_values = []
            sign_accuracy_values = []
            
            for h in range(num_horizons):
                # R^2
                if r2_denominators[h] > 0:
                    r2 = 1 - (r2_numerators[h] / r2_denominators[h])
                else:
                    r2 = 0.0
                r2_values.append(r2)
                
                # R^2 out-of-sample according to equation (18) in the paper
                benchmark_mse = benchmark_mse_sums[h] / total_samples
                if benchmark_mse > 0:
                    r2_oos = 1 - (mse_values[h] / benchmark_mse)
                else:
                    r2_oos = 0.0
                r2_oos_values.append(r2_oos)
                
                # Sign accuracy
                if total_non_zero[h] > 0:
                    sign_acc = correct_signs[h] / total_non_zero[h]
                else:
                    sign_acc = np.nan
                sign_accuracy_values.append(sign_acc)
        else:
            # If we have the full arrays, calculate metrics directly
            mse_values = np.array([mean_squared_error(targets[:, h], predictions[:, h]) 
                                  for h in range(num_horizons)])
            mae_values = np.array([np.mean(np.abs(targets[:, h] - predictions[:, h])) 
                                  for h in range(num_horizons)])
            
            r2_values = []
            r2_oos_values = []
            sign_accuracy_values = []
            
            for h in range(num_horizons):
                # R^2
                r2 = r2_score(targets[:, h], predictions[:, h])
                r2_values.append(r2)
                
                # R^2 out-of-sample according to equation (18) in the paper
                avg_return = benchmark_means[h]
                mse_benchmark = mean_squared_error(targets[:, h], np.full_like(targets[:, h], avg_return))
                r2_oos = 1 - mse_values[h] / mse_benchmark
                r2_oos_values.append(r2_oos)
                
                # Sign accuracy
                correct_signs = np.sign(predictions[:, h]) == np.sign(targets[:, h])
                non_zero_mask = targets[:, h] != 0
                if np.sum(non_zero_mask) > 0:
                    acc = np.mean(correct_signs[non_zero_mask])
                else:
                    acc = np.nan
                sign_accuracy_values.append(acc)
        
        # Calculate correlation only if predictions are available
        if save_predictions and predictions is not None:
            pred_corr = np.corrcoef(predictions.T)
        else:
            # Create a dummy correlation matrix
            pred_corr = np.eye(num_horizons)
        
        # Convert to numpy arrays
        mse_values = np.array(mse_values)
        mae_values = np.array(mae_values)
        r2_values = np.array(r2_values)
        r2_oos_values = np.array(r2_oos_values)
        sign_accuracy_values = np.array(sign_accuracy_values)
        
        # Combine results (excluding the large arrays unless needed)
        results = {
            'mse': mse_values,
            'mae': mae_values,
            'r2': r2_values,
            'r2_oos': r2_oos_values,
            'pred_corr': pred_corr,
            'sign_accuracy': sign_accuracy_values,
            'benchmark_means': benchmark_means  # Add benchmark means to results
        }
        
        # Only add predictions and targets if explicitly requested
        if save_predictions and predictions is not None and targets is not None:
            results['predictions'] = predictions
            results['targets'] = targets
        
        # Save results if output directory is specified
        if output_dir is not None:
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'horizon': range(1, num_horizons + 1),
                'mse': mse_values,
                'mae': mae_values,
                'r2': r2_values,
                'r2_oos': r2_oos_values,
                'sign_accuracy': sign_accuracy_values,
                'benchmark_mean': benchmark_means
            })
            
            metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)
            
            # Save predictions and targets only if explicitly requested
            if save_predictions and predictions is not None and targets is not None:
                try:
                    logger.info(f"Saving predictions and targets to disk...")
                    np.savez_compressed(os.path.join(output_dir, f"{model_name}_predictions.npz"), 
                                      predictions=predictions, targets=targets)
                except Exception as e:
                    logger.warning(f"Error saving predictions: {e}")
            
            # Generate and save plots if requested
            if generate_plots:
                logger.info("Generating evaluation plots...")
                
                try:
                    # Generate plots directly to avoid multiprocessing issues
                    # R2 plot
                    plt.figure(figsize=(12, 6))
                    plt.bar(range(1, num_horizons + 1), r2_oos_values, color='royalblue')
                    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
                    plt.xlabel('Horizon')
                    plt.ylabel('R² Out-of-Sample')
                    plt.title(f'{model_name}: R² Out-of-Sample by Horizon')
                    plt.grid(alpha=0.3)
                    plt.savefig(os.path.join(output_dir, "r2_by_horizon.png"), dpi=100)
                    plt.close()
                    
                    # Sign accuracy plot
                    plt.figure(figsize=(12, 6))
                    plt.bar(range(1, num_horizons + 1), sign_accuracy_values, color='green')
                    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.7, label='Random Guess')
                    plt.xlabel('Horizon')
                    plt.ylabel('Sign Accuracy')
                    plt.title(f'{model_name}: Directional Accuracy by Horizon')
                    plt.grid(alpha=0.3)
                    plt.ylim(0.4, 0.8)
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f"{model_name}_sign_accuracy.png"), dpi=100)
                    plt.close()
                    
                    # Correlation plot
                    if save_predictions and predictions is not None:
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            pred_corr,
                            annot=True,
                            cmap='coolwarm',
                            vmin=-1,
                            vmax=1,
                            center=0,
                            square=True,
                            fmt='.2f',
                            cbar_kws={'label': 'Correlation'}
                        )
                        plt.title(f'{model_name}: Correlation between Horizon Predictions')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"{model_name}_pred_correlation.png"), dpi=100)
                        plt.close()
                except Exception as e:
                    logger.warning(f"Error generating plots: {e}")
        
        logger.info("Model evaluation completed")
        
        # Clean up memory before returning
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    except Exception as e:
        # Ensure memory is cleaned up even on error
        logger.error(f"Error during evaluation: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        raise
    finally:
        # Always clean up big objects
        if 'all_predictions' in locals() and all_predictions:
            del all_predictions
        if 'all_targets' in locals() and all_targets:
            del all_targets
        if 'predictions' in locals() and predictions is not None:
            del predictions
        if 'targets' in locals() and targets is not None:
            del targets
        gc.collect()

def compare_models(
    model_results: Dict[str, Dict[str, Union[np.ndarray, List[float]]]],
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models based on their evaluation metrics.
    
    Parameters:
    ----------
    model_results : Dict[str, Dict[str, Union[np.ndarray, List[float]]]]
        Dictionary mapping model names to their evaluation results
    output_dir : str, optional
        Directory to save comparison results and plots
        
    Returns:
    -------
    pd.DataFrame
        DataFrame containing model comparison metrics
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize comparison DataFrame
    comparison_data = []
    
    # Get number of horizons (assuming all models have the same number)
    num_horizons = len(next(iter(model_results.values()))['r2_oos'])
    
    # Gather metrics for each model and horizon
    for model_name, results in model_results.items():
        for h in range(num_horizons):
            comparison_data.append({
                'model': model_name,
                'horizon': h + 1,
                'mse': results['mse'][h],
                'mae': results['mae'][h],
                'r2': results['r2'][h],
                'r2_oos': results['r2_oos'][h],
                'sign_accuracy': results['sign_accuracy'][h]
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV if output directory is specified
    if output_dir is not None:
        comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        
        # Generate comparison plots
        logger.info("Generating model comparison plots...")
        
        # 1. R^2 OOS comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=comparison_df,
            x='horizon',
            y='r2_oos',
            hue='model',
            palette='viridis'
        )
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
        plt.xlabel('Horizon')
        plt.ylabel('R² Out-of-Sample')
        plt.title('Model Comparison: R² Out-of-Sample by Horizon')
        plt.grid(alpha=0.3)
        plt.legend(title='Model')
        plt.savefig(os.path.join(output_dir, "model_comparison_r2_oos.png"))
        plt.close()
        
        # 2. Sign accuracy comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=comparison_df,
            x='horizon',
            y='sign_accuracy',
            hue='model',
            palette='viridis'
        )
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.7, label='Random Guess')
        plt.xlabel('Horizon')
        plt.ylabel('Sign Accuracy')
        plt.title('Model Comparison: Directional Accuracy by Horizon')
        plt.grid(alpha=0.3)
        plt.legend(title='Model')
        plt.savefig(os.path.join(output_dir, "model_comparison_sign_accuracy.png"))
        plt.close()
        
        # 3. MSE comparison
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=comparison_df,
            x='horizon',
            y='mse',
            hue='model',
            palette='viridis'
        )
        plt.xlabel('Horizon')
        plt.ylabel('Mean Squared Error')
        plt.title('Model Comparison: MSE by Horizon')
        plt.grid(alpha=0.3)
        plt.legend(title='Model')
        plt.savefig(os.path.join(output_dir, "model_comparison_mse.png"))
        plt.close()
        
        # 4. Model performance heatmap
        # Pivot the data for heatmap
        r2_oos_pivot = comparison_df.pivot(index='model', columns='horizon', values='r2_oos')
        
        plt.figure(figsize=(12, len(model_results) * 1.5))
        sns.heatmap(
            r2_oos_pivot,
            annot=True,
            cmap='RdYlGn',
            center=0,
            fmt='.3f',
            cbar_kws={'label': 'R² Out-of-Sample'}
        )
        plt.title('Model Comparison: R² Out-of-Sample Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison_heatmap.png"))
        plt.close()
    
    logger.info("Model comparison completed")
    return comparison_df


def evaluate_rolling_window_models(
    models: List[nn.Module],
    X: pd.DataFrame,
    y: pd.DataFrame,
    window_size: int,
    step_size: int,
    seq_length: int,
    batch_size: int,
    device: torch.device,
    output_dir: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Evaluate models trained with rolling window approach.
    
    Parameters:
    ----------
    models : List[nn.Module]
        List of trained models for each window
    X : pd.DataFrame
        Feature DataFrame
    y : pd.DataFrame
        Target DataFrame
    window_size : int
        Size of each rolling window
    step_size : int
        Step size between windows
    seq_length : int
        Sequence length for model input
    batch_size : int
        Batch size for evaluation
    device : torch.device
        Device to run evaluation on
    output_dir : str, optional
        Directory to save evaluation results
        
    Returns:
    -------
    Dict[str, List[float]]
        Dictionary containing evaluation metrics for each window
    """
    # Import needed modules here to avoid circular imports
    from src.model_architectures import LOBDataset
    
    num_samples = len(X)
    num_windows = len(models)
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'window': list(range(1, num_windows + 1)),
        'start_idx': [],
        'end_idx': [],
        'r2_oos': [],
        'mse': [],
        'sign_accuracy': []
    }
    
    # Evaluate each model on its respective out-of-sample window
    for window, model in enumerate(models):
        window_start = window * step_size
        window_end = window_start + window_size
        
        if window_end >= num_samples:
            # Skip windows that don't have out-of-sample data
            logger.info(f"Window {window+1}/{num_windows}: No out-of-sample data available")
            continue
        
        # Define test period (next step_size after window)
        test_start = window_end
        test_end = min(test_start + step_size, num_samples)
        
        logger.info(f"Window {window+1}/{num_windows}: Evaluating on indices {test_start} to {test_end-1}")
        
        # Store window indices
        results['start_idx'].append(test_start)
        results['end_idx'].append(test_end)
        
        # Get test data
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        # Create test dataset and loader
        test_dataset = LOBDataset(X_test, y_test, seq_length=seq_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate model
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.vstack(all_predictions) if all_predictions else np.array([])
        targets = np.vstack(all_targets) if all_targets else np.array([])
        
        if len(predictions) == 0 or len(targets) == 0:
            # No valid predictions for this window
            results['r2_oos'].append(np.nan)
            results['mse'].append(np.nan)
            results['sign_accuracy'].append(np.nan)
            continue
        
        # Calculate average metrics across all horizons
        r2_oos_values = []
        mse_values = []
        sign_acc_values = []
        
        for h in range(predictions.shape[1]):
            # R^2 out-of-sample
            mse_model = mean_squared_error(targets[:, h], predictions[:, h])
            mse_naive = mean_squared_error(targets[:, h], np.zeros_like(targets[:, h]))
            r2_oos = 1 - mse_model / mse_naive
            r2_oos_values.append(r2_oos)
            
            # MSE
            mse_values.append(mse_model)
            
            # Sign accuracy
            correct_signs = np.sign(predictions[:, h]) == np.sign(targets[:, h])
            non_zero_mask = targets[:, h] != 0
            if np.sum(non_zero_mask) > 0:
                acc = np.mean(correct_signs[non_zero_mask])
            else:
                acc = np.nan
            sign_acc_values.append(acc)
        
        # Store average metrics
        results['r2_oos'].append(np.nanmean(r2_oos_values))
        results['mse'].append(np.nanmean(mse_values))
        results['sign_accuracy'].append(np.nanmean(sign_acc_values))
        
        logger.info(f"  Avg R² OOS: {results['r2_oos'][-1]:.4f}")
        logger.info(f"  Avg MSE: {results['mse'][-1]:.6f}")
        logger.info(f"  Avg Sign Accuracy: {results['sign_accuracy'][-1]:.4f}")
    
    # Save results to CSV if output directory is specified
    if output_dir is not None:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, "rolling_window_results.csv"), index=False)
        
        # Generate time series plot of R^2 OOS
        plt.figure(figsize=(14, 7))
        plt.plot(results['window'], results['r2_oos'], 'o-', color='blue', markersize=8)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.grid(alpha=0.3)
        plt.xlabel('Window')
        plt.ylabel('Average R² Out-of-Sample')
        plt.title('Model Performance across Time Windows')
        plt.savefig(os.path.join(output_dir, "rolling_window_r2_oos.png"))
        plt.close()
        
        # Generate time series plot of sign accuracy
        plt.figure(figsize=(14, 7))
        plt.plot(results['window'], results['sign_accuracy'], 'o-', color='green', markersize=8)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Guess')
        plt.grid(alpha=0.3)
        plt.xlabel('Window')
        plt.ylabel('Average Sign Accuracy')
        plt.title('Directional Accuracy across Time Windows')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "rolling_window_sign_accuracy.png"))
        plt.close()
    
    logger.info("Rolling window evaluation completed")
    return results