import os
import yaml
import json
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import time
import psutil  # For memory tracking
import sys
import glob
from src.data_utils import load_multiple_parquet_files, preprocess_lob_data
from src.feature_engineering import (
    calculate_ofi, 
    hybrid_alpha_term_structure, 
    create_lob_features,
    prepare_features_with_gpu
)

def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} minutes {int(secs)} seconds"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hours {int(minutes)} minutes"

def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Return in MB

def process_symbol_month(symbol, year, month, config_path="config/data_config.yaml", use_gpu=None):
    """Process a single symbol for a specific month with detailed logging."""
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"STARTING PROCESSING: {symbol} for {year}-{month:02d}")
    print(f"{'='*80}")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading configuration from {config_path}...")
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create output directory
    output_dir = os.path.join(config['data']['output_dir'], symbol, f"{year}-{month:02d}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Output directory: {output_dir}")
    
    # Define the directory containing all files for the month
    input_dir = f"C:/Users/trgrd/OneDrive/ByBit Data/{symbol}/{year}-{month:02d}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Input directory: {input_dir}")
    
    # Find all parquet files for the month
    file_pattern = f"{symbol}_{year}-{month:02d}-*.parquet.gzip"
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not input_files:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: No input files found matching pattern {file_pattern}")
        return False
    
    # Sort files to process them in chronological order
    input_files.sort()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(input_files)} files to process:")
    for i, file in enumerate(input_files):
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   {i+1}. {os.path.basename(file)} ({file_size_mb:.2f} MB)")
    
    total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in input_files)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total input size: {total_size_mb:.2f} MB")
    
    # Load and preprocess data
    try:
        # STEP 1: Load raw data from all files
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 1/4: Loading raw data from all files...")
        step_start = time.time()
        
        # Option 1: Load all files at once (if memory allows)
        df = load_multiple_parquet_files(input_dir, pattern=file_pattern)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory usage: {get_memory_usage():.2f} MB")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sample of columns: {list(df.columns[:10])}...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Time elapsed: {format_time(time.time() - step_start)}")
        
        # STEP 2: Preprocess data
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 2/4: Preprocessing data...")
        step_start = time.time()
        initial_row_count = len(df)
        df = preprocess_lob_data(
            df, 
            levels=config['preprocessing']['levels'],
            convert_timestamps=config['preprocessing']['convert_timestamps'],
            timestamp_col=config['preprocessing']['timestamp_col'],
            add_mid_price=config['preprocessing']['add_mid_price'],
            clean_outliers=config['preprocessing']['clean_outliers'],
            sort_by_time=config['preprocessing']['sort_by_time']
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] After preprocessing: {len(df):,} rows")
        rows_removed = initial_row_count - len(df)
        if rows_removed > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Removed {rows_removed:,} rows during preprocessing")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory usage: {get_memory_usage():.2f} MB")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Time elapsed: {format_time(time.time() - step_start)}")
        
        # STEP 3: Feature engineering using optimized methods
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 3/4: Feature engineering...")
        step_start = time.time()
        
        # Check for GPU availability
        try:
            import cupy as cp
            has_gpu = True
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU acceleration is available!")
            if use_gpu is None:
                use_gpu = True
        except ImportError:
            has_gpu = False
            use_gpu = False
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU acceleration not available - using CPU only")
        
        # Use the combined optimized method for feature engineering
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Using prepare_features_with_gpu (GPU: {use_gpu})")
        features_df, targets_df = prepare_features_with_gpu(
            df,
            latency_ms=config['features']['alpha_term_structure']['latency_ms'],
            levels=config['preprocessing']['levels'],
            normalize=config['features']['additional_features']['normalize'],
            use_gpu=use_gpu,
            n_jobs=-1  # Use all available CPU cores if using CPU
        )
        
        # Combine features and targets back into one dataframe for saving
        df = pd.concat([features_df, targets_df], axis=1)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature engineering complete")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Final dataframe shape: {df.shape}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory usage: {get_memory_usage():.2f} MB")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Time elapsed: {format_time(time.time() - step_start)}")
        
        # STEP 4: Save processed data
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 4/4: Saving processed data...")
        step_start = time.time()
        
        # Save to parquet
        output_path = os.path.join(output_dir, f"{symbol}_{year}-{month:02d}_processed.parquet")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving data to {output_path}")
        df.to_parquet(output_path, compression="gzip")
        
        # Calculate file size difference
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved processed data: {output_size_mb:.2f} MB")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Compression ratio: {output_size_mb/total_size_mb:.2f}x")
        
        # Save a small metadata file with processing info
        metadata = {
            "symbol": symbol,
            "year": year,
            "month": month,
            "num_files_processed": len(input_files),
            "file_list": [os.path.basename(f) for f in input_files],
            "rows": len(df),
            "columns": list(df.columns),
            "time_range": [df['time'].min().strftime('%Y-%m-%d %H:%M:%S'), 
                          df['time'].max().strftime('%Y-%m-%d %H:%M:%S')],
            "file_size_mb": output_size_mb,
            "input_size_mb": total_size_mb,
            "compression_ratio": output_size_mb/total_size_mb,
            "processing_time_seconds": time.time() - start_time,
            "gpu_used": use_gpu,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Time elapsed: {format_time(time.time() - step_start)}")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE: {symbol} for {year}-{month:02d}")
        print(f"Processed {len(input_files)} files covering {df['time'].min()} to {df['time'].max()}")
        print(f"Total time: {format_time(total_time)}")
        print(f"Final memory usage: {get_memory_usage():.2f} MB")
        print(f"Output file: {output_path}")
        print(f"{'='*80}\n")
        
        return True
    
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR processing {symbol} for {year}-{month:02d}: {e}")
        print(f"{'!'*80}\n")
        import traceback
        traceback.print_exc()
        return False

def process_symbol_month_chunked(symbol, year, month, config_path="config/data_config.yaml", use_gpu=None):
    """
    Process each day separately with GPU acceleration and save incrementally.
    Memory-optimized version that prevents OOM errors with large datasets.
    """
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"STARTING CHUNKED PROCESSING: {symbol} for {year}-{month:02d}")
    print(f"{'='*80}")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading configuration from {config_path}...")
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create output directory
    output_dir = os.path.join(config['data']['output_dir'], symbol, f"{year}-{month:02d}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Output directory: {output_dir}")
    
    # Define the directory containing all files for the month
    input_dir = f"C:/Users/trgrd/OneDrive/ByBit Data/{symbol}/{year}-{month:02d}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Input directory: {input_dir}")
    
    # Find all parquet files for the month
    file_pattern = f"{symbol}_{year}-{month:02d}-*.parquet.gzip"
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not input_files:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: No input files found matching pattern {file_pattern}")
        return False
    
    # Sort files to process them in chronological order
    input_files.sort()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(input_files)} files to process:")
    for i, file in enumerate(input_files):
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   {i+1}. {os.path.basename(file)} ({file_size_mb:.2f} MB)")
    
    total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in input_files)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total input size: {total_size_mb:.2f} MB")
    
    # Check for GPU availability
    try:
        import cupy as cp
        has_gpu = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU acceleration is available!")
        if use_gpu is None:
            use_gpu = True
    except ImportError:
        has_gpu = False
        use_gpu = False
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU acceleration not available - using CPU only")
    
    # Process each file individually
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing files in chunks...")
    
    # ---- MEMORY OPTIMIZATION: Process in smaller batches and save incrementally ----
    batch_size = 5  # Process files in batches of 5 days
    all_batches_metadata = []
    chunk_files = []  # Track intermediate files
    total_rows = 0
    time_min = None
    time_max = None
    
    # Helper function to reduce memory usage of dataframes
    def reduce_memory_usage(df):
        """Reduce memory usage by converting numeric columns to smaller dtypes."""
        start_mem = df.memory_usage().sum() / 1024**2
        
        # Process object columns (mostly strings)
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'time':  # Skip timestamp column
                df[col] = df[col].astype('category')
        
        # Process numeric columns
        # for col in df.select_dtypes(include=['float64']).columns:
        #     df[col] = df[col].astype('float32')
            
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
        
        end_mem = df.memory_usage().sum() / 1024**2
        savings = (start_mem - end_mem) / start_mem * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({savings:.2f}% saved)")
        return df
    
    try:
        # Process in smaller batches
        for batch_idx in range(0, len(input_files), batch_size):
            batch_files = input_files[batch_idx:batch_idx + batch_size]
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing batch {batch_idx//batch_size + 1}/{(len(input_files) + batch_size - 1)//batch_size}: files {batch_idx+1}-{min(batch_idx+batch_size, len(input_files))}")
            
            batch_processed_dfs = []
            batch_metadata = []
            
            # Process each file in the current batch
            for i, file_path in enumerate(batch_files):
                file_name = os.path.basename(file_path)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing file {batch_idx+i+1}/{len(input_files)}: {file_name}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] File size: {file_size_mb:.2f} MB")
                
                # STEP 1: Load raw data for this file
                chunk_start = time.time()
                file_dir = os.path.dirname(file_path)
                specific_file_name = os.path.basename(file_path)
                df = load_multiple_parquet_files(file_dir, pattern=specific_file_name, limit=1)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(df):,} rows")
                
                # STEP 2: Preprocess data
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Preprocessing data...")
                initial_row_count = len(df)
                df = preprocess_lob_data(
                    df, 
                    levels=config['preprocessing']['levels'],
                    convert_timestamps=config['preprocessing']['convert_timestamps'],
                    timestamp_col=config['preprocessing']['timestamp_col'],
                    add_mid_price=config['preprocessing']['add_mid_price'],
                    clean_outliers=config['preprocessing']['clean_outliers'],
                    sort_by_time=config['preprocessing']['sort_by_time']
                )
                
                rows_removed = initial_row_count - len(df)
                if rows_removed > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Removed {rows_removed:,} rows during preprocessing")
                
                # STEP 3: Feature engineering for this chunk
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating features...")
                features_df, targets_df = prepare_features_with_gpu(
                    df,
                    latency_ms=config['features']['alpha_term_structure']['latency_ms'],
                    levels=config['preprocessing']['levels'],
                    normalize=False,  # Don't normalize yet
                    use_gpu=use_gpu,
                    n_jobs=-1
                )
                
                # Combine features and targets
                processed_df = pd.concat([features_df, targets_df], axis=1)
                processed_df = reduce_memory_usage(processed_df)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed chunk size: {processed_df.shape}")
                
                # Store essential metadata for this file
                target_cols = [f'target_horizon_{k}' for k in range(1, 11) if f'target_horizon_{k}' in processed_df.columns]
                feature_cols = [col for col in processed_df.columns if col not in target_cols and col != 'time']
                
                # Update time range tracking
                if time_min is None or processed_df['time'].min() < time_min:
                    time_min = processed_df['time'].min()
                
                if time_max is None or processed_df['time'].max() > time_max:
                    time_max = processed_df['time'].max()
                
                batch_metadata.append({
                    'file_name': file_name,
                    'original_rows': initial_row_count,
                    'processed_rows': len(processed_df),
                    'rows_removed': rows_removed,
                    'feature_count': len(feature_cols),
                    'time_range': [processed_df['time'].min().strftime('%Y-%m-%d %H:%M:%S'), 
                                 processed_df['time'].max().strftime('%Y-%m-%d %H:%M:%S')],
                    'processing_time': time.time() - chunk_start
                })
                
                # Store processed chunk
                batch_processed_dfs.append(processed_df)
                
                # Free memory
                del df, features_df, targets_df
                if use_gpu and has_gpu:
                    cp.get_default_memory_pool().free_all_blocks()
                import gc
                gc.collect()
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Chunk processed in {format_time(time.time() - chunk_start)}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory usage: {get_memory_usage():.2f} MB")
            
            # Combine this batch and save as temporary file
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Combining and saving batch {batch_idx//batch_size + 1}...")
            
            # ---- MEMORY OPTIMIZATION: Concatenate without copying ----
            batch_df = pd.concat(batch_processed_dfs, copy=False)
            batch_size_before = len(batch_df)
            
            # Sort by time (more memory efficient)
            batch_df = batch_df.sort_values('time')
            
            # ---- MEMORY OPTIMIZATION: Avoid reset_index() ----
            # Instead of reset_index(), which creates a copy, just create a new RangeIndex
            batch_df.index = pd.RangeIndex(len(batch_df))
            
            # Drop rows with missing values in targets
            target_cols = [col for col in batch_df.columns if col.startswith('target_horizon_')]
            valid_rows = ~batch_df[target_cols].isna().any(axis=1)
            batch_df = batch_df[valid_rows]
            batch_df.index = pd.RangeIndex(len(batch_df))
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Removed {batch_size_before - len(batch_df):,} rows with missing targets")
            
            # Save as a temporary file
            batch_file = os.path.join(output_dir, f"{symbol}_{year}-{month:02d}_batch_{batch_idx//batch_size + 1}.parquet")
            batch_df.to_parquet(batch_file, compression="gzip", index=False)
            chunk_files.append(batch_file)
            
            total_rows += len(batch_df)
            all_batches_metadata.extend(batch_metadata)
            
            # Free memory
            del batch_df, batch_processed_dfs
            gc.collect()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {batch_idx//batch_size + 1} saved to {batch_file}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Memory usage: {get_memory_usage():.2f} MB")
        
        # FINAL STEP: Create single final file incrementally
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Creating final processed file...")
        final_path = os.path.join(output_dir, f"{symbol}_{year}-{month:02d}_processed.parquet")
        
        # Get schema from first batch to initialize the file
        first_batch = pd.read_parquet(chunk_files[0], engine='pyarrow')
        target_cols = [col for col in first_batch.columns if col.startswith('target_horizon_')]
        schema = first_batch.dtypes.to_dict()
        
        # ---- MEMORY OPTIMIZATION: Write incrementally without loading all data ----
        # Use pyarrow for appending to parquet
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Initialize a writer with the schema
        schema_pa = pa.Schema.from_pandas(first_batch)
        writer = pq.ParquetWriter(final_path, schema_pa, compression='gzip')
        
        # Append each batch file
        final_rows = 0
        for i, batch_file in enumerate(chunk_files):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Appending batch {i+1}/{len(chunk_files)}...")
            batch_df = pd.read_parquet(batch_file, engine='pyarrow')
            batch_table = pa.Table.from_pandas(batch_df)
            writer.write_table(batch_table)
            final_rows += len(batch_df)
            del batch_df, batch_table
            gc.collect()
        
        # Close the writer
        writer.close()
        
        # Calculate file size
        output_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Final file saved: {output_size_mb:.2f} MB")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Compression ratio: {output_size_mb/total_size_mb:.2f}x")
        
        # Clean up temporary files
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cleaning up temporary files...")
        for file in chunk_files:
            os.remove(file)
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "year": year,
            "month": month,
            "num_files_processed": len(input_files),
            "file_list": [os.path.basename(f) for f in input_files],
            "file_details": all_batches_metadata,
            "rows": final_rows,
            "columns": list(schema.keys()),
            "time_range": [time_min.strftime('%Y-%m-%d %H:%M:%S'), 
                          time_max.strftime('%Y-%m-%d %H:%M:%S')],
            "file_size_mb": output_size_mb,
            "input_size_mb": total_size_mb,
            "compression_ratio": output_size_mb/total_size_mb,
            "processing_time_seconds": time.time() - start_time,
            "gpu_used": use_gpu,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"CHUNKED PROCESSING COMPLETE: {symbol} for {year}-{month:02d}")
        print(f"Processed {len(input_files)} files covering {time_min} to {time_max}")
        print(f"Total time: {format_time(total_time)}")
        print(f"Final memory usage: {get_memory_usage():.2f} MB")
        print(f"Output file: {final_path}")
        print(f"{'='*80}\n")
        
        return True
    
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR during chunked processing of {symbol} for {year}-{month:02d}: {e}")
        print(f"{'!'*80}\n")
        import traceback
        traceback.print_exc()
        
        # Clean up any temporary files if an error occurs
        for file in chunk_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
        
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process symbol data for a specific month')
    parser.add_argument('--symbol', type=str, required=True, choices=['ADAUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'], 
                      help='Symbol to process')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2024)')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
    parser.add_argument('--config', type=str, default='config/data_config.yaml', help='Path to config file')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--chunked', action='store_true', 
                      help='Process files one by one (use for memory-constrained systems)')
    
    args = parser.parse_args()
    
    # Determine GPU usage based on flags
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False
    
    if args.chunked:
        process_symbol_month_chunked(args.symbol, args.year, args.month, args.config, use_gpu)
    else:
        process_symbol_month(args.symbol, args.year, args.month, args.config, use_gpu)

# Usage examples:
# Process entire month at once:
# python process_data.py --symbol BTCUSDT --year 2024 --month 1
# 
# Process with GPU acceleration:
# python -m src.process_data --symbol BTCUSDT --year 2024 --month 1 --gpu
#
# Process day by day (for memory-constrained systems):
# python -m src.process_data --symbol BTCUSDT --year 2024 --month 1 --chunked --gpu