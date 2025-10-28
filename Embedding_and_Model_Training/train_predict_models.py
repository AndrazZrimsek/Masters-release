#!/usr/bin/env python3
"""
Machine Learning Pipeline for Arabidopsis Environmental Variables Prediction

This script:
1. Loads embeddings from Masters/Embeddings/
2. Loads environmental data from CSV files
3. Trains MultiOutput SVM models for all environmental variables simultaneously
4. Uses train/validation splits from GWAS/train_ids.txt and GWAS/val_ids.txt
5. Evaluates and saves results

Usage:
    python train_predict_models.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import TruncatedSVD
import warnings
import argparse
from pathlib import Path
from scipy.stats import spearmanr  # Added for Spearman correlation

# Try to import torch if available (for tensor conversion)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

def load_embeddings(embeddings_dir, vstack=False, average_embeddings=False):
    """
    Load all embedding files from the specified directory.
    
    Args:
        embeddings_dir (str): Path to the embeddings directory
        vstack (bool): If True, stack chunk embeddings vertically (default: False)
        average_embeddings (bool): If True, average all chunk embeddings per accession to a single vector
        
    Returns:
        dict: Dictionary with embedding file names as keys and embeddings as values
    """
    embeddings = {}
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    # Find all .pkl files in the embeddings directory
    pkl_files = list(embeddings_path.glob("*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {embeddings_dir}")
    
    print(f"Found {len(pkl_files)} embedding files:")
    for pkl_file in pkl_files:
        print(f"  - {pkl_file.name}")
        
        try:
            with open(pkl_file, 'rb') as f:
                embedding_data = pickle.load(f)
            
            # Process embeddings based on the structure from embed_sequences.py with --join_regions
            processed_embeddings = {}
            for accession_id, embedding_list in embedding_data.items():
                # Handle possible nested structure: [chr1_list, chr2_list, chr3_list, chr4_list, chr5_list]
                if isinstance(embedding_list, list) and len(embedding_list) == 5 and all(isinstance(x, (list, np.ndarray)) for x in embedding_list):
                    flat_list = []
                    for part in embedding_list:
                        if isinstance(part, list):
                            flat_list.extend(part)
                        else:
                            flat_list.append(part)
                    embedding_list = flat_list
                # Now expect a flat list of np.ndarray
                all_embeddings = []
                for i, seq_embedding in enumerate(embedding_list):
                    # If any residual list sneaks in, flatten one level
                    if isinstance(seq_embedding, list):
                        for sub in seq_embedding:
                            if TORCH_AVAILABLE and hasattr(sub, 'cpu'):
                                sub = sub.cpu().detach().to(torch.float32).numpy()
                            all_embeddings.append(sub)
                        continue
                    # Convert bfloat16 to float32 if needed
                    if TORCH_AVAILABLE and hasattr(seq_embedding, 'cpu'):
                        seq_embedding = seq_embedding.cpu().detach().to(torch.float32).numpy()
                    all_embeddings.append(seq_embedding)
                    # print(f"      Accession {accession_id} chunk {i} shape: {np.shape(seq_embedding)}")
                if all_embeddings:
                    if average_embeddings:
                        processed_embeddings[accession_id] = np.mean(all_embeddings, axis=0)
                    elif vstack:
                        processed_embeddings[accession_id] = np.vstack(all_embeddings)
                    else:
                        processed_embeddings[accession_id] = np.hstack(all_embeddings)
            embeddings[pkl_file.stem] = processed_embeddings
            print(f"    Loaded {len(processed_embeddings)} accessions")
            
        except Exception as e:
            print(f"    Error loading {pkl_file.name}: {e}")
            continue
    
    return embeddings

def load_environmental_data(csv_files):
    """
    Load environmental data from CSV files.
    
    Args:
        csv_files (list): List of CSV file paths
        
    Returns:
        dict: Dictionary with CSV file names as keys and DataFrames as values
    """
    env_data = {}
    
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_file}")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {csv_path.name}: {df.shape[0]} samples, {df.shape[1]} columns")
            print(f"  Columns: {list(df.columns)}")
            # Drop unnecessary columns
            if 'FID' in df.columns:
                df = df.drop('FID', axis=1)
            if 'LONG' in df.columns:
                df = df.drop('LONG', axis=1)
            if 'LAT' in df.columns:
                df = df.drop('LAT', axis=1)
            # Drop columns ending with "_uncertainty"
            uncertainty_columns = [col for col in df.columns if col.endswith('_uncertainty')]
            if uncertainty_columns:
                print(f"  Dropping {len(uncertainty_columns)} uncertainty columns")
                df = df.drop(uncertainty_columns, axis=1)

            # # Keep only IID, BIO1, BIO3, and BIO5 columns if they exist
            # columns_to_keep = ['IID']
            # for bio_col in ['BIO1', 'BIO3', 'BIO5']:
            #     if bio_col in df.columns:
            #         columns_to_keep.append(bio_col)
            #     else:
            #         print(f"  Warning: {bio_col} not found in {csv_path.name}")

            # # Keep only the specified columns
            # df = df[columns_to_keep]
                        
            print(f"  After dropping columns: {df.shape[1]} columns remain")
            env_data[csv_path.stem] = df
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return env_data

def load_train_val_ids(train_file, val_file):
    """
    Load training and validation IDs from text files.
    
    Args:
        train_file (str): Path to training IDs file
        val_file (str): Path to validation IDs file
        
    Returns:
        tuple: (train_ids, val_ids) as lists of strings
    """
    train_ids = []
    val_ids = []
    
    # Load training IDs
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            content = f.read().strip()
            train_ids = content.split()
        print(f"Loaded {len(train_ids)} training IDs")
    else:
        print(f"Warning: Training IDs file not found: {train_file}")
    
    # Load validation IDs
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            content = f.read().strip()
            val_ids = content.split()
        print(f"Loaded {len(val_ids)} validation IDs")
    else:
        print(f"Warning: Validation IDs file not found: {val_file}")
    
    return train_ids, val_ids

def prepare_data(embeddings_dict, env_data_dict, train_ids, val_ids):
    """
    Prepare training and validation data by matching embeddings with environmental data.
    
    Args:
        embeddings_dict (dict): Dictionary of embeddings
        env_data_dict (dict): Dictionary of environmental DataFrames
        train_ids (list): List of training IDs
        val_ids (list): List of validation IDs
        
    Returns:
        dict: Prepared data for each embedding file (no longer embedding x dataset)
    """
    prepared_data = {}
    for emb_name, embeddings in embeddings_dict.items():
        # Use only the first (and only) environmental dataset
        env_name, env_df = list(env_data_dict.items())[0]
        env_df = env_df.copy()
        env_df['IID'] = env_df['IID'].astype(str)
        embedding_ids = set(embeddings.keys())
        env_ids = set(env_df['IID'].values)
        common_ids = embedding_ids.intersection(env_ids)
        print(f"\n{emb_name}:")
        print(f"  Embedding IDs: {len(embedding_ids)}")
        print(f"  Environment IDs: {len(env_ids)}")
        print(f"  Common IDs: {len(common_ids)}")
        if len(common_ids) == 0:
            print(f"  Warning: No common IDs found")
            continue
        train_common = [id for id in train_ids if id in common_ids]
        val_common = [id for id in val_ids if id in common_ids]
        print(f"  Train IDs (common): {len(train_common)}")
        print(f"  Val IDs (common): {len(val_common)}")
        if len(train_common) == 0 or len(val_common) == 0:
            print(f"  Warning: Insufficient train or validation data")
            continue
        env_vars = [col for col in env_df.columns if col not in ['FID', 'IID', 'LONG', 'LAT']]
        if not env_vars:
            print(f"  Warning: No environmental variables found")
            continue
        train_env = env_df[env_df['IID'].isin(train_common)].set_index('IID').loc[train_common]
        val_env = env_df[env_df['IID'].isin(val_common)].set_index('IID').loc[val_common]
        y_train_matrix = train_env[env_vars].values
        y_val_matrix = val_env[env_vars].values
        train_nan_mask = ~np.any(np.isnan(y_train_matrix), axis=1)
        val_nan_mask = ~np.any(np.isnan(y_val_matrix), axis=1)
        if not np.all(train_nan_mask):
            print(f"  Removing {np.sum(~train_nan_mask)} training samples with NaN values")
            train_common = [train_common[i] for i in range(len(train_common)) if train_nan_mask[i]]
            y_train_matrix = y_train_matrix[train_nan_mask]
        if not np.all(val_nan_mask):
            print(f"  Removing {np.sum(~val_nan_mask)} validation samples with NaN values")
            val_common = [val_common[i] for i in range(len(val_common)) if val_nan_mask[i]]
            y_val_matrix = y_val_matrix[val_nan_mask]
        if len(train_common) == 0 or len(val_common) == 0:
            print(f"  Warning: No valid samples after removing NaN values")
            continue
        X_train = np.array([embeddings[id] for id in train_common])
        X_val = np.array([embeddings[id] for id in val_common])
        print(f"  Final data shapes: X_train {X_train.shape}, y_train {y_train_matrix.shape}")
        print(f"                     X_val {X_val.shape}, y_val {y_val_matrix.shape}")
        print(f"  Environmental variables: {env_vars}")
        prepared_data[emb_name] = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train_matrix,
            'y_val': y_val_matrix,
            'env_vars': env_vars,
            'train_ids': train_common,
            'val_ids': val_common
        }
    return prepared_data

def apply_svd_to_embeddings(embeddings, svd_dim):
    """
    Optionally apply SVD to reduce embedding dimensionality for each accession.
    Args:
        embeddings (dict): accession_id -> embedding matrix (n_snps x n_features)
        svd_dim (int): target number of columns (features) after SVD
    Returns:
        dict: accession_id -> reduced embedding matrix (n_snps x svd_dim)
    """
    if svd_dim <= 0:
        return embeddings
    reduced = {}
    for acc, emb in embeddings.items():
        if emb.shape[1] > svd_dim:
            svd = TruncatedSVD(n_components=svd_dim, random_state=42)
            reduced_emb = svd.fit_transform(emb)
            # Flatten to 1D vector
            reduced[acc] = reduced_emb.flatten()
        else:
            reduced[acc] = emb.flatten() if emb.ndim > 1 else emb
    return reduced

def train_multioutput_svm_model(X_train, y_train, X_val, y_val, env_vars, param_grid=None):
    """
    Train a MultiOutput SVM regression model with hyperparameter tuning.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training targets matrix (samples x variables)
        X_val (np.array): Validation features  
        y_val (np.array): Validation targets matrix (samples x variables)
        env_vars (list): List of environmental variable names
        param_grid (dict): Parameter grid for hyperparameter tuning
        
    Returns:
        dict: Dictionary with model, predictions, and metrics
    """
    if param_grid is None:
        param_grid = {
            'estimator__C': np.logspace(1, 2, 6).tolist(),  # C values from 0.01 to 100
            'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'estimator__kernel': ['rbf'],#, 'linear']
        }
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_models = []
    best_params = []
    y_train_pred = np.zeros_like(y_train)
    y_val_pred = np.zeros_like(y_val)
    metrics = {}

    for i, var_name in enumerate(env_vars):
        y_train_var = y_train[:, i]
        y_val_var = y_val[:, i]
        svr = SVR()
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_var)
        best_model = grid_search.best_estimator_
        best_models.append(best_model)
        best_params.append(grid_search.best_params_)
        y_train_pred[:, i] = best_model.predict(X_train_scaled)
        y_val_pred[:, i] = best_model.predict(X_val_scaled)
        # Calculate Spearman correlation
        train_spearman = spearmanr(y_train_var, y_train_pred[:, i]).correlation
        val_spearman = spearmanr(y_val_var, y_val_pred[:, i]).correlation
        metrics[var_name] = {
            'train_mse': mean_squared_error(y_train_var, y_train_pred[:, i]),
            'val_mse': mean_squared_error(y_val_var, y_val_pred[:, i]),
            'train_r2': r2_score(y_train_var, y_train_pred[:, i]),
            'val_r2': r2_score(y_val_var, y_val_pred[:, i]),
            'train_mae': mean_absolute_error(y_train_var, y_train_pred[:, i]),
            'val_mae': mean_absolute_error(y_val_var, y_val_pred[:, i]),
            'train_spearman': train_spearman,
            'val_spearman': val_spearman,
            'best_params': grid_search.best_params_
        }

    # Overall Spearman (mean across variables)
    overall_train_spearman = np.mean([
        spearmanr(y_train[:, i], y_train_pred[:, i]).correlation for i in range(y_train.shape[1])
    ])
    overall_val_spearman = np.mean([
        spearmanr(y_val[:, i], y_val_pred[:, i]).correlation for i in range(y_val.shape[1])
    ])

    overall_metrics = {
        'overall_train_mse': mean_squared_error(y_train, y_train_pred),
        'overall_val_mse': mean_squared_error(y_val, y_val_pred),
        'overall_train_r2': r2_score(y_train, y_train_pred),
        'overall_val_r2': r2_score(y_val, y_val_pred),
        'overall_train_mae': mean_absolute_error(y_train, y_train_pred),
        'overall_val_mae': mean_absolute_error(y_val, y_val_pred),
        'overall_train_spearman': overall_train_spearman,
        'overall_val_spearman': overall_val_spearman
    }

    return {
        'model': best_models,
        'scaler': scaler,
        'best_params': best_params,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred,
        'individual_metrics': metrics,
        'overall_metrics': overall_metrics,
        'env_vars': env_vars
    }

def train_multioutput_elasticnet_model(X_train, y_train, X_val, y_val, env_vars, param_grid=None):
    """
    Train a MultiOutput ElasticNet regression model with hyperparameter tuning.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training targets matrix (samples x variables)
        X_val (np.array): Validation features  
        y_val (np.array): Validation targets matrix (samples x variables)
        env_vars (list): List of environmental variable names
        param_grid (dict): Parameter grid for hyperparameter tuning
        
    Returns:
        dict: Dictionary with model, predictions, and metrics
    """
    if param_grid is None:
        param_grid = {
            'alpha': np.logspace(-2, 1, 16).tolist(), 
            'l1_ratio': [0.1],
            'max_iter': [25000]  # High limit to ensure convergence with high-dimensional embeddings
        }
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # X_train_scaled = X_train
    # X_val_scaled = X_val

    best_models = []
    best_params = []
    y_train_pred = np.zeros_like(y_train)
    y_val_pred = np.zeros_like(y_val)
    metrics = {}

    for i, var_name in enumerate(env_vars):
        y_train_var = y_train[:, i]
        y_val_var = y_val[:, i]
        enet = ElasticNet()  # Add tolerance for better convergence detection
        grid_search = GridSearchCV(enet, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_var)
        best_model = grid_search.best_estimator_
        best_models.append(best_model)
        best_params.append(grid_search.best_params_)
        y_train_pred[:, i] = best_model.predict(X_train_scaled)
        y_val_pred[:, i] = best_model.predict(X_val_scaled)
        # Calculate Spearman correlation
        train_spearman = spearmanr(y_train_var, y_train_pred[:, i]).correlation
        val_spearman = spearmanr(y_val_var, y_val_pred[:, i]).correlation
        metrics[var_name] = {
            'train_mse': mean_squared_error(y_train_var, y_train_pred[:, i]),
            'val_mse': mean_squared_error(y_val_var, y_val_pred[:, i]),
            'train_r2': r2_score(y_train_var, y_train_pred[:, i]),
            'val_r2': r2_score(y_val_var, y_val_pred[:, i]),
            'train_mae': mean_absolute_error(y_train_var, y_train_pred[:, i]),
            'val_mae': mean_absolute_error(y_val_var, y_val_pred[:, i]),
            'train_spearman': train_spearman,
            'val_spearman': val_spearman,
            'best_params': grid_search.best_params_
        }

    # Overall Spearman (mean across variables)
    overall_train_spearman = np.mean([
        spearmanr(y_train[:, i], y_train_pred[:, i]).correlation for i in range(y_train.shape[1])
    ])
    overall_val_spearman = np.mean([
        spearmanr(y_val[:, i], y_val_pred[:, i]).correlation for i in range(y_val.shape[1])
    ])

    overall_metrics = {
        'overall_train_mse': mean_squared_error(y_train, y_train_pred),
        'overall_val_mse': mean_squared_error(y_val, y_val_pred),
        'overall_train_r2': r2_score(y_train, y_train_pred),
        'overall_val_r2': r2_score(y_val, y_val_pred),
        'overall_train_mae': mean_absolute_error(y_train, y_train_pred),
        'overall_val_mae': mean_absolute_error(y_val, y_val_pred),
        'overall_train_spearman': overall_train_spearman,
        'overall_val_spearman': overall_val_spearman
    }

    return {
        'model': best_models,
        'scaler': scaler,
        'best_params': best_params,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred,
        'individual_metrics': metrics,
        'overall_metrics': overall_metrics,
        'env_vars': env_vars
    }

def main():
    parser = argparse.ArgumentParser(description='Train MultiOutput ML models for environmental variable prediction')
    parser.add_argument('--embeddings_dir', default='Masters/Embeddings', 
                       help='Directory containing embedding files')
    parser.add_argument('--csv_files', nargs='+', 
                       default=['Masters/GWAS/coords_with_bioclim_30s_fixed.csv', 
                               'Masters/GWAS/coords_with_soil.csv'],
                       help='List of CSV files with environmental data')
    parser.add_argument('--train_ids', default='Masters/GWAS/train_ids.txt',
                       help='File containing training IDs')
    parser.add_argument('--val_ids', default='Masters/GWAS/val_ids.txt', 
                       help='File containing validation IDs')
    parser.add_argument('--output_dir', default='Masters/Results',
                       help='Directory to save results')
    parser.add_argument('--model', default='elasticnet', choices=['svm', 'elasticnet'], help='Model type to train: svm or elasticnet')
    parser.add_argument('--job_id', default=None, help='SLURM job ID for unique output filenames')
    parser.add_argument('--jobname', default=None, help='Job name for output folder (e.g., SLURM job name)')
    parser.add_argument('--svd-dim', type=int, default=0,
        help='If >0, reduce embedding columns to this dimension using SVD (e.g., 10)')
    parser.add_argument('--average-embeddings', action='store_true',
        help='Average all chunk embeddings per accession to a single vector (for variable-length chunked embeddings)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Arabidopsis Environmental Variables Multi-Output Prediction ===\n")
    
    # Load data
    print("1. Loading embeddings...")
    embeddings_dict = load_embeddings(args.embeddings_dir, args.svd_dim > 0, args.average_embeddings)
    if args.svd_dim > 0:
        print(f"Applying SVD to reduce embedding columns to {args.svd_dim}...")
        for emb_name in embeddings_dict:
            embeddings_dict[emb_name] = apply_svd_to_embeddings(embeddings_dict[emb_name], args.svd_dim)
    
    print("\n2. Loading environmental data...")
    env_data_dict = load_environmental_data(args.csv_files)
    
    print("\n3. Loading train/validation splits...")
    train_ids, val_ids = load_train_val_ids(args.train_ids, args.val_ids)
    
    print("\n4. Preparing data...")
    prepared_data = prepare_data(embeddings_dict, env_data_dict, train_ids, val_ids)
    
    if not prepared_data:
        print("No valid data combinations found. Exiting.")
        return
    
    print(f"\n5. Training multi-output models for {len(prepared_data)} data combinations...")
    
    # Results storage
    all_results = []
    
    for data_name, data in prepared_data.items():
        print(f"\n--- Processing {data_name} ---")
        
        X_train = data['X_train']
        X_val = data['X_val']
        y_train = data['y_train']
        y_val = data['y_val']
        env_vars = data['env_vars']

        param_grid = {
            'C': np.logspace(-6, 2, 10).tolist(),
            'gamma': ['scale'],
            'kernel': ['linear']
        }
        

        print(f"  Training multi-output model for {len(env_vars)} variables: {env_vars}")
        try:
            if args.model == 'svm':
                result = train_multioutput_svm_model(
                    X_train, y_train, X_val, y_val, env_vars, param_grid=param_grid
                )
            elif args.model == 'elasticnet':
                result = train_multioutput_elasticnet_model(
                    X_train, y_train, X_val, y_val, env_vars, param_grid=None
                )
            else:
                raise ValueError(f"Unknown model type: {args.model}")
            
            # Store individual variable results
            for var_name in env_vars:
                var_metrics = result['individual_metrics'][var_name]
                result_entry = {
                    'data_combination': data_name,
                    'variable': var_name,
                    'model_type': 'multioutput',
                    'best_params': str(result['best_params']),
                    **var_metrics
                }
                all_results.append(result_entry)
                
                print(f"    {var_name} - Val R²: {var_metrics['val_r2']:.3f}, "
                      f"Val MSE: {var_metrics['val_mse']:.3f}, "
                      f"Val Spearman: {var_metrics['val_spearman']:.3f}")
            
            # Store overall results
            overall_result_entry = {
                'data_combination': data_name,
                'variable': 'overall',
                'model_type': 'multioutput',
                'best_params': str(result['best_params']),
                **result['overall_metrics']
            }
            all_results.append(overall_result_entry)
            
            print(f"    Overall - Val R²: {result['overall_metrics']['overall_val_r2']:.3f}, "
                  f"Val MSE: {result['overall_metrics']['overall_val_mse']:.3f}")
            
            # Save model
            if args.job_id and args.jobname:
                job_folder = f"{args.jobname}_{args.job_id}"
                model_dir = os.path.join(args.output_dir, job_folder)
                os.makedirs(model_dir, exist_ok=True)
            else:
                model_dir = args.output_dir
            model_file = os.path.join(model_dir, f"{args.model}_model_{data_name}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': result['model'],
                    'scaler': result['scaler'],
                    'best_params': result['best_params'],
                    'env_vars': result['env_vars'],
                    'train_ids': data['train_ids'],
                    'val_ids': data['val_ids'],
                    'individual_metrics': result['individual_metrics'],
                    'overall_metrics': result['overall_metrics']
                }, f)
            print(f"    Model saved to: {model_file}")
            
        except Exception as e:
            print(f"    Error training multi-output model: {e}")
            continue
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    if args.job_id and args.jobname:
        job_folder = f"{args.jobname}_{args.job_id}"
        results_dir = os.path.join(args.output_dir, job_folder)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'multioutput_results_{args.job_id}.csv')
    else:
        results_dir = args.output_dir
        results_file = os.path.join(results_dir, 'multioutput_results.csv')
    results_df.to_csv(results_file, index=False)

    # Save per-variable summary for each data_name
    for data_name, data in prepared_data.items():
        # Filter for individual variable results for this data_name
        var_results = results_df[(results_df['data_combination'] == data_name) & (results_df['variable'] != 'overall')]
        if not var_results.empty:
            summary_file = os.path.join(results_dir, f'results_{data_name}.txt')
            with open(summary_file, 'w') as f:
                f.write('BIO_ID R2_Coefficient Spearman_Coefficient\n')
                for _, row in var_results.iterrows():
                    f.write(f"{row['variable']} {row['val_r2']:.3f} {row['val_spearman']:.3f}\n")
            print(f"   Variable summary saved to: {summary_file}")
    
    print(f"\n6. Results Summary:")
    print(f"   Total model combinations trained: {len(prepared_data)}")
    print(f"   Total variable results: {len(all_results)}")
    print(f"   Results saved to: {results_file}")
    
    # Print top performing models by individual variables
    if len(all_results) > 0:
        # Filter out overall results for top models display
        individual_results = results_df[results_df['variable'] != 'overall']
        
        if len(individual_results) > 0:
            print("\n   Top 5 individual variable models by validation R²:")
            top_models = individual_results.nlargest(5, 'val_r2')
            for _, row in top_models.iterrows():
                print(f"     {row['data_combination']} - {row['variable']}: "
                      f"R² = {row['val_r2']:.3f}")
        
        # Show overall model performance
        overall_results = results_df[results_df['variable'] == 'overall']
        if len(overall_results) > 0:
            print("\n   Overall multi-output model performance:")
            for _, row in overall_results.iterrows():
                print(f"     {row['data_combination']}: "
                      f"R² = {row['overall_val_r2']:.3f}, "
                      f"MSE = {row['overall_val_mse']:.3f}")

if __name__ == "__main__":
    main()
