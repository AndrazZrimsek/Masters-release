#!/usr/bin/env python3
"""
Script to combine highly correlated variables into single variables per cluster.
Takes the high correlation clusters and creates a new dataset where:
- Variables in clusters (correlation > 0.9) are combined (averaged)
- Individual variables (not highly correlated) are kept as-is
"""

import pandas as pd
import numpy as np
import os

def load_data():
    """Load the original data and cluster information"""
    print("Loading data...")
    
    # Load original merged data
    bioclim = pd.read_csv("coords_with_bioclim_30s_fixed.csv")
    soil = pd.read_csv("coords_with_soil.csv")
    
    # Filter and merge data (keep ALL samples, not just training)
    bioclim = bioclim.replace(-3.4e+38, np.nan).dropna()
    soil = soil.replace(-3.4e+38, np.nan).dropna()
    
    # Merge all available data
    merged = pd.merge(bioclim, soil, on=["LONG", "LAT", "IID", "FID"], how='inner')
    
    # Get numeric columns
    numeric_cols = [col for col in merged.columns 
                    if col not in ['IID', 'FID', 'LONG', 'LAT'] and not col.endswith('_uncertainty')]
    
    print(f"Loaded data with {len(merged)} samples and {len(numeric_cols)} variables")
    print(f"Using ALL available samples (not filtered to training set)")
    return merged, numeric_cols

def load_clusters():
    """Load the high correlation cluster information"""
    print("Loading cluster information...")
    
    # Load high correlation clusters
    if os.path.exists("high_correlation_clusters.csv"):
        cluster_df = pd.read_csv("high_correlation_clusters.csv")
        print(f"Found {len(cluster_df)} variables in {cluster_df['High_Corr_Cluster'].nunique()} high correlation clusters")
    else:
        print("No high correlation clusters file found. Creating empty DataFrame.")
        cluster_df = pd.DataFrame(columns=['Variable', 'High_Corr_Cluster', 'Cluster_Size'])
    
    # Load variable analysis structure
    if os.path.exists("variable_analysis_structure.csv"):
        analysis_df = pd.read_csv("variable_analysis_structure.csv")
        print(f"Found {len(analysis_df)} analysis units")
    else:
        print("No variable analysis structure file found.")
        analysis_df = pd.DataFrame()
    
    return cluster_df, analysis_df

def create_cluster_name(cluster_vars):
    """Create a meaningful name for a cluster based on its variables"""
    # Sort variables for consistent naming
    sorted_vars = sorted(cluster_vars)
    
    # Identify common prefixes/patterns
    bio_vars = [v for v in sorted_vars if v.startswith('BIO')]
    soil_vars = [v for v in sorted_vars if not v.startswith('BIO')]
    
    if bio_vars and not soil_vars:
        # Only bioclimatic variables
        if len(bio_vars) <= 3:
            return f"BIO_{'_'.join([v.replace('BIO', '') for v in bio_vars])}"
        else:
            # Too many, use range or abbreviated form
            bio_nums = [int(v.replace('BIO', '')) for v in bio_vars]
            min_bio, max_bio = min(bio_nums), max(bio_nums)
            return f"BIO_{min_bio}to{max_bio}_group"
    
    elif soil_vars and not bio_vars:
        # Only soil variables
        if len(soil_vars) <= 2:
            # Use abbreviated soil names
            soil_abbrev = []
            for var in soil_vars:
                if 'clay' in var.lower():
                    soil_abbrev.append('Clay')
                elif 'bdod' in var.lower():
                    soil_abbrev.append('BulkDens')
                elif 'phh2o' in var.lower():
                    soil_abbrev.append('pH')
                elif 'soc' in var.lower():
                    soil_abbrev.append('SOC')
                elif 'nitrogen' in var.lower():
                    soil_abbrev.append('Nitrogen')
                elif 'wv0033' in var.lower():
                    soil_abbrev.append('WaterCont')
                else:
                    soil_abbrev.append(var[:8])  # First 8 chars if unknown
            return f"Soil_{'_'.join(soil_abbrev)}"
        else:
            return "Soil_Mixed"
    
    elif bio_vars and soil_vars:
        # Mixed bio and soil
        return f"Mixed_BIO{len(bio_vars)}_Soil{len(soil_vars)}"
    
    else:
        # Fallback - use first few variable names
        if len(sorted_vars) <= 3:
            return f"Combined_{'_'.join([v[:6] for v in sorted_vars])}"
        else:
            return f"Combined_{len(sorted_vars)}vars"

def combine_variables(merged_data, numeric_cols, cluster_df, normalization_method='zscore'):
    """Combine highly correlated variables and keep individual variables
    
    Args:
        normalization_method: 'zscore', 'minmax', 'robust', or 'none'
    """
    print(f"\nCombining variables using {normalization_method} normalization...")
    
    # Start with ID columns
    combined_data = merged_data[['IID', 'FID', 'LONG', 'LAT']].copy()
    
    # Get variables that are in clusters
    if not cluster_df.empty:
        clustered_variables = set(cluster_df['Variable'].tolist())
        
        # Process each cluster
        for cluster_id in cluster_df['High_Corr_Cluster'].unique():
            cluster_vars = cluster_df[cluster_df['High_Corr_Cluster'] == cluster_id]['Variable'].tolist()
            
            if len(cluster_vars) > 1:
                # Create meaningful cluster name
                combined_var_name = create_cluster_name(cluster_vars)
                print(f"  Combining cluster {cluster_id}: {cluster_vars}")
                print(f"    -> Creating variable: {combined_var_name}")
                
                # Get cluster data
                cluster_data = merged_data[cluster_vars].copy()
                
                # Normalize variables before combining
                if normalization_method == 'zscore':
                    # Z-score normalization (mean=0, std=1)
                    cluster_data_norm = (cluster_data - cluster_data.mean()) / cluster_data.std()
                    print(f"    -> Applied Z-score normalization")
                    
                elif normalization_method == 'minmax':
                    # Min-Max normalization (range 0-1)
                    cluster_data_norm = (cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min())
                    print(f"    -> Applied Min-Max normalization")
                    
                elif normalization_method == 'robust':
                    # Robust normalization using median and IQR
                    median = cluster_data.median()
                    q75 = cluster_data.quantile(0.75)
                    q25 = cluster_data.quantile(0.25)
                    iqr = q75 - q25
                    cluster_data_norm = (cluster_data - median) / iqr
                    print(f"    -> Applied Robust normalization")
                    
                else:  # 'none'
                    cluster_data_norm = cluster_data
                    print(f"    -> No normalization applied")
                
                # Calculate the mean of normalized variables
                combined_normalized = cluster_data_norm.mean(axis=1)
                
                # Optional: Scale back to original range
                # For now, keep normalized scale for consistency
                combined_data[combined_var_name] = combined_normalized
                
                # Print some statistics
                orig_ranges = cluster_data.max() - cluster_data.min()
                print(f"    -> Original ranges: {dict(orig_ranges)}")
                if normalization_method != 'none':
                    norm_ranges = cluster_data_norm.max() - cluster_data_norm.min()
                    print(f"    -> Normalized ranges: {dict(norm_ranges)}")
    else:
        clustered_variables = set()
    
    # Add individual variables (not in any cluster) - NOW NORMALIZED
    individual_vars = [var for var in numeric_cols if var not in clustered_variables]
    print(f"\nAdding {len(individual_vars)} individual variables (with normalization):")
    for var in individual_vars:
        var_data = merged_data[[var]].copy()
        
        # Apply same normalization as clusters
        if normalization_method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            var_normalized = (var_data - var_data.mean()) / var_data.std()
            print(f"  {var} (Z-score normalized)")
            
        elif normalization_method == 'minmax':
            # Min-Max normalization (range 0-1)
            var_normalized = (var_data - var_data.min()) / (var_data.max() - var_data.min())
            print(f"  {var} (Min-Max normalized)")
            
        elif normalization_method == 'robust':
            # Robust normalization using median and IQR
            median = var_data.median()
            q75 = var_data.quantile(0.75)
            q25 = var_data.quantile(0.25)
            iqr = q75 - q25
            var_normalized = (var_data - median) / iqr
            print(f"  {var} (Robust normalized)")
            
        else:  # 'none'
            var_normalized = var_data
            print(f"  {var} (no normalization)")
        
        combined_data[var] = var_normalized[var]
    
    print(f"\nFinal dataset shape: {combined_data.shape}")
    print(f"Original variables: {len(numeric_cols)}")
    print(f"Variables in clusters: {len(clustered_variables)}")
    print(f"Individual variables: {len(individual_vars)}")
    print(f"Combined clusters: {len(cluster_df['High_Corr_Cluster'].unique()) if not cluster_df.empty else 0}")
    print(f"Final variables: {len(combined_data.columns) - 4}")  # Subtract ID columns
    
    return combined_data

def create_variable_mapping(cluster_df, numeric_cols):
    """Create a mapping showing which original variables went into which combined variables"""
    mapping_data = []
    
    if not cluster_df.empty:
        # Add cluster mappings
        for cluster_id in cluster_df['High_Corr_Cluster'].unique():
            cluster_vars = cluster_df[cluster_df['High_Corr_Cluster'] == cluster_id]['Variable'].tolist()
            combined_var_name = create_cluster_name(cluster_vars)
            
            for var in cluster_vars:
                mapping_data.append({
                    'Original_Variable': var,
                    'Final_Variable': combined_var_name,
                    'Type': 'Combined',
                    'Cluster_ID': cluster_id,
                    'Cluster_Size': len(cluster_vars)
                })
        
        clustered_variables = set(cluster_df['Variable'].tolist())
    else:
        clustered_variables = set()
    
    # Add individual variable mappings
    individual_vars = [var for var in numeric_cols if var not in clustered_variables]
    for var in individual_vars:
        mapping_data.append({
            'Original_Variable': var,
            'Final_Variable': var,
            'Type': 'Individual',
            'Cluster_ID': np.nan,
            'Cluster_Size': 1
        })
    
    return pd.DataFrame(mapping_data)

def main():
    """Main function to combine correlated variables"""
    print("="*60)
    print("COMBINING HIGHLY CORRELATED VARIABLES")
    print("="*60)
    
    # Load data and clusters
    merged_data, numeric_cols = load_data()
    cluster_df, analysis_df = load_clusters()
    
    # Choose normalization method
    normalization_method = 'zscore'  # Options: 'zscore', 'minmax', 'robust', 'none'
    print(f"\nUsing normalization method: {normalization_method}")
    
    # Combine variables
    combined_data = combine_variables(merged_data, numeric_cols, cluster_df, normalization_method)
    
    # Create variable mapping
    mapping_df = create_variable_mapping(cluster_df, numeric_cols)
    
    # Save results
    print("\nSaving results...")
    
    # Save combined dataset (normalized)
    output_file = "combined_variables_dataset_normalized.csv"
    combined_data.to_csv(output_file, index=False)
    print(f"Combined dataset (normalized) saved to: {output_file}")
    
    # Save variable mapping
    mapping_file = "variable_combination_mapping.csv"
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Variable mapping saved to: {mapping_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original dataset: {merged_data.shape}")
    print(f"Combined dataset: {combined_data.shape}")
    print(f"Original variables: {len(numeric_cols)}")
    print(f"Final variables: {len(combined_data.columns) - 4}")
    print(f"Data reduction: {len(numeric_cols) - (len(combined_data.columns) - 4)} variables combined")
    
    if not cluster_df.empty:
        print(f"\nCluster details:")
        for cluster_id in cluster_df['High_Corr_Cluster'].unique():
            cluster_vars = cluster_df[cluster_df['High_Corr_Cluster'] == cluster_id]['Variable'].tolist()
            combined_name = create_cluster_name(cluster_vars)
            print(f"  Cluster {cluster_id}: {len(cluster_vars)} variables -> {combined_name}")
            print(f"    Variables: {', '.join(cluster_vars)}")
    
    # Show first few rows of mapping
    print(f"\nVariable mapping preview:")
    print(mapping_df.head(10).to_string(index=False))
    
    print(f"\nFiles created:")
    print(f"  - {output_file}")
    print(f"  - {mapping_file}")

if __name__ == "__main__":
    main()
