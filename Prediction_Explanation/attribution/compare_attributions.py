#!/usr/bin/env python3
"""
Compare mean attributions between two different attribution methods:
1. Attributions from individual samples (mean calculated from all accessions)
2. Integrated gradients (already averaged)

Usage:
    python compare_attributions.py --attr-file /path/to/attributions.csv --ig-file /path/to/integrated_gradients.csv --output /path/to/output.png
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_attribution_data(attr_file):
    """Load attribution data and compute mean attribution per position."""
    df = pd.read_csv(attr_file)
    
    # Group by position and compute mean attribution across all accessions
    mean_attr = df.groupby('position')['attribution'].mean().reset_index()
    mean_attr = mean_attr.sort_values('position')
    
    print(f"Loaded {len(df)} attribution records for {df['accession'].nunique()} accessions")
    print(f"Position range: {mean_attr['position'].min()} to {mean_attr['position'].max()}")
    
    return mean_attr

def load_integrated_gradients_data(ig_file):
    """Load integrated gradients data."""
    df = pd.read_csv(ig_file)
    
    # Select relevant columns
    ig_data = df[['position', 'mean_saliency']].copy()
    ig_data = ig_data.sort_values('position')
    
    print(f"Loaded {len(ig_data)} integrated gradient records")
    print(f"Position range: {ig_data['position'].min()} to {ig_data['position'].max()}")
    
    return ig_data

def normalize_data(data1, data2, method='minmax'):
    """
    Normalize two datasets separately to their own scales.
    
    Args:
        data1, data2: arrays to normalize
        method: 'minmax', 'zscore', or 'robust'
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1] for each array separately
        norm1 = (data1 - data1.min()) / (data1.max() - data1.min())
        norm2 = (data2 - data2.min()) / (data2.max() - data2.min())
    elif method == 'zscore':
        # Z-score normalization for each array separately
        norm1 = (data1 - data1.mean()) / data1.std()
        norm2 = (data2 - data2.mean()) / data2.std()
    elif method == 'robust':
        # Robust normalization using median and IQR for each array separately
        q75_1, q25_1 = np.percentile(data1, [75, 25])
        iqr_1 = q75_1 - q25_1
        norm1 = (data1 - np.median(data1)) / iqr_1
        
        q75_2, q25_2 = np.percentile(data2, [75, 25])
        iqr_2 = q75_2 - q25_2
        norm2 = (data2 - np.median(data2)) / iqr_2
    else:
        raise ValueError("Method must be 'minmax', 'zscore', or 'robust'")
    
    return norm1, norm2

def plot_comparison(attr_data, ig_data, output_path, normalize=True, norm_method='minmax'):
    """Plot comparison of the two attribution methods."""
    
    # Merge data on position
    merged = pd.merge(attr_data, ig_data, on='position', how='inner', suffixes=('_attr', '_ig'))
    
    if len(merged) == 0:
        raise ValueError("No overlapping positions found between the two datasets")
    
    print(f"Comparing {len(merged)} overlapping positions")
    
    # Extract values for plotting
    positions = merged['position'].values
    attr_values = merged['attribution'].values
    ig_values = merged['mean_saliency'].values
    
    # Normalize if requested
    if normalize:
        attr_norm, ig_norm = normalize_data(attr_values, ig_values, method=norm_method)
        attr_plot = attr_norm
        ig_plot = ig_norm
        y_label = f'Normalized Attribution ({norm_method})'
    else:
        attr_plot = attr_values
        ig_plot = ig_values
        y_label = 'Attribution'
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Overlay comparison plot
    ax.plot(positions, attr_plot, label='Input perturbation', color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax.plot(positions, ig_plot, label='Integrated Gradients', color='#ff7f0e', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel(y_label)
    ax.set_title('Attribution Comparison: Input perturbation vs Integrated Gradients\nSNP75, Dimension 16, BIO_11_6')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate correlation for summary stats
    correlation = np.corrcoef(attr_plot, ig_plot)[0, 1]
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    
    plt.close(fig)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Mean Attribution range: {attr_values.min():.6e} to {attr_values.max():.6e}")
    print(f"Integrated Gradients range: {ig_values.min():.6e} to {ig_values.max():.6e}")
    print(f"Mean Attribution std: {attr_values.std():.6e}")
    print(f"Integrated Gradients std: {ig_values.std():.6e}")
    
    return correlation, merged

def main():
    parser = argparse.ArgumentParser(description='Compare attribution methods')
    parser.add_argument('--attr-file', required=True, help='Path to attributions CSV file')
    parser.add_argument('--ig-file', required=True, help='Path to integrated gradients CSV file')
    parser.add_argument('--output', required=True, help='Output path for the plot')
    parser.add_argument('--no-normalize', action='store_true', help='Do not normalize the data')
    parser.add_argument('--norm-method', choices=['minmax', 'zscore', 'robust'], default='minmax',
                       help='Normalization method (default: minmax)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading attribution data...")
    attr_data = load_attribution_data(args.attr_file)
    
    print("Loading integrated gradients data...")
    ig_data = load_integrated_gradients_data(args.ig_file)
    
    # Create comparison plot
    print("Creating comparison plot...")
    normalize = not args.no_normalize
    correlation, merged_data = plot_comparison(
        attr_data, ig_data, args.output, 
        normalize=normalize, norm_method=args.norm_method
    )
    
    print(f"\nPlot saved to: {args.output}")
    print(f"Correlation between methods: {correlation:.4f}")

if __name__ == "__main__":
    main()
