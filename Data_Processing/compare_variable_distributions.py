#!/usr/bin/env python3
"""
Compare distributions of variables before and after combining highly correlated variables.
Creates visualization to show how the combination process affects the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_data():
    """Load original and combined datasets"""
    print("Loading datasets...")
    
    # Load original data
    bioclim = pd.read_csv("coords_with_bioclim_30s_fixed.csv")
    soil = pd.read_csv("coords_with_soil.csv")
    
    # Load train IDs
    with open("train_ids_plink.txt") as f:
        train_ids = set(line.strip().split()[0] for line in f)
    
    # Filter and merge original data
    bioclim = bioclim.replace(-3.4e+38, np.nan).dropna()
    soil = soil.replace(-3.4e+38, np.nan).dropna()
    
    bioclim = bioclim[bioclim["IID"].astype(str).isin(train_ids)]
    soil = soil[soil["IID"].astype(str).isin(train_ids)]
    
    original_data = pd.merge(bioclim, soil, on=["LONG", "LAT", "IID", "FID"], how='inner')
    
    # Get numeric columns from original data
    original_numeric_cols = [col for col in original_data.columns 
                           if col not in ['IID', 'FID', 'LONG', 'LAT'] and not col.endswith('_uncertainty')]
    
    # Load combined data (now normalized)
    combined_data = pd.read_csv("combined_variables_dataset_normalized.csv")
    combined_numeric_cols = [col for col in combined_data.columns 
                           if col not in ['IID', 'FID', 'LONG', 'LAT']]
    
    # Load variable mapping
    mapping_df = pd.read_csv("Results/variable_combination_mapping.csv")
    
    print(f"Original data: {original_data.shape} with {len(original_numeric_cols)} variables")
    print(f"Combined data: {combined_data.shape} with {len(combined_numeric_cols)} variables")
    
    return original_data, combined_data, original_numeric_cols, combined_numeric_cols, mapping_df

def plot_individual_vs_combined_distributions(original_data, combined_data, mapping_df):
    """Plot distributions of normalized original variables vs their normalized combined counterparts"""
    print("\nCreating normalized original vs normalized combined variable distribution plots...")
    
    from sklearn.preprocessing import StandardScaler
    
    # Get clusters that have combined variables
    combined_vars = mapping_df[mapping_df['Type'] == 'Combined']
    unique_combined = combined_vars['Final_Variable'].unique()
    
    # Create subplots for each combined variable
    n_combined = len(unique_combined)
    if n_combined == 0:
        print("No combined variables found!")
        return

    # --- Only keep the middle row (3rd & 4th overall assuming 2 columns) ---
    # We treat the layout as 2 columns; determine middle row index then pick those vars
    n_cols_full = 2 if n_combined >= 2 else n_combined
    n_rows_full = (n_combined + n_cols_full - 1) // n_cols_full
    if n_rows_full >= 3:
        # Middle row (0-based)
        mid_row = n_rows_full // 2
        start_idx = mid_row * n_cols_full
        end_idx = min(start_idx + n_cols_full, n_combined)
        selected_indices = list(range(start_idx, end_idx))
        # For clarity, if specifically wanting 3rd & 4th, ensure those exist and override
        if n_combined >= 4:
            selected_indices = [2,3]
        unique_combined = unique_combined[selected_indices]
        print(f"Plotting only middle row combined variables: {list(unique_combined)}")
    elif n_combined >= 4:
        # If there are at least 4 but < 3 rows (i.e., exactly 4 with 2 rows), still take 3rd & 4th
        unique_combined = unique_combined[2:4]
        print(f"Plotting only variables 3 & 4: {list(unique_combined)}")
    else:
        print("Fewer than 4 combined variables; plotting all available.")
    # Recompute counts after filtering
    n_combined = len(unique_combined)
    
    # Calculate grid dimensions for A4 report - use 2 columns for better fit
    n_cols = min(2, n_combined)
    n_rows = (n_combined + n_cols - 1) // n_cols
    
    # A4-appropriate figure size (width ~7 inches fits well in A4 with margins)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 3*n_rows))
    # Normalize axes to a 2D array for consistent indexing
    if n_rows == 1:
        # axes is 1D (length = n_cols) when n_rows == 1
        axes_2d = np.expand_dims(axes, axis=0)
    else:
        axes_2d = axes
    
    for i, combined_var in enumerate(unique_combined):
        row, col = divmod(i, n_cols)
        ax = axes_2d[row, col]
        
        # Get original variables that went into this combined variable
        orig_vars = combined_vars[combined_vars['Final_Variable'] == combined_var]['Original_Variable'].tolist()
        
        # Normalize original variables individually using z-score
        scaler = StandardScaler()
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for j, orig_var in enumerate(orig_vars):
            color = colors[j % len(colors)]
            normalized_orig = scaler.fit_transform(original_data[[orig_var]]).flatten()
            ax.hist(normalized_orig, alpha=0.5, bins=30, label=f'{orig_var}', density=True, color=color)
        
        # Plot normalized combined variable
        ax.hist(combined_data[combined_var].dropna(), alpha=0.5, bins=30, 
               label=f'{combined_var}', density=True, 
               color='red', edgecolor='red', linewidth=1, histtype='step')
        
        # Add statistics text with both original and normalized ranges - more compact for A4
        # orig_ranges = [f"{orig_var}: {original_data[orig_var].min():.0f}-{original_data[orig_var].max():.0f}" 
        #               for orig_var in orig_vars]
        # combined_range = f"{combined_data[combined_var].min():.2f}-{combined_data[combined_var].max():.2f}"
        
        # More compact title for A4 format
        ax.set_title(f'{combined_var}', 
                    fontsize=8)
        ax.set_xlabel('Normalized Value (Z-Score)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add vertical lines for means (should be ~0 for normalized data)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Zero mean')
    
    # Hide unused subplots
    for i in range(n_combined, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes_2d[row, col].set_visible(False)
    
    plt.suptitle('Normalized Original Variables vs Z-Score Combined Variables', fontsize=10, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.savefig("Plots/normalized_original_vs_combined_distributions.png", dpi=300, bbox_inches='tight')
    print("Saved: Plots/normalized_original_vs_combined_distributions.png")
    plt.show()


def main():
    """Main function to create all comparison plots"""
    print("="*60)
    print("COMPARING VARIABLE DISTRIBUTIONS BEFORE/AFTER COMBINATION")
    print("="*60)
    
    # Create plots directory if it doesn't exist
    os.makedirs("Plots", exist_ok=True)
    
    # Load data
    original_data, combined_data, original_numeric_cols, combined_numeric_cols, mapping_df = load_data()
    
    # Create comparison plots
    plot_individual_vs_combined_distributions(original_data, combined_data, mapping_df)

    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated plots:")
    print("  - Plots/normalized_original_vs_combined_distributions.png")
    # print("  - Plots/correlation_matrices_comparison.png") 
    # print("  - Plots/correlation_distributions.png")
    # print("  - Plots/variance_comparison_normalized.png")
    print("Generated files:")
    print("  - variable_combination_summary.csv")

if __name__ == "__main__":
    main()
