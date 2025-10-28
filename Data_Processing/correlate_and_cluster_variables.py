import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage

os.chdir("/home/andrazzrimsek/DataM/WorldClim/")

# Mapping for soil variable codes to full names
soil_name_map = {
    "wv0033_mean": "Volumetric Water Content",
    "clay_mean": "Clay Content",
    "bdod_mean": "Bulk Density",
    "phh2o_mean": "Soil pH (H2O)",
    "soc_mean": "Soil Organic Carbon",
    "nitrogen_mean": "Soil Nitrogen Content",
    "Soil_Nitrogen_SOC": "Soil Nitrogen & Organic Carbon"
}

bio_name_map = {
    "BIO1": "Annual Mean Temperature",
    "BIO2": "Mean Diurnal Range",
    "BIO3": "Isothermality",
    "BIO4": "Temperature Seasonality",
    "BIO5": "Max Temperature of Warmest Month",
    "BIO6": "Min Temperature of Coldest Month",
    "BIO7": "Temperature Annual Range",
    "BIO8": "Mean Temperature of Wettest Quarter",
    "BIO9": "Mean Temperature of Driest Quarter",
    "BIO10": "Mean Temperature of Warmest Quarter",
    "BIO11": "Mean Temperature of Coldest Quarter",
    "BIO12": "Annual Precipitation",
    "BIO13": "Precipitation of Wettest Month",
    "BIO14": "Precipitation of Driest Month",
    "BIO15": "Precipitation Seasonality",
    "BIO16": "Precipitation of Wettest Quarter",
    "BIO17": "Precipitation of Driest Quarter",
    "BIO18": "Precipitation of Warmest Quarter",
    "BIO19": "Precipitation of Coldest Quarter",
    "BIO_11_6": "Min & Mean Temperature of Coldest Period",
    "BIO_4_7": "Temperature Seasonality & Annual Range",
    "BIO_10_5":    "Max & Mean Temperature of Warmest Period",
    "BIO_14_17":   "Precipitation of Driest Period",
    "BIO_12_13_16": "Annual Precipitation & Wettest Periods"
}

def prettify_bio_id(bio_id):
    if bio_id in soil_name_map:
        return soil_name_map[bio_id]
    elif bio_id in bio_name_map:
        return bio_name_map[bio_id]
    else:
        return bio_id

# File paths
bioclim_path = "coords_with_bioclim_30s_fixed.csv"
soil_path = "coords_with_soil.csv"
train_ids_path = "train_ids_plink.txt"
val_ids_path = "val_ids_plink.txt"

# Load accession IDs for train and val
with open(train_ids_path) as f:
    train_ids = set(line.strip().split()[0] for line in f)
with open(val_ids_path) as f:
    val_ids = set(line.strip().split()[0] for line in f)
all_ids = train_ids# | val_ids

# Load bioclim and soil data
bioclim = pd.read_csv(bioclim_path, sep=',')
bioclim = bioclim.replace(-3.4e+38, np.nan).dropna()


soil = pd.read_csv(soil_path, sep=',')
soil = soil.replace(-3.4e+38, np.nan).dropna()

# Filter to only train+val individuals
bioclim = bioclim[bioclim["IID"].astype(str).isin(all_ids)]
soil = soil[soil["IID"].astype(str).isin(all_ids)]

# Merge on ID
merged = pd.merge(bioclim, soil, on=["LONG", "LAT", "IID", "FID"], how='inner')

# Drop non-numeric columns except ID
numeric_cols = [col for col in merged.columns if col not in ['IID', 'FID', 'LONG', 'LAT'] and not col.endswith('_uncertainty')]

# Standardize the data before clustering and correlation analysis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged[numeric_cols])

# Compute correlation matrix from standardized data
corr = merged[numeric_cols].corr()

n_clusters = 18

# Cluster variables by correlation using optimal leaf ordering
corr_linkage = linkage(corr.values, method='ward', optimal_ordering=True)
# Apply AgglomerativeClustering to the scaled data transposed (variables as samples)
labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(scaled_data.T)

# Assign cluster labels to variables
clustered_vars = pd.DataFrame({
    'Variable': numeric_cols,
    'Corr_Cluster': labels
})

# Save cluster assignments for later comparison
clustered_vars.to_csv("correlation_variable_clusters.csv", index=False)

# Save the correlation matrix (with original variable names)
corr.to_csv("correlation_matrix.csv")

# Create mapping from original names to prettified names
prettified_map = {var: prettify_bio_id(var) for var in numeric_cols}

# Create a copy of the correlation matrix with prettified names
corr_pretty = corr.copy()
corr_pretty.index = [prettified_map[var] for var in corr_pretty.index]
corr_pretty.columns = [prettified_map[var] for var in corr_pretty.columns]

print("Variable clusters (AgglomerativeClustering):")
for c in range(n_clusters):
    print(f"\nCluster {c+1}:")
    for v in clustered_vars[clustered_vars['Corr_Cluster'] == c]['Variable']:
        print(f"  {v}")

# Compute hierarchical linkage with optimal ordering for clustering (for clustermap)
corr_linkage = linkage(corr.values, method='ward', optimal_ordering=True)
corr_linkage_col = linkage(corr.values.T, method='ward', optimal_ordering=True)

# --- Add cluster color bars ---
# Create a color palette for 9 clusters
cluster_palette = sns.color_palette('tab20', n_clusters)
cluster_color_map = {i: cluster_palette[i] for i in range(n_clusters)}
# Map each variable to its cluster color using prettified names
pretty_clustered_vars = clustered_vars.copy()
pretty_clustered_vars['Pretty'] = pretty_clustered_vars['Variable'].map(prettified_map)
pretty_clustered_vars = pretty_clustered_vars.set_index('Pretty')

row_colors = pretty_clustered_vars.loc[corr_pretty.index]['Corr_Cluster'].map(cluster_color_map)
col_colors = pretty_clustered_vars.loc[corr_pretty.columns]['Corr_Cluster'].map(cluster_color_map)

# Optional: plot heatmap with optimal ordering and cluster color bars
g = sns.clustermap(
    corr_pretty,
    row_linkage=corr_linkage,
    col_linkage=corr_linkage_col,
    cmap='vlag',
    annot=True,
    fmt=".2f",
    dendrogram_ratio=(.1, .1),
    linewidths=0.75,
    figsize=(14, 13),
    row_colors=row_colors,
    col_colors=col_colors,
    row_cluster=True,
    col_cluster=True,
    xticklabels=False,
    yticklabels=True,
)

g.ax_row_dendrogram.remove()

# Remove the colorbar since we have annotations
g.cax.remove()

g.fig.suptitle('Variable Correlation Clustermap', fontsize=16, y=0.95)

# Add padding at the top for the title
g.fig.subplots_adjust(top=0.92)

# Manually reposition the row colors to the right side
heatmap_pos = g.ax_heatmap.get_position()
row_colors_pos = g.ax_row_colors.get_position()

# Position row colors on the right side of the heatmap
new_row_colors_pos = [heatmap_pos.x1, heatmap_pos.y0, 0.028, heatmap_pos.height]
g.ax_row_colors.set_position(new_row_colors_pos)

# Move the y-axis ticks and labels to the right side
g.ax_heatmap.yaxis.tick_right()
g.ax_heatmap.yaxis.set_label_position("right")

# Add padding to move the tick labels further right to avoid overlap with color bar
# Also adjust tick length and direction to align with the padding
g.ax_heatmap.tick_params(axis='y', which='major', pad=30, length=0, direction='out')

# Remove the axis labels from the color bars
g.ax_row_colors.set_xlabel('')
g.ax_row_colors.set_ylabel('')
# g.ax_col_colors.set_xlabel('')
# g.ax_col_colors.set_ylabel('')

# Remove all ticks and tick labels from color bar axes
g.ax_row_colors.set_xticks([])
g.ax_row_colors.set_yticks([])
# g.ax_col_colors.set_xticks([])
# g.ax_col_colors.set_yticks([])

# Turn off the axis for color bars completely
g.ax_row_colors.axis('off')
# g.ax_col_colors.axis('off')

g.savefig("/home/andrazzrimsek/DataM/WorldClim/Plots/variable_correlation_heatmap.png", dpi=300)

# Save the dendrogram order for variables (row order)
from scipy.cluster.hierarchy import leaves_list
row_leaves = leaves_list(g.dendrogram_row.linkage)
pd.DataFrame({'Variable': corr.index[row_leaves]}).to_csv("correlation_variable_order.csv", index=False)

# Create high correlation clusters (correlation > 0.9)
print("\n" + "="*50)
print("CREATING HIGH CORRELATION CLUSTERS (r > 0.9)")
print("="*50)

# Get absolute correlation matrix (to consider both positive and negative correlations)
abs_corr = corr.abs()

# Create a mask for correlations > 0.9 (excluding diagonal)
high_corr_mask = (abs_corr > 0.9) & (abs_corr < 1.0)

# Find variable pairs with high correlation
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if high_corr_mask.iloc[i, j]:
            var1, var2 = corr.columns[i], corr.columns[j]
            correlation = corr.iloc[i, j]
            high_corr_pairs.append((var1, var2, correlation))

print(f"\nFound {len(high_corr_pairs)} variable pairs with |correlation| > 0.9:")
for var1, var2, corr_val in high_corr_pairs:
    print(f"  {var1} - {var2}: {corr_val:.3f}")

# Create clusters using connected components approach
from collections import defaultdict

# Build a graph where edges connect highly correlated variables
graph = defaultdict(set)
for var1, var2, _ in high_corr_pairs:
    graph[var1].add(var2)
    graph[var2].add(var1)

# Find connected components (clusters)
visited = set()
high_corr_clusters = []

def dfs(node, cluster):
    if node in visited:
        return
    visited.add(node)
    cluster.add(node)
    for neighbor in graph[node]:
        dfs(neighbor, cluster)

for variable in corr.columns:
    if variable not in visited:
        cluster = set()
        dfs(variable, cluster)
        if len(cluster) > 1:  # Only keep clusters with more than 1 variable
            high_corr_clusters.append(cluster)

print(f"\nHigh correlation clusters (correlation > 0.9):")
print(f"Number of clusters: {len(high_corr_clusters)}")

for i, cluster in enumerate(high_corr_clusters):
    print(f"\nCluster {i+1} ({len(cluster)} variables):")
    for var in sorted(cluster):
        pretty_name = prettify_bio_id(var)
        print(f"  {var} - {pretty_name}")
    
    # Show correlations within this cluster
    cluster_vars = list(cluster)
    if len(cluster_vars) > 1:
        print("  Correlations within cluster:")
        for j in range(len(cluster_vars)):
            for k in range(j+1, len(cluster_vars)):
                var1, var2 = cluster_vars[j], cluster_vars[k]
                corr_val = corr.loc[var1, var2]
                print(f"    {var1} - {var2}: {corr_val:.3f}")

# Save high correlation clusters to CSV
high_corr_cluster_df = []
for i, cluster in enumerate(high_corr_clusters):
    for var in cluster:
        high_corr_cluster_df.append({
            'Variable': var,
            'High_Corr_Cluster': i,
            'Cluster_Size': len(cluster)
        })

if high_corr_cluster_df:
    high_corr_df = pd.DataFrame(high_corr_cluster_df)
    high_corr_df.to_csv("high_correlation_clusters.csv", index=False)
    print(f"\nHigh correlation clusters saved to 'high_correlation_clusters.csv'")
else:
    print("\nNo high correlation clusters found (no variables with correlation > 0.9)")

# Summary of clusters for combining variables
print(f"\nSummary for combining variables:")
print(f"Total variables: {len(corr.columns)}")
print(f"Variables in high correlation clusters: {sum(len(cluster) for cluster in high_corr_clusters)}")
print(f"Variables not in any high correlation cluster: {len(corr.columns) - sum(len(cluster) for cluster in high_corr_clusters)}")

# Create a list of all variables to use for analysis (combining clusters)
variables_for_analysis = []

# Add all high correlation clusters (all variables in each cluster will be combined)
for i, cluster in enumerate(high_corr_clusters):
    cluster_vars = sorted(list(cluster))
    variables_for_analysis.append({
        'Type': 'Cluster',
        'ID': f'HighCorr_Cluster_{i}',
        'Variables': cluster_vars,
        'Count': len(cluster_vars)
    })

# Add individual variables that are not in any high correlation cluster
clustered_vars_set = set()
for cluster in high_corr_clusters:
    clustered_vars_set.update(cluster)

individual_vars = [var for var in corr.columns if var not in clustered_vars_set]
for var in individual_vars:
    variables_for_analysis.append({
        'Type': 'Individual',
        'ID': var,
        'Variables': [var],
        'Count': 1
    })

print(f"\nFinal analysis units:")
print(f"High correlation clusters: {len(high_corr_clusters)}")
print(f"Individual variables: {len(individual_vars)}")
print(f"Total analysis units: {len(variables_for_analysis)}")

# Save the analysis structure
analysis_df = pd.DataFrame(variables_for_analysis)
analysis_df.to_csv("variable_analysis_structure.csv", index=False)
print(f"\nVariable analysis structure saved to 'variable_analysis_structure.csv'")

# Compare with Agglomerative Clustering approach
print("\n" + "="*60)
print("COMPARING WITH AGGLOMERATIVE CLUSTERING APPROACH")
print("="*60)

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

# Convert correlation to distance matrix
corr_distance = np.sqrt(2 * (1 - abs_corr))
distance_threshold = np.sqrt(2 * (1 - 0.9))  # Equivalent to 0.9 correlation

print(f"Distance threshold (equivalent to r=0.9): {distance_threshold:.4f}")

# Get clusters using agglomerative clustering with distance threshold
agg_cluster_labels = fcluster(corr_linkage, t=distance_threshold, criterion='distance')

# Create DataFrame for agglomerative clustering results
agg_clustered_vars = pd.DataFrame({
    'Variable': numeric_cols,
    'Agg_Cluster': agg_cluster_labels - 1  # Make 0-indexed
})

# Find multi-variable clusters only
agg_cluster_counts = agg_clustered_vars['Agg_Cluster'].value_counts()
agg_multi_clusters = agg_cluster_counts[agg_cluster_counts > 1].index.tolist()

print(f"\nAgglomerative clustering results:")
print(f"Number of multi-variable clusters: {len(agg_multi_clusters)}")
print(f"Total variables in clusters: {sum(agg_cluster_counts[agg_multi_clusters])}")

agg_high_corr_clusters = []
for cluster_id in sorted(agg_multi_clusters):
    cluster_vars = agg_clustered_vars[agg_clustered_vars['Agg_Cluster'] == cluster_id]['Variable'].tolist()
    agg_high_corr_clusters.append(set(cluster_vars))
    
    print(f"\nAgglomerative Cluster {cluster_id} ({len(cluster_vars)} variables):")
    for var in sorted(cluster_vars):
        pretty_name = prettify_bio_id(var)
        print(f"  {var} - {pretty_name}")
    
    # Show correlations within this cluster
    if len(cluster_vars) > 1:
        print("  Correlations within cluster:")
        min_corr = float('inf')
        max_corr = -float('inf')
        high_corr_count = 0
        total_pairs = 0
        
        for j in range(len(cluster_vars)):
            for k in range(j+1, len(cluster_vars)):
                var1, var2 = cluster_vars[j], cluster_vars[k]
                corr_val = abs_corr.loc[var1, var2]  # Use absolute correlation
                print(f"    {var1} - {var2}: {corr_val:.3f}")
                
                min_corr = min(min_corr, corr_val)
                max_corr = max(max_corr, corr_val)
                if corr_val > 0.9:
                    high_corr_count += 1
                total_pairs += 1
        
        print(f"  Cluster statistics:")
        print(f"    Min correlation: {min_corr:.3f}")
        print(f"    Max correlation: {max_corr:.3f}")
        print(f"    Pairs with |r| > 0.9: {high_corr_count}/{total_pairs}")

# Compare the two approaches
print("\n" + "="*60)
print("COMPARISON: CONNECTED COMPONENTS vs AGGLOMERATIVE CLUSTERING")
print("="*60)

print(f"\nConnected Components (|r| > 0.9 threshold):")
print(f"  Number of clusters: {len(high_corr_clusters)}")
print(f"  Variables in clusters: {sum(len(cluster) for cluster in high_corr_clusters)}")

print(f"\nAgglomerative Clustering (distance threshold):")
print(f"  Number of clusters: {len(agg_high_corr_clusters)}")
print(f"  Variables in clusters: {sum(len(cluster) for cluster in agg_high_corr_clusters)}")

# Find differences between approaches
def cluster_sets_to_frozensets(cluster_list):
    return [frozenset(cluster) for cluster in cluster_list]

connected_sets = cluster_sets_to_frozensets(high_corr_clusters)
agg_sets = cluster_sets_to_frozensets(agg_high_corr_clusters)

print(f"\nCluster comparison:")
print(f"Identical clusters: {len(set(connected_sets) & set(agg_sets))}")
print(f"Only in Connected Components: {len(set(connected_sets) - set(agg_sets))}")
print(f"Only in Agglomerative: {len(set(agg_sets) - set(connected_sets))}")

# Show differences in detail
if set(connected_sets) != set(agg_sets):
    print(f"\nDetailed differences:")
    
    # Clusters only in connected components
    only_connected = set(connected_sets) - set(agg_sets)
    if only_connected:
        print(f"\nClusters only found by Connected Components:")
        for i, cluster in enumerate(only_connected):
            print(f"  Cluster {i+1}: {sorted(list(cluster))}")
    
    # Clusters only in agglomerative
    only_agg = set(agg_sets) - set(connected_sets)
    if only_agg:
        print(f"\nClusters only found by Agglomerative Clustering:")
        for i, cluster in enumerate(only_agg):
            cluster_vars = sorted(list(cluster))
            print(f"  Cluster {i+1}: {cluster_vars}")
            
            # Check if this cluster has any pairs with correlation < 0.9
            low_corr_pairs = []
            for j in range(len(cluster_vars)):
                for k in range(j+1, len(cluster_vars)):
                    var1, var2 = cluster_vars[j], cluster_vars[k]
                    corr_val = abs_corr.loc[var1, var2]
                    if corr_val <= 0.9:
                        low_corr_pairs.append((var1, var2, corr_val))
            
            if low_corr_pairs:
                print(f"    Pairs with |r| ≤ 0.9:")
                for var1, var2, corr_val in low_corr_pairs:
                    print(f"      {var1} - {var2}: {corr_val:.3f}")
else:
    print(f"\n✅ Both methods produced identical clusters!")

# Save agglomerative clustering results
agg_cluster_df = []
for i, cluster in enumerate(agg_high_corr_clusters):
    for var in cluster:
        agg_cluster_df.append({
            'Variable': var,
            'Agg_Cluster': i,
            'Cluster_Size': len(cluster)
        })

if agg_cluster_df:
    agg_df = pd.DataFrame(agg_cluster_df)
    agg_df.to_csv("agglomerative_correlation_clusters.csv", index=False)
    print(f"\nAgglomerative clustering results saved to 'agglomerative_correlation_clusters.csv'")

# Create comparison summary
comparison_summary = {
    'Method': ['Connected Components', 'Agglomerative Clustering'],
    'Num_Clusters': [len(high_corr_clusters), len(agg_high_corr_clusters)],
    'Variables_in_Clusters': [
        sum(len(cluster) for cluster in high_corr_clusters),
        sum(len(cluster) for cluster in agg_high_corr_clusters)
    ],
    'Threshold_Type': ['Hard (|r| > 0.9)', 'Soft (distance-based)'],
    'Guarantees_High_Corr': ['Yes', 'No']
}

comparison_df = pd.DataFrame(comparison_summary)
comparison_df.to_csv("clustering_method_comparison.csv", index=False)
print(f"\nMethod comparison saved to 'clustering_method_comparison.csv'")