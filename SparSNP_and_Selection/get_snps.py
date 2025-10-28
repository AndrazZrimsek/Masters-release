import numpy as np
import pandas as pd
from collections import Counter
import os
import pickle

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


def get_top_regions(top_snps, window_size=1000, top_n=100):
    """
    For each ID, calculate the top n regions (where a region is defined as (chrom, pos // window_size))
    and print the top regions by count.
    """
    # top_regions_ids = {}
    all_regions = []
    all_regions_per_id = {}

    # Process each ID
    for id in top_snps.keys():
        # Get SNPs for this ID
        snps = top_snps[id]
        
        # Calculate regions for all SNPs
        regions = []
        for snp in snps:
            try:
                # Extract the chromosome and position from the SNP
                chrom = int(snp.split("_")[0])
                pos = int(snp.split("_")[1])
                region = (chrom, pos // window_size)
                regions.append(region)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse SNP {snp}")
                continue
        
        all_regions.extend(list(set(regions)))
        all_regions_per_id[id] = list(set(regions))

        # all_regions.extend(regions)
        # all_regions_per_id[id] = regions
        # Count occurrences of each region
        # region_counts = Counter(regions)
        
        # # Print the top 10 regions by count
        # for i, (region, count) in enumerate(region_counts.most_common(top_n), 1):
        #     chrom, region_num = region
        #     print(f"{i}. Chromosome {chrom}, Region {region_num}: {count} SNPs")
        #     if region not in top_regions:
        #         top_regions_ids[region] = [id] 
        #     else:
        #         top_regions_ids[region].append(id)

        #     top_regions.append(region)

    all_regions_counter = Counter(all_regions)
    
    # Group regions by their count (how many variables they belong to)
    import random
    from collections import defaultdict
    
    regions_by_count = defaultdict(list)
    for region, count in all_regions_counter.items():
        regions_by_count[count].append(region)
    
    # Randomize regions within each count level to avoid variable clustering
    for count in regions_by_count:
        random.shuffle(regions_by_count[count])
    
    # Combine regions: highest count first, but randomized within each count level
    ordered_regions = []
    for count in sorted(regions_by_count.keys(), reverse=True):
        ordered_regions.extend(regions_by_count[count])
    
    top_regions_all = []
    # Print the top regions across all IDs
    print("\nTop regions across all variables:")
    for i, region in enumerate(ordered_regions[:top_n], 1):
        chrom, region_num = region
        count = all_regions_counter[region]
        # Find variables (IDs) that have this region
        variables_in_region = [var for var, regions in all_regions_per_id.items() if region in regions]
        print(f"{i}. Chromosome {chrom}, Region {region_num}: {count} variables ({', '.join(variables_in_region)})")
        top_regions_all.append(region)

    # Print the number of top regions for each ID
    print("\nNumber of top regions for each variable:")
    for variable, regions in all_regions_per_id.items():
        print(f"{variable}: {len([region for region in regions if region in top_regions_all ])} regions")

    return top_regions_all

dir = "JointVariables"

top_snps = {}
top_count = {}
print(f"Current working directory: {os.getcwd()}")
snp_count_file = f"Datasets/{dir}/best_model_snps_all.txt"
# os.chdir("WorldClim")

with open(snp_count_file, "r") as f:
    for line in f:
        line = line.strip()
        # if not line.startswith("BIO"):
        #     continue
        id = line.split()[0][:-1]
        count = int(line.split()[1])
        top_count[id] = count

# # Get top SNPs for each model
for id in ["BIO_11_6"]: #top_count.keys():
    top_snps_marker = pd.read_csv(f"Datasets/{dir}/discovery/" + id + "/topsnps.txt" , sep=" ")
    num_snps = top_count[id]
    top_snps_marker = top_snps_marker.iloc[:288]
    top_snps[id] = top_snps_marker['RS'].to_list()

# Create a set of all SNPs across all IDs
all_snps_set = set()
for id, snps in top_snps.items():
    all_snps_set.update(snps)

print(f"Total number of unique SNPs across all variables: {len(all_snps_set)}")

with open("all_snps.txt", "w") as f:
    for snp in all_snps_set:
        f.write(f"{snp}\n")
window_size = 1
# top_regions = get_top_regions(top_snps, window_size=window_size, top_n=100)

# with open("top_regions.txt", "w") as f:
#     for region in top_regions:
#         f.write(f"{region[0]}_{region[1]}\n")


# # Get every 50th region from the ordered top regions
top_regions_all = get_top_regions(top_snps, window_size=window_size, top_n=len(all_snps_set))
# top_regions_filtered = set(top_regions_all[i] for i in range(1, len(top_regions_all), 50))
# Save the top regions to a file
# with open("top_regions_50.txt", "w") as f:
#     for region in top_regions_filtered:
#         f.write(f"{region[0]}_{region[1]}\n")

with open("top_snps_all_11_6.pkl", "wb") as f:
    pickle.dump(top_regions_all, f)


# with open("top_snps_100.pkl", "wb") as f:
#     pickle.dump(top_regions, f)

# # all_snps = []
# # for id in top_snps.keys():
# #     all_snps += list(set(np.array(top_snps[id])))

# from sklearn.cluster import FeatureAgglomeration
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Add function to check if a SNP is in top regions
# def snp_in_top_regions(snp, top_regions, window_size=1):
#     try:
#         chrom = int(snp.split("_")[0])
#         pos = int(snp.split("_")[1])
#         region = (chrom, pos // window_size)
#         return region in top_regions
#     except (ValueError, IndexError):
#         print(f"Warning: Could not parse SNP {snp}")
#         return False

# # Parameter to control whether to use only SNPs from top regions
# use_top_regions_only = True  # Set to True to use only SNPs from top regions

# # Create a list of all unique SNPs, filtered by top regions if needed
# if use_top_regions_only:
#     all_unique_snps = [snp for snp in all_snps_set if snp_in_top_regions(snp, top_regions, window_size)]
#     print(f"Using {len(all_unique_snps)} SNPs from top regions (out of {len(all_snps_set)} total)")
# else:
#     all_unique_snps = list(all_snps_set)

# # Create a matrix of IDs vs SNPs (binary values)
# ids = list(top_snps.keys())
# n_ids = len(ids)
# n_snps = len(all_unique_snps)

# print(f"\nCreating a matrix with {n_ids} IDs and {n_snps} SNPs")

# # Create a mapping from SNP to column index
# snp_to_col = {snp: i for i, snp in enumerate(all_unique_snps)}

# # Create the matrix
# X = np.zeros((n_ids, n_snps))

# # Fill in the matrix with 1s where an ID has the SNP
# # for i, id in enumerate(ids):
# #     for snp in top_snps[id]:
# #         if snp in snp_to_col:
# #             X[i, snp_to_col[snp]] = 1

# # Fill in the matrix with actual weights from SparSNP
# for i, id in enumerate(ids):
#     # Read the weights file for this ID
#     weights_file = f"Datasets/{dir}/discovery/{id}/avg_weights_opt.score"
    
#     if os.path.exists(weights_file):
#         try:
#             # Read the weights file - format is: SNP_ID ALLELE WEIGHT
#             weights_df = pd.read_csv(weights_file, sep=r'\s+', header=None)
#             # Extract weights
#             weights = []
#             snp_ids = []
            
#             for _, row in weights_df.iterrows():
#                 snp = row[0]  # SNP ID is in first column
#                 weight = row[2]  # Weight is in the last column
#                 if snp in snp_to_col:
#                     weights.append(abs(weight))
#                     snp_ids.append(snp)
            
#             # Normalize weights if there are any valid weights
#             if weights:
#                 # Use L2 normalization (Euclidean norm)
#                 norm = np.linalg.norm(weights)
#                 if norm > 0:
#                     normalized_weights = np.array(weights) / norm
                    
#                     # Fill the matrix with normalized weights
#                     for j, snp in enumerate(snp_ids):
#                         X[i, snp_to_col[snp]] = normalized_weights[j]
#         except Exception as e:
#             print(f"Error reading weights file for {id}: {e}")
#     else:
#         print(f"Warning: Weights file for {id} not found: {weights_file}")

# # Create a DataFrame for visualization
# df_snps = pd.DataFrame(
#     X,
#     index=ids,
#     columns=all_unique_snps
# )

# # Replace variable IDs with their prettified names for plotting only
# ids_prettified = [prettify_bio_id(id) for id in ids]
# df_snps_pretty = df_snps.copy()
# df_snps_pretty.index = ids_prettified

# # For all saving/cluster assignment, use original variable names (ids)

# # Create output directory if it doesn't exist
# output_dir = "Plots/Report/Clustering"
# os.makedirs(output_dir, exist_ok=True)

# # Create the clustermap with weights instead of binary values
# print("\nCreating SNP clustermap with actual weights...")
# plt.figure(figsize=(14, 16))

# # Calculate the absolute maximum for proper color scaling
# vmax = np.max(np.abs(X))
# vmin = -vmax  # For symmetric colormap


# from scipy.cluster.hierarchy import linkage

# # Compute linkage with optimal ordering for rows and columns
# row_linkage = linkage(df_snps.values, method='ward', optimal_ordering=True)
# col_linkage = linkage(df_snps.values.T, method='ward', optimal_ordering=True)

# # Assign cluster labels to variables (IDs)
# from scipy.cluster.hierarchy import leaves_list, fcluster
# row_leaves = leaves_list(row_linkage)
# snp_cluster_labels = fcluster(row_linkage, t=1.75, criterion='distance') - 1
# snps_clustered_vars = pd.DataFrame({
#     'Variable': df_snps.index,
#     'SNP_Cluster': snp_cluster_labels
# })
# snps_clustered_vars.to_csv("snps_variable_clusters.csv", index=False)

# # Add cluster color bars to the clustermap for variables (rows)
# # Load correlation-based clusters to match the correlation clustermap
# try:
#     corr_clusters = pd.read_csv("correlation_variable_clusters.csv")
#     print("Loaded correlation clusters for color coding")
    
#     # Create color mapping for correlation clusters
#     n_corr_clusters = len(corr_clusters['Corr_Cluster'].unique())
#     corr_cluster_palette = sns.color_palette('tab20', n_corr_clusters)
#     corr_cluster_color_map = {i: corr_cluster_palette[i] for i in range(n_corr_clusters)}
    
#     # Create a simple mapping from variable to cluster
#     corr_cluster_dict = dict(zip(corr_clusters['Variable'], corr_clusters['Corr_Cluster']))
    
#     # Map each variable in df_snps to its color
#     corr_row_colors = []
#     for var in df_snps.index:
#         if var in corr_cluster_dict:
#             cluster_id = corr_cluster_dict[var]
#             color = corr_cluster_color_map[cluster_id]
#         else:
#             # Default to first color if variable not found
#             color = corr_cluster_color_map[0]
#             print(f"Warning: Variable {var} not found in correlation clusters, using default color")
#         corr_row_colors.append(color)
    
#     print(f"Using correlation clusters with {n_corr_clusters} clusters for color coding")
#     print(f"Mapped {len(corr_row_colors)} variables to colors")
    
# except FileNotFoundError:
#     print("Warning: correlation_variable_clusters.csv not found, using SNP-based clusters")
#     # Fallback to SNP-based clusters
#     n_clusters = len(np.unique(snp_cluster_labels))
#     cluster_palette = sns.color_palette('tab10', n_clusters)
#     cluster_color_map = {i: cluster_palette[i] for i in range(n_clusters)}
#     corr_row_colors = [cluster_color_map[cluster_id] for cluster_id in snp_cluster_labels]

# # Recreate the clustermap with row_colors
# clustermap = sns.clustermap(
#     df_snps_pretty,
#     row_linkage=row_linkage,
#     col_linkage=col_linkage,
#     figsize=(14, 10),
#     cmap='RdBu_r',
#     center=0,
#     vmin=vmin,
#     vmax=vmax,
#     yticklabels=True,
#     xticklabels=False,
#     col_cluster=True,
#     row_cluster=True,
#     dendrogram_ratio=(0.1, 0.1),
#     tree_kws={"linewidth": 0.5, "colors": "#000000"},
#     cbar_pos=(0.02, 0.8, 0.01, 0.15),  # [x, y, width, height] - narrow width (0.02)
#     row_colors=corr_row_colors
# )

# # # Remove the row dendrogram like in the correlation plot
# # clustermap.ax_row_dendrogram.remove()

# # Immediately reposition the row colors to the right side (same as correlation plot)
# heatmap_pos = clustermap.ax_heatmap.get_position()

# # Position row colors on the right side of the heatmap
# new_row_colors_pos = [heatmap_pos.x1, heatmap_pos.y0, 0.028, heatmap_pos.height]
# clustermap.ax_row_colors.set_position(new_row_colors_pos)

# # Move the y-axis ticks and labels to the right side
# clustermap.ax_heatmap.yaxis.tick_right()
# clustermap.ax_heatmap.yaxis.set_label_position("right")

# # Add padding to move the tick labels further right to avoid overlap with color bar
# # clustermap.ax_heatmap.tick_params(axis='y', which='major', pad=30, length=0, direction='out')

# # Remove the axis labels and ticks from the color bar
# clustermap.ax_row_colors.set_xlabel('')
# clustermap.ax_row_colors.set_ylabel('')
# clustermap.ax_row_colors.set_xticks([])
# clustermap.ax_row_colors.set_yticks([])
# clustermap.ax_row_colors.axis('off')

# # --- Manually adjust colorbar width (seaborn may ignore cbar_pos width tweaks) ---
# # Force a draw so seaborn finalizes positions, then shrink the colorbar thickness.
# fig = clustermap.fig
# fig.canvas.draw()
# if hasattr(clustermap, 'cax') and clustermap.cax is not None:
#     cbar_ax = clustermap.cax
#     pos = cbar_ax.get_position()
#     # Set desired absolute width in figure fraction (tweak this value as needed)
#     desired_width = 0.006  # decrease for thinner, increase for thicker
#     # Keep the right edge fixed (or left edge if bar is on left). Currently it's on the left, so anchor x0.
#     new_pos = [pos.x0, pos.y0, desired_width, pos.height]
#     cbar_ax.set_position(new_pos)
#     # Optionally shrink label/ticks if they collide
#     for tick in cbar_ax.get_yticklabels():
#         tick.set_fontsize(8)
#     cbar_ax.yaxis.offsetText.set_fontsize(8)
# else:
#     print("Warning: Colorbar axis not found for manual resizing.")

# # Save SNP-based cluster assignments for later comparison
# from scipy.cluster.hierarchy import leaves_list, fcluster
# row_leaves = leaves_list(row_linkage)
# snp_cluster_labels = fcluster(row_linkage, t=1.75, criterion='distance') - 1
# snps_clustered_vars = pd.DataFrame({
#     'Variable': df_snps.index,
#     'SNP_Cluster': snp_cluster_labels
# })
# snps_clustered_vars.to_csv("snps_variable_clusters.csv", index=False)

# # Save the SNP weight matrix (variables x variables, for comparison)
# df_snps.to_csv("snp_weight_matrix.csv")

# # Save the dendrogram order for variables (row order)
# pd.DataFrame({'Variable': df_snps.index[row_leaves]}).to_csv("snps_variable_order.csv", index=False)

# # Adjust figure appearance
# clustermap.ax_heatmap.xaxis.tick_top()
# clustermap.ax_heatmap.xaxis.set_label_position('top')
# plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90, ha='center')

# # Add vertical lines between columns
# # for i in range(1, len(all_unique_snps)):
# #     clustermap.ax_heatmap.axvline(x=i-0.5, color='black', linewidth=0.5, alpha=0.3)

# # Change font color for y-axis labels
# for label in clustermap.ax_heatmap.get_yticklabels():
#     text = label.get_text()
#     if text in soil_name_map.values():
#         label.set_color("#B8860B")  # dark goldenrod for soil
#     else:
#         label.set_color("#205375")  # dark blue for bioclimatic

# # Add a title
# plt.suptitle('Clustered Variables vs SNP Weights from SparSNP', fontsize=16, y=0.95)

# # Add padding at the top for the title instead of tight_layout
# plt.subplots_adjust(top=0.92)

# # Save the visualization
# plt.savefig(f'{output_dir}/{dir}_abs_weights_100' + ('_top_regions' if use_top_regions_only else '') + '.png', 
#             bbox_inches='tight', dpi=300)
# # print(f"Clustermap visualization with weights saved as 'snp_variable_weights_clustermap{('_top_regions' if use_top_regions_only else '')}.png'")

