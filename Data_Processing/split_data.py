from sklearn.cluster import KMeans
from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from collections import defaultdict
from time import time

from sklearn import metrics
from sklearn.cluster import SpectralClustering
from consensusClustering import ConsensusCluster

evaluations = []
evaluations_std = []


def fit_and_evaluate(km, X, name=None, n_runs=5, n_clusters=10):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    labels_runs = []
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=None)
        )
        labels_runs.append(km.labels_)

    train_times = np.asarray(train_times)
    labels_runs = np.asarray(labels_runs)

    # Build a co-occurrence matrix across runs
    co_occurrence = np.zeros((labels_runs.shape[1], labels_runs.shape[1]))
    for run_labels in labels_runs:
        for c in np.unique(run_labels):
            idx = np.where(run_labels == c)[0]
            co_occurrence[np.ix_(idx, idx)] += 1

    # Normalize by the number of runs
    co_occurrence /= labels_runs.shape[0]

    from sklearn.cluster import AgglomerativeClustering

    # Convert co-occurrence matrix to distance matrix for clustering
    # Points with co-occurrence > 0.9 (90%) should be in the same cluster
    distance_matrix = 1 - co_occurrence

    # If the number of clusters doesn't match desired count, adjust
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'  # Average linkage for more balanced clusters
    )
    consensus_labels = hierarchical.fit_predict(distance_matrix)
    
    # Assign consensus labels by grouping points that co-occur often
    # Use spectral clustering which is better suited for similarity matrices
    # spectral_consensus = SpectralClustering(n_clusters=n_clusters, random_state=42, 
    #                                        affinity='precomputed', n_init=100)
    # consensus_labels = spectral_consensus.fit_predict(co_occurrence)

    consensus_silhouette = metrics.silhouette_score(X, consensus_labels, sample_size=None)
    scores["Consensus Silhouette Coefficient"] = consensus_silhouette

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)

    return consensus_labels

def cluster_data(data, n_clusters=5, random_state=42):
    """
    Splits the data into training and testing sets based on clustering.

    Parameters:
    - data: DataFrame containing the data to be split.
    - n_clusters: Number of clusters to form.
    - random_state: Random state for reproducibility.

    Returns:
    - DataFrame with an additional column for cluster labels.
    """
    # Ensure the data is in a DataFrame format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    # Initialize KMeans clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)

    # cluster_labels = fit_and_evaluate(kmeans, data, name="KMeans\environmental variables", n_runs=100, n_clusters=n_clusters)

    data_np = data.values

    consensusCluster = ConsensusCluster(
        cluster=KMeans,
        L=n_clusters,
        K=n_clusters*2,
        H=100,
        resample_proportion=0.5
    )

    consensusCluster.fit(data_np, verbose=True)

    cluster_labels = consensusCluster.predict()
    
    # Fit the model and predict cluster labels
    # cluster_labels = kmeans.fit_predict(data)
    print("Silhouette Coefficient:", metrics.silhouette_score(data, cluster_labels, sample_size=None))

    # Add cluster labels to the original data
    data['cluster'] = cluster_labels

    return data

def split_data(data, counts, val_size=0.1*0.8, test_size=0.2):
    """
    Splits the clustered data into training, validation, and testing sets based on 
    clusters so that the final number of samples matches the splits and clusters 
    have proportional representation in each set.

    Parameters:
    - data: DataFrame containing the data to be split.
    - counts: Series with occurence counts for each unique sample.
    - val_size: Proportion of data to use for validation.
    - test_size: Proportion of data to use for testing.

    Returns:
    - train_data: DataFrame for training data.
    - val_data: DataFrame for validation data.
    - test_data: DataFrame for testing data.
    """
    # Ensure the data is in a DataFrame format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    # Ensure counts is a Series
    if not isinstance(counts, pd.Series):
        raise ValueError("Counts must be a pandas Series")

    # Calculate the number of samples for each set
    total_samples = len(data)
    val_count_total = 0
    test_count_total  = 0
    train_count_total  = 0

    val_count = 0
    test_count = 0
    train_count = 0

    # Initialize empty DataFrames for each set
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Split the data into training, validation, and testing sets based on clusters
    for cluster in sorted(data['cluster'].unique()):  # Sort clusters before iterating
        cluster_data = data[data['cluster'] == cluster]
        n_unique_samples = len(cluster_data['IID'].unique())
        skip_val = False
        skip_train = False
        
        # Shuffle the cluster data
        cluster_data = cluster_data.sample(frac=1, random_state=42).reset_index(drop=True)

        cluster_counts = counts.loc[cluster_data.set_index(['LONG', 'LAT']).index]
        # Replace LONG and LAT in cluster_counts with IID from cluster_data
        cluster_counts.index = cluster_data.index
        cluster_counts_iid = cluster_counts.copy()
        cluster_counts_iid.index = cluster_data['IID']


        # Repeat cluster data points based on counts and order them descending by counts
        cluster_data = cluster_data.loc[cluster_counts.index.repeat(cluster_counts)]
        cluster_data = cluster_data.reset_index(drop=True)

        total_samples = len(cluster_data)
        if n_unique_samples < 2:
            skip_val = True
            skip_train = True
        elif n_unique_samples < 3:
            skip_train = True

        # Calculate the number of samples for each set based on the cluster size
        val_count = max(int(val_size * len(cluster_data)), 1)
        test_count = max(int(test_size * len(cluster_data)), 1)
        train_count = len(cluster_data) - val_count - test_count

        test_iids = cluster_data.loc[:(test_count-1), 'IID'].unique()
        test_samples = cluster_data[cluster_data['IID'].isin(test_iids)]

        remaining_samples = cluster_data[~cluster_data['IID'].isin(test_iids)].reset_index(drop=True)

        if not skip_val:
            val_iids = remaining_samples.loc[:(val_count - 1), 'IID'].unique()
            val_samples = remaining_samples[remaining_samples['IID'].isin(val_iids)]

            remaining_samples = remaining_samples[~remaining_samples['IID'].isin(val_iids)].reset_index(drop=True)
        else:
            val_samples = pd.DataFrame()

        if not skip_train:
            train_iids = remaining_samples.loc[:, 'IID'].unique()
            train_samples = remaining_samples[remaining_samples['IID'].isin(train_iids)]
            remaining_samples = remaining_samples[~remaining_samples['IID'].isin(train_iids)].reset_index(drop=True)
        else:
            train_samples = pd.DataFrame()

        val_count_total += len(val_iids)
        test_count_total += len(test_iids)
        train_count_total += len(train_iids)
        print("Cluster:", cluster, "Total samples:", len(cluster_data), " / Train count:", len(train_samples), " / Validation count:", len(val_samples), " / Test count:", len(test_samples))
        # Concatenate the samples into the respective DataFrames

        train_data = pd.concat([train_data, train_samples])
        val_data = pd.concat([val_data, val_samples])
        test_data = pd.concat([test_data, test_samples])

    print("Final sample counts - Train:", len(train_data), "Validation:", len(val_data), "Test:", len(test_data))
    print("Final IID counts - Train:", train_count_total, "Validation:", val_count_total, "Test:", test_count_total)
    total_samples = len(train_data) + len(val_data) + len(test_data)
    print("Final proportion - Train:", len(train_data)/total_samples, "Validation:", len(val_data)/total_samples, "Test:", len(test_data)/total_samples)
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)

if __name__ == "__main__":
    data_path = "Data/long_lat.csv"
    bioclim_path = "Data/coords_with_bioclim_30s_fixed.csv"
    soil_path = "Data/coords_with_soil.csv"

    # if os.path.exists(data_path):
    #     # data = pd.read_csv(data_path, sep='\t')
    #     soil_data = pd.read_csv(soil_path, sep=',')
    #     bioclim_data = pd.read_csv(bioclim_path, sep=',')
    #     bioclim_data = pd.merge(bioclim_data, soil_data, on=['FID', 'IID', 'LONG', 'LAT'], how='inner')
    #     bioclim_data = bioclim_data[[c for c in bioclim_data.columns if 'uncertainty' not in c.lower()]]
    #     bioclim_data = bioclim_data.replace(-3.4e+38, np.nan).dropna()

    #     unique_samples = bioclim_data.drop(['IID', 'FID'], axis=1).drop_duplicates()
    #     unique_samples = unique_samples.merge(bioclim_data[['IID', 'LONG', 'LAT']].drop_duplicates(subset=['LONG', 'LAT']), on=['LONG', 'LAT'], how='right')
    #     unique_samples = unique_samples[['IID'] + [col for col in unique_samples.columns if col != 'IID']]
    #     unique_values = unique_samples.iloc[:, 3:]
    #     unique_count = bioclim_data.groupby(['LONG', 'LAT']).size()#.reset_index(name='count')
        
    #     print("Sampled data shape:", unique_values.shape)
    #     clustered_data = cluster_data(unique_values, n_clusters=25, random_state=42)
    #     print("Clustered data shape:", clustered_data.shape)
    #     clustered_data_with_coords = pd.concat([unique_samples[['IID','LONG', 'LAT']].reset_index(drop=True), clustered_data.reset_index(drop=True)], axis=1)
    #     print("Clustered data with coordinates shape:", clustered_data_with_coords.shape)
        
    #     train_data, val_data, test_data = split_data(clustered_data_with_coords, unique_count, val_size=0.1, test_size=0.2*0.9)

    #     # Replace duplicated IIDs in train, test, and val with matching LONG, LAT from bioclim_data
    #     train_coords = train_data[['LONG', 'LAT']].drop_duplicates()
    #     val_coords = val_data[['LONG', 'LAT']].drop_duplicates()
    #     test_coords = test_data[['LONG', 'LAT']].drop_duplicates()

    #     train_data = bioclim_data[bioclim_data.set_index(['LONG', 'LAT']).index.isin(train_coords.set_index(['LONG', 'LAT']).index)]
    #     val_data = bioclim_data[bioclim_data.set_index(['LONG', 'LAT']).index.isin(val_coords.set_index(['LONG', 'LAT']).index)]
    #     test_data = bioclim_data[bioclim_data.set_index(['LONG', 'LAT']).index.isin(test_coords.set_index(['LONG', 'LAT']).index)]

    #     # Save the IIDs for the splits to a file
    #     with open("Results/splits_all.json", "w") as f:
    #         json.dump({
    #             "train": train_data['IID'].unique().tolist(),
    #             "val": val_data['IID'].unique().tolist(),
    #             "test": test_data['IID'].unique().tolist()
    #         }, f, indent=4)
        
    #     # # # print("Testing data shape:", test_data.shape)

    #     # # # Plot the distribution of each bioclim variable for each split
    #     bioclim_columns = [col for col in bioclim_data.columns if col not in ['IID', 'FID', 'LONG', 'LAT']]

    #     for col in bioclim_columns:
    #         plt.figure(figsize=(10, 6))
    #         plt.hist(train_data[col], bins=30, alpha=0.5, label='Train', color='blue', density=True)
    #         plt.hist(val_data[col], bins=30, alpha=0.5, label='Validation', color='orange', density=True)
    #         plt.hist(test_data[col], bins=30, alpha=0.5, label='Test', color='green', density=True)
    #         plt.title(f'Distribution of {col}')
    #         plt.xlabel(col)
    #         plt.ylabel('Density')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(f"Results/plots_test/{col}_distribution.png")  # Save the plot
    #         plt.close()

        
    #     # Read the split IDs from the JSON file
    # else:
    #     print(f"Data file not found at {data_path}.")
    
    with open("Results/splits_all.json", "r") as f:
        splits = json.load(f)

    train_ids = splits.get("train", [])
    val_ids = splits.get("val", [])
    test_ids = splits.get("test", [])

    bioclim_data = pd.read_csv(bioclim_path, sep=',')
    bioclim_data = bioclim_data.replace(-3.4e+38, np.nan).dropna()

    train_data = bioclim_data[bioclim_data['IID'].isin(train_ids)]
    val_data = bioclim_data[bioclim_data['IID'].isin(val_ids)]
    test_data = bioclim_data[bioclim_data['IID'].isin(test_ids)]

    # bioclim_columns = [col for col in bioclim_data.columns if col not in ['IID', 'FID', 'LONG', 'LAT']]

    # for col in bioclim_columns:
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(train_data[col], bins=30, alpha=0.5, label='Train', color='blue', density=True)
    #     plt.hist(val_data[col], bins=30, alpha=0.5, label='Validation', color='orange', density=True)
    #     plt.hist(test_data[col], bins=30, alpha=0.5, label='Test', color='green', density=True)
    #     plt.title(f'Distribution of {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"Results/plotsNew/{col}_distribution.png")  # Save the plot
    #     plt.close()

    soil_data = pd.read_csv(soil_path, sep=',')
    soil_data = soil_data.replace(-3.4e+38, np.nan).dropna()

    train_soil = soil_data[soil_data['IID'].isin(train_ids)]
    val_soil = soil_data[soil_data['IID'].isin(val_ids)]
    test_soil = soil_data[soil_data['IID'].isin(test_ids)]

    # print("Train samples:", len(train_soil), "Validation samples:", len(val_soil), "Test samples:", len(test_soil))

    # # Plot the soil distribution for each split
    # soil_columns = [col for col in soil_data.columns if col not in ['IID', 'FID', 'LONG', 'LAT']]
    # # Remove columns that end with '_uncertainty' from soil_data
    # soil_columns = [col for col in soil_columns if not col.endswith('_uncertainty')]
    # for col in soil_columns:
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(train_soil[col], bins=30, alpha=0.5, label='Train', color='blue', density=True)
    #     plt.hist(val_soil[col], bins=30, alpha=0.5, label='Validation', color='orange', density=True)
    #     plt.hist(test_soil[col], bins=30, alpha=0.5, label='Test', color='green', density=True)
    #     plt.title(f'Distribution of {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"Results/plotsNew/{col}_soil_distribution.png")


    plot_columns = ['BIO11', 'BIO12', 'nitrogen_mean', 'wv0033_mean']

    # --- Merge bioclimatic and soil data for joint plotting ---
    # We merge per-split to avoid losing split membership; inner join keeps only samples present in both.
    # If you prefer retaining all bioclim samples and adding soil where available, change how='left'.
    merge_keys = ['FID', 'IID', 'LONG', 'LAT']
    missing_keys_bioclim = [k for k in merge_keys if k not in bioclim_data.columns]
    missing_keys_soil = [k for k in merge_keys if k not in soil_data.columns]
    if missing_keys_bioclim or missing_keys_soil:
        print(f"Warning: Missing merge keys. Bioclim missing: {missing_keys_bioclim}, Soil missing: {missing_keys_soil}. Skipping merge – using separate bioclim data only.")
        use_combined = False
    else:
        use_combined = True
        # Filter soil splits already created (train_soil / val_soil / test_soil)
        def safe_merge(left, right, name):
            if left.empty or right.empty:
                print(f"Split '{name}' empty after filtering – cannot merge.")
                return pd.DataFrame()
            merged = pd.merge(left, right, on=merge_keys, how='inner', suffixes=('', '_soil'))
            if merged.empty:
                print(f"Split '{name}' produced 0 rows after merge (intersection empty).")
            return merged

        train_combined = safe_merge(train_data, train_soil, 'Train')
        val_combined = safe_merge(val_data, val_soil, 'Validation')
        test_combined = safe_merge(test_data, test_soil, 'Test')

        # Remove columns with `_uncertainty` and duplicate columns if suffixes occurred
        def clean_columns(df):
            if df.empty:
                return df
            drop_cols = [c for c in df.columns if c.lower().endswith('uncertainty')]
            df = df.drop(columns=drop_cols, errors='ignore')
            # If duplicate base names due to suffixes, prefer non _soil columns for plotting list
            for pc in plot_columns:
                if pc not in df.columns:
                    # Try soil suffixed variant
                    soil_variant = pc + '_soil'
                    if soil_variant in df.columns:
                        df = df.rename(columns={soil_variant: pc})
            return df

        train_combined = clean_columns(train_combined)
        val_combined = clean_columns(val_combined)
        test_combined = clean_columns(test_combined)

        # If merged splits are too small, fallback
        min_rows_required = 5
        if any(df.shape[0] < min_rows_required for df in [train_combined, val_combined, test_combined]):
            print("Merged splits too small for at least one split – falling back to bioclim-only data for those.")
            if train_combined.shape[0] < min_rows_required:
                train_combined = train_data
            if val_combined.shape[0] < min_rows_required:
                val_combined = val_data
            if test_combined.shape[0] < min_rows_required:
                test_combined = test_data

    if use_combined:
        print("Using merged bioclim + soil data for plotting.")
    else:
        print("Using bioclim-only data for plotting (merge unavailable).")

    # --- Combined multi-panel figure for selected variables (A4-ready) ---
    # Create output directory if it doesn't exist
    combined_out_dir = "Plots/Report"
    os.makedirs(combined_out_dir, exist_ok=True)

    # Figure settings for A4 (landscape). A4 in inches: 11.69 x 8.27
    fig_size = (11.69, 8.27)  # landscape orientation
    n_cols = 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)

    # Global font sizes
    plt.rcParams.update({
        'font.size': 13,          # base font size
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    # Use combined splits if available; else fall back to original
    splits_data = {
        'Train': train_combined if use_combined else train_data,
        'Validation': val_combined if use_combined else val_data,
        'Test': test_combined if use_combined else test_data
    }
    colors = {
        'Train': 'blue',       # blue
        'Validation': 'orange',  # orange
        'Test': 'green'         # green
    }

    # Helper to plot a single variable
    def plot_distribution(ax, var_name):
        available = [name for name, df in splits_data.items() if var_name in df.columns]
        if len(available) == 0:
            ax.set_visible(False)
            return
        # Concatenate all values to define common bins
        all_vals = np.concatenate([splits_data[name][var_name].dropna().values for name in available])
        if all_vals.size == 0:
            ax.set_visible(False)
            return
        bins = 30
        # Histogram for each split
        for name in available:
            vals = splits_data[name][var_name].dropna().values
            if vals.size == 0:
                continue
            ax.hist(vals, bins=bins, density=True, alpha=0.5, label=name, color=colors.get(name, None), edgecolor='none')
        ax.set_title(var_name)
        # ax.set_xlabel(var_name)
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.6)
        # Only add legend if not already added
        ax.legend(frameon=False)

    for idx, var in enumerate(plot_columns):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        plot_distribution(ax, var)

    # If fewer than 4 variables present, hide empty axes
    total_slots = n_rows * n_cols
    if len(plot_columns) < total_slots:
        for j in range(len(plot_columns), total_slots):
            r = j // n_cols
            c = j % n_cols
            axes[r, c].set_visible(False)

    # Add figure title with extra padding at top
    fig.suptitle('Distributions of Selected Environmental / Soil Variables Across Splits', fontsize=18)
    # Slightly reduce the occupied area of subplots to create more space for the title
    plt.subplots_adjust(top=0.90)  # increase padding by lowering the top fraction

    combined_path = os.path.join(combined_out_dir, 'combined_selected_variables.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined multi-panel figure to: {combined_path}")

    


    