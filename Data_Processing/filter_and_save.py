import numpy as np
import pandas as pd
import os

csv_file = "Data/1001_snp.csv"

with open(csv_file) as f:
    genome_names = f.readline().strip().split(",")[3:]

# Read in the CSV file
dtype_dict = {"CHROM": np.int8, "POS": np.int64, "INDEL": bool}
dtype_dict.update({name: np.int8 for name in genome_names})

positions = np.empty((0, 3), dtype=np.int64)
snp_genotypes = np.empty((0, len(genome_names)), dtype=np.int8)

chunksize = 10 ** 6
with pd.read_csv(csv_file, header=0, dtype=dtype_dict, chunksize=chunksize) as reader:
    for chunk in reader:
        positions_chunk = chunk[["CHROM", "POS", "INDEL"]].to_numpy().astype(np.int64)
        positions = np.vstack((positions, positions_chunk))
        numpy_snps = chunk[genome_names].to_numpy().astype(np.int8)
        snp_genotypes = np.vstack((snp_genotypes, numpy_snps))

# Filter out SNPs with genotype missingness > 5%
missingness = np.sum(snp_genotypes == 3, axis=1) / snp_genotypes.shape[1]
filtered_snps = snp_genotypes[missingness <= 0.05]
positions = positions[missingness <= 0.05]

# Replace missing genotypes with the most frequent genotype
for i in range(filtered_snps.shape[0]):
    unique, counts = np.unique(filtered_snps[i][filtered_snps[i] != 3], return_counts=True)
    most_frequent_genotype = unique[np.argmax(counts)]
    filtered_snps[i][filtered_snps[i] == 3] = most_frequent_genotype

# Filter out SNPs with minor allele frequency < 5%
allele_freqs = np.sum(filtered_snps, axis=1) / (filtered_snps.shape[1] * 2)
filtered_snps_5 = filtered_snps[allele_freqs >= 0.05]
positions_5 = positions[allele_freqs >= 0.05]

# Filter out SNPs with minor allele frequency < 10%
filtered_snps_10 = filtered_snps[allele_freqs >= 0.10]
positions_10 = positions[allele_freqs >= 0.10]

# Filter out SNPs with minor allele frequency < 15%
filtered_snps_15 = filtered_snps[allele_freqs >= 0.15]
positions_15 = positions[allele_freqs >= 0.15]

# Save the filtered SNPs to a pickle file
# np.save("filtered_snps.npy", filtered_snps, allow_pickle=True)
# print("Filtered SNPs saved to 'filtered_snps.npy'")
# np.save("positions.npy", positions, allow_pickle=True)
# print("Positions saved to 'positions.npy'")

# Save individual names to a text file
# with open("individual_names.txt", "w") as f:
#     f.write("\n".join(genome_names))

# Save the filtered SNPs to a pickle file
np.save("Data/SNPs/filtered_snps_5.npy", filtered_snps_5, allow_pickle=True)
print("Filtered SNPs saved to 'filtered_snps_5.npy'")
np.save("Data/SNPs/positions_5.npy", positions_5, allow_pickle=True)
print("Positions saved to 'positions_5.npy'")

# Save the filtered SNPs to a pickle file
np.save("Data/SNPs/filtered_snps_10.npy", filtered_snps_10, allow_pickle=True)
print("Filtered SNPs saved to 'filtered_snps_10.npy'")
np.save("Data/SNPs/positions_10.npy", positions_10, allow_pickle=True)
print("Positions saved to 'positions_10.npy'")

# Save the filtered SNPs to a pickle file
np.save("Data/SNPs/filtered_snps_15.npy", filtered_snps_15, allow_pickle=True)
print("Filtered SNPs saved to 'filtered_snps_15.npy'")
np.save("Data/SNPs/positions_15.npy", positions_15, allow_pickle=True)
print("Positions saved to 'positions_15.npy'")
