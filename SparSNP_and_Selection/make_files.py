import json
import os
import pandas as pd

bioclim_name = "coords_with_bioclim_30s_fixed.csv"
soil_name = "coords_with_soil.csv"
joint_name = "combined_variables_dataset_normalized.csv"
plink_file = "plink"
plink_dir = "Plink"
out_dir = "Datasets/JointVariables"

bioclim = pd.read_csv(bioclim_name)
soil = pd.read_csv(soil_name)

# joint = pd.merge(bioclim, soil, on=['IID', 'FID', 'LONG', 'LAT'], how='inner')
joint = pd.read_csv(joint_name)


for split in ['train', 'val', 'test']:
    os.makedirs(f"{out_dir}/{split}", exist_ok=True)
    fam_file = f"{plink_dir}/{plink_file}_{split}.fam"
    for id in [col for col in joint.columns if col not in ['IID', 'FID', 'LONG', 'LAT'] and not col.endswith('uncertainty')]:
        print(id)
        out_fam = f"{out_dir}/{split}/{plink_file}_{id}.fam"
        fam_data = pd.read_csv(fam_file, delim_whitespace=True, header=None)

        fam_data.iloc[:, -1] = fam_data.iloc[:, 1].map(joint.set_index('IID')[id])
        fam_data.to_csv(out_fam, sep=" ", header=False, index=False)

        for ext in ['bed', 'bim']:
            src_file = f"{plink_dir}/{plink_file}_{split}.{ext}"
            dest_file = f"{out_dir}/{split}/{plink_file}_{id}.{ext}"
            os.system(f"cp {src_file} {dest_file}")


