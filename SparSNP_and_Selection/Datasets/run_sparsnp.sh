#!/bin/bash

# This script runs SparSNP on a set of VCF files.
# Define the variables to process - updated to match combined_variables_dataset.csv
# Combined variables: BIO_4_7, BIO_10_5, BIO_11_6, BIO_12_13_16, BIO_14_17, Soil_Nitrogen_SOC
# Individual variables: BIO1, BIO2, BIO3, BIO8, BIO9, BIO15, BIO18, BIO19, clay_mean, bdod_mean, wv0033_mean, phh2o_mean
variables=('BIO_4_7' 'BIO_10_5' 'BIO_11_6' 'BIO_12_13_16' 'BIO_14_17' 'Soil_Nitrogen_SOC' 'BIO1' 'BIO2' 'BIO3' 'BIO8' 'BIO9' 'BIO15' 'BIO18' 'BIO19' 'clay_mean' 'bdod_mean' 'wv0033_mean' 'phh2o_mean') 
# Check if dataset argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset="$1"
# variables=('phh2o_mean')
# Process each variable
for id in "${variables[@]}"; do
    echo "Processing ID: $id"
    
    plink_file="$dataset/train/plink_$id"
    test_file="$dataset/test/plink_$id"

    mkdir -p $dataset/discovery/$id
    mkdir -p $dataset/predict/$id

    NUMPROCS=5 DIR="$dataset/discovery/${id}" crossval.sh ${plink_file} linear

    best_model_snps=$(eval.R dir="$dataset/discovery/$id" | grep "Best model at" | awk '{print $4}')
    echo "Best model SNPs: $best_model_snps"

    # Append best model SNPs with ID info to a single file
    echo "$id: $best_model_snps" >> $dataset/best_model_snps_all.txt

    DIR="$dataset/discovery/$id"
    # echo "$best_model_snps"
    getmodels.R nzreq=$best_model_snps dir=$DIR

    OUTDIR="$dataset/predict/$id" DIR="$dataset/discovery/$id" predict.sh ${test_file}

    evalprofile.R model=linear indir="$dataset/discovery/$id" outdir="$dataset/predict/$id"

done

# get_results.R 