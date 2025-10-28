setwd("/home/andrazzrimsek/DataM/WorldClim/Datasets/JointVariables/")

# Define the variables to match the directories created in the Bash script
variables <- c('BIO_4_7', 'BIO_10_5', 'BIO_11_6', 'BIO_12_13_16', 'BIO_14_17', 
               'Soil_Nitrogen_SOC', 'BIO1', 'BIO2', 'BIO3', 'BIO8', 'BIO9', 
               'BIO15', 'BIO18', 'BIO19', 'clay_mean', 'bdod_mean', 
               'wv0033_mean', 'phh2o_mean')
# variables <- c('BIO1')

results_table <- data.frame(BIO_ID = character(), R2_Coefficient = numeric(), Spearman_Coefficient = numeric())
file_content <- readLines("val/plink_BIO1.fam")
# print(file_content)

val_ids <- sapply(strsplit(file_content, "\\s+"), function(x) x[1])
val_ids <- val_ids[val_ids != ""]  # Remove empty entries
# print("Parsed val_ids:")
print(length(val_ids))

for (id in variables) {
    folder_name <- paste0("predict/", id, "/results.RData")
    
    if (file.exists(folder_name)) {
        load(folder_name)
        predictions <- lapply(res, function(x) x$pred)
        pred_matrix <- do.call(cbind, predictions)

        # Print the prediction matrix
        cat("Prediction matrix for", id, ":\n")
        print(dim(pred_matrix))
        cat("Dimensions:", nrow(pred_matrix), "rows x", ncol(pred_matrix), "columns\n")
        consensus_pred <- rowMeans(pred_matrix, na.rm = TRUE)
        # Create the prediction directory if it doesn't exist
        dir.create("prediction", showWarnings = FALSE)
        # Save consensus predictions to CSV file
        consensus_file <- paste0("prediction/consensus_pred_", id, ".csv")
        write.csv(data.frame(prediction = consensus_pred), file = consensus_file, row.names = FALSE)
        cat("Consensus predictions saved to", consensus_file, "\n")
        observed <- res[[1]]$observed
        
        # Calculate R^2 as correlation squared
        R2_consensus <- cor(consensus_pred, observed)^2
        
        # Calculate Spearman correlation coefficient
        spearman_consensus <- cor(consensus_pred, observed, method = "spearman")

        results_table <- rbind(results_table, data.frame(BIO_ID = id, R2_Coefficient = R2_consensus, Spearman_Coefficient = spearman_consensus))
        # Calculate prediction errors
        prediction_errors <- consensus_pred - observed

        # Check if the number of IDs matches the number of predictions
        if(length(val_ids) == length(prediction_errors)) {
            # Create a data frame with IDs and errors
            error_df <- data.frame(ID = val_ids, Error = prediction_errors)
            
            # Save error data to CSV file
            error_file <- paste0("prediction/prediction_errors_", id, ".csv")
            write.csv(error_df, file = error_file, row.names = FALSE)
            cat("Prediction errors saved to", error_file, "\n")
        } else {
            warning(paste("Number of IDs (", length(val_ids), ") doesn't match number of predictions (", 
                                        length(prediction_errors), ") for", id))
        }
    } else {
        warning(paste("File not found:", folder_name))
    }
}

# Sort results table by R2_Coefficient in descending order
results_table <- results_table[order(results_table$R2_Coefficient, decreasing = TRUE), ]

# Write the sorted table to file
write.table(results_table, file = "prediction/consensus_results.txt", row.names = FALSE, col.names = TRUE, quote = FALSE)
