# Author: Temi
# Date: Wednesday May 21 2025
# Description: script to evaluate enpact models on held-out test data
# Usage: Rscript evaluate_enpact.R [options]

suppressPackageStartupMessages(library("optparse"))

option_list <- list(
    make_option("--test_data_file", help='path to held-out test data'),
    make_option("--weights_file", help='path to weights file from train_enpact.R'),
    make_option("--predictions_file", help='output tsv.gz file for predictions'),
    make_option("--metadata", help='model name; must match column name in weights file'),
    make_option("--features_group", help='CHIP_TF,DNASE', default = NULL)
)

opt <- parse_args(OptionParser(option_list=option_list))

print(opt)

library(tidyverse)
library(glue)
library(data.table)
library(pROC)

metadata_columns <- c('chrom', 'start', 'end', 'interval_id', 'binding_class')

if(file.exists(opt$test_data_file)){
    dt_test <- data.table::fread(opt$test_data_file)
} else {
    stop('ERROR - Test data cannot be found.')
}

dt_test <- dplyr::distinct(dt_test)
cc <- complete.cases(dt_test)
dt_test <- dt_test[cc, ]

if(file.exists(opt$weights_file)){
    weights_dt <- data.table::fread(opt$weights_file)
} else {
    stop('ERROR - Weights file cannot be found.')
}

if(!is.null(opt$features_group)){
    features_group <- strsplit(opt$features_group, ',')[[1]]
    print(glue("INFO - Evaluating using these groups of features: {paste(features_group, collapse=', ')}"))
    X_test <- dt_test %>%
        dplyr::select(!any_of(metadata_columns)) %>%
        dplyr::select(dplyr::starts_with(features_group)) %>%
        as.matrix()
} else {
    X_test <- dt_test %>% dplyr::select(!any_of(metadata_columns)) %>% as.matrix()
}

y_test <- dt_test$binding_class

print(glue('INFO - Read test data, number of features = {ncol(X_test)}, number of samples = {nrow(X_test)}'))

# separate intercept from feature weights
intercept_val <- weights_dt[feature == 'intercept', get(opt$metadata)]
feature_weights <- weights_dt[feature != 'intercept']

# align features between weights and test data
common_features <- intersect(colnames(X_test), feature_weights$feature)
print(glue('INFO - {length(common_features)} features in common between test data and weights'))

if(length(common_features) == 0){
    stop('ERROR - No features in common between test data and weights.')
}

w <- feature_weights[match(common_features, feature), get(opt$metadata)]
X_aligned <- X_test[, common_features, drop = FALSE]

# compute predictions
log_odds <- as.vector(X_aligned %*% w) + intercept_val
predicted_prob <- 1 / (1 + exp(-log_odds))

# evaluate
roc_obj <- pROC::roc(response = y_test, predictor = predicted_prob, quiet = TRUE)
auroc <- as.numeric(pROC::auc(roc_obj))
print(glue('INFO - AUROC = {round(auroc, 4)}'))

# save predictions with metadata
output_dt <- dt_test %>%
    dplyr::select(any_of(metadata_columns)) %>%
    dplyr::mutate(log_odds = log_odds, predicted_prob = predicted_prob)

print(glue('INFO - Saving predictions to {opt$predictions_file}'))
data.table::fwrite(output_dt, file=opt$predictions_file, sep='\t', row.names=F, col.names=T, quote=F, compress='gzip')

print(glue('INFO - Done'))
