# Author: Temi
# Description: Evaluate a trained XGBoost TF binding model on held-out test data.
#              Companion to train_xgboost.R.  Produces the same predictions output
#              format as evaluate_enpact.R (chrom, start, end, interval_id,
#              binding_class, log_odds, predicted_prob) so downstream scripts work
#              unchanged.
#
# Usage:
#   Rscript evaluate_xgboost.R \
#       --test_data_file   test.tsv.gz \
#       --model_rds_file   model.rds \
#       --predictions_file predictions.tsv.gz \
#       --metadata         ag_xgb_weight \
#       [--features_group  embeddings_128bp]
#
# Note: unlike evaluate_enpact.R this script loads the full xgb.Booster and calls
#       predict() rather than doing a linear dot-product with the weights file.

suppressPackageStartupMessages(library("optparse"))

option_list <- list(
    make_option("--test_data_file",   help = "Path to held-out test data (TSV.GZ)"),
    make_option("--model_rds_file",   help = "Path to xgb.Booster RDS from train_xgboost.R"),
    make_option("--predictions_file", help = "Output TSV.GZ for predictions"),
    make_option("--metadata",         help = "Model name tag written into output (informational)"),
    make_option("--features_group",   default = NULL,
                help = "Comma-separated feature-column prefix(es), e.g. embeddings_128bp")
)

opt <- parse_args(OptionParser(option_list = option_list))
print(opt)

suppressPackageStartupMessages({
    library(tidyverse)
    library(glue)
    library(data.table)
    library(xgboost)
    library(pROC)
})

metadata_columns <- c("chrom", "start", "end", "interval_id", "binding_class")

# ── Load test data ────────────────────────────────────────────────────────────
if (!file.exists(opt$test_data_file)) stop("ERROR — Test data not found.")
dt_test <- data.table::fread(opt$test_data_file)
dt_test <- dplyr::distinct(dt_test)
dt_test <- dt_test[complete.cases(dt_test), ]

# ── Build feature matrix ──────────────────────────────────────────────────────
if (!is.null(opt$features_group)) {
    groups <- strsplit(opt$features_group, ",")[[1]]
    print(glue("INFO — Using feature groups: {paste(groups, collapse=', ')}"))
    X_test <- dt_test %>%
        dplyr::select(!any_of(metadata_columns)) %>%
        dplyr::select(dplyr::starts_with(groups)) %>%
        as.matrix()
} else {
    X_test <- dt_test %>%
        dplyr::select(!any_of(metadata_columns)) %>%
        as.matrix()
}

y_test <- dt_test$binding_class
print(glue("INFO — Test matrix: {nrow(X_test)} rows × {ncol(X_test)} features"))

# ── Load model ────────────────────────────────────────────────────────────────
if (!file.exists(opt$model_rds_file)) stop("ERROR — Model RDS not found.")
model <- readRDS(opt$model_rds_file)
print(glue("INFO — Loaded model from {opt$model_rds_file}"))

# Align columns to model's expected features
model_features <- model$feature_names
if (!is.null(model_features)) {
    missing_in_test  <- setdiff(model_features, colnames(X_test))
    missing_in_model <- setdiff(colnames(X_test), model_features)
    if (length(missing_in_test) > 0) {
        print(glue("WARNING — {length(missing_in_test)} model features absent from test data; filling with 0"))
        extra <- matrix(0, nrow = nrow(X_test), ncol = length(missing_in_test),
                        dimnames = list(NULL, missing_in_test))
        X_test <- cbind(X_test, extra)
    }
    X_test <- X_test[, model_features, drop = FALSE]
}

dtest <- xgb.DMatrix(data = X_test)

# ── Predict ───────────────────────────────────────────────────────────────────
# predict() with binary:logistic returns probabilities in [0, 1]
predicted_prob <- predict(model, dtest)
# Convert to log-odds for output consistency with evaluate_enpact.R
log_odds <- log(predicted_prob / (1 - predicted_prob + 1e-9))

# ── Evaluate ──────────────────────────────────────────────────────────────────
roc_obj <- pROC::roc(response = y_test, predictor = predicted_prob, quiet = TRUE)
auroc   <- as.numeric(pROC::auc(roc_obj))
print(glue("INFO — AUROC = {round(auroc, 4)}  (model: {opt$metadata})"))

# ── Save predictions ──────────────────────────────────────────────────────────
output_dt <- dt_test %>%
    dplyr::select(any_of(metadata_columns)) %>%
    dplyr::mutate(
        log_odds       = log_odds,
        predicted_prob = predicted_prob
    )

print(glue("INFO — Saving predictions to {opt$predictions_file}"))
data.table::fwrite(output_dt, file = opt$predictions_file,
                   sep = "\t", row.names = FALSE, col.names = TRUE,
                   quote = FALSE, compress = "gzip")

print("INFO — Done.")
