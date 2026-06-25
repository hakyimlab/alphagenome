# Author: Temi
# Description: Train an XGBoost model for TF binding prediction (Enpact-style).
#              Accepts the same CLI flags as train_enpact.R so it can be dropped
#              into the same Snakemake pipeline.  Uses xgb.cv to choose nrounds
#              via early stopping, then trains a final model on all training data.
#
# Usage:
#   Rscript train_xgboost.R \
#       --train_data_file  train.tsv.gz \
#       --weights_file     weights.tsv.gz \
#       --model_rds_file   model.rds \
#       --nfolds           5 \
#       --ncores           12 \
#       --metadata         ag_xgb_weight \
#       [--features_group  embeddings_128bp]
#
# Output
# ------
# model_rds_file  : the xgb.Booster object (use predict() for new data)
# weights_file    : feature importance (gain) — same column layout as
#                   train_enpact.R output so the same tooling can read it.
#                   NOTE: evaluation must use predict(model, X), not a linear
#                   dot-product.  Use evaluate_xgboost.R instead of
#                   evaluate_enpact.R.

suppressPackageStartupMessages(library("optparse"))

option_list <- list(
    make_option("--train_data_file", help = "Training data (TSV.GZ)"),
    make_option("--weights_file",    help = "Output: feature importance TSV.GZ"),
    make_option("--model_rds_file",  help = "Output: trained xgb.Booster RDS"),
    make_option("--nfolds",    type = "integer", default = 5L,
                help = "CV folds for early stopping [default: 5]"),
    make_option("--ncores",    type = "integer", default = 12L,
                help = "Threads for XGBoost [default: 12]"),
    make_option("--metadata",  help = "Column name written to weights file"),
    make_option("--features_group", default = NULL,
                help = "Comma-separated feature-column prefix(es), e.g. embeddings_128bp"),
    make_option("--tuning_results_file", default = NULL,
                help = "TSV.GZ of hyperparameter tuning results; best row (highest cv_auc_mean) overrides default params")
)

opt <- parse_args(OptionParser(option_list = option_list))
print(opt)

# opt <- list()
# opt$train_data_file <- "/scratch/midway2/temi/alphagenome/predictions.Enpact.AR_Prostate/reference_middle_embeddings_AR_Prostate/training/training.embedding_128bp.Enpact.AR_Prostate.tsv.gz"

suppressPackageStartupMessages({
    library(tidyverse)
    library(glue)
    library(data.table)
    library(xgboost)
})

metadata_columns <- c("chrom", "start", "end", "interval_id", "binding_class")

# ── Load training data ────────────────────────────────────────────────────────
if (!file.exists(opt$train_data_file)) stop("ERROR — Training data not found.")
dt_train <- data.table::fread(opt$train_data_file)
dt_train <- dplyr::distinct(dt_train)
dt_train <- dt_train[complete.cases(dt_train), ]

# ── Build feature matrix ──────────────────────────────────────────────────────
if (!is.null(opt$features_group)) {
    groups <- strsplit(opt$features_group, ",")[[1]]
    print(glue("INFO — Using feature groups: {paste(groups, collapse=', ')}"))
    X_train <- dt_train %>%
        dplyr::select(!any_of(metadata_columns)) %>%
        dplyr::select(dplyr::starts_with(groups)) %>%
        as.matrix()
} else {
    X_train <- dt_train %>%
        dplyr::select(!any_of(metadata_columns)) %>%
        as.matrix()
}

y_train <- dt_train$binding_class
print(glue("INFO — Training matrix: {nrow(X_train)} rows × {ncol(X_train)} features"))
print(glue("INFO — Class balance: {sum(y_train == 1)} positive / {sum(y_train == 0)} negative"))

dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# ── XGBoost parameters ────────────────────────────────────────────────────────
if (!is.null(opt$tuning_results_file)) {
    if (!file.exists(opt$tuning_results_file)) stop("ERROR — Tuning results file not found.")
    tuning_dt <- data.table::fread(opt$tuning_results_file)
    best_row  <- tuning_dt[which.max(cv_auc_mean)]
    print(glue("INFO — Using tuning results from {opt$tuning_results_file}"))
    print(glue("INFO — Best tuning row: cv_auc_mean = {round(best_row$cv_auc_mean, 4)}"))
    params <- list(
        objective        = "binary:logistic",
        eval_metric      = "auc",
        eta              = best_row$eta,
        max_depth        = as.integer(best_row$max_depth),
        min_child_weight = best_row$min_child_weight,
        gamma            = best_row$gamma,
        subsample        = best_row$subsample,
        colsample_bytree = best_row$colsample_bytree,
        colsample_bylevel = best_row$colsample_bylevel,
        lambda           = best_row$lambda,
        alpha            = best_row$alpha,
        max_delta_step   = best_row$max_delta_step,
        nthread          = opt$ncores
    )
} else {
    print("INFO — No tuning results file provided; using default parameters.")
    params <- list(
        objective        = "binary:logistic",
        eval_metric      = "auc",
        eta              = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.5,
        min_child_weight = 5,
        lambda           = 0.5,
        alpha            = 0.5,
        nthread          = opt$ncores
    )
}
print(glue("INFO — XGBoost params: {paste(names(params), unlist(params), sep='=', collapse=', ')}"))

# ── Cross-validation to find optimal nrounds ─────────────────────────────────
set.seed(2023)
print(glue("INFO — Running {opt$nfolds}-fold CV (early stopping at 50 rounds)..."))

cv_result <- xgb.cv(
    params            = params,
    data              = dtrain,
    nrounds           = 5000,
    nfold             = opt$nfolds,
    early_stopping_rounds = 100,
    verbose           = 1,
    print_every_n     = 100
)

# Extract best nrounds from the evaluation log directly — more robust than
# cv_result$best_iteration, which is empty when early stopping never triggers
# or under xgboost v2+ where the field behaviour changed.
best_nrounds <- which.max(cv_result$evaluation_log$test_auc_mean)
best_auc     <- cv_result$evaluation_log$test_auc_mean[best_nrounds]

# Fallback: if which.max returns nothing (empty log), use all rounds
if (length(best_nrounds) == 0 || is.na(best_nrounds)) {
    best_nrounds <- nrow(cv_result$evaluation_log)
    print(glue("WARNING — Could not determine best_nrounds; using {best_nrounds}"))
}

print(glue("INFO — Best nrounds: {best_nrounds}  (CV AUC = {round(best_auc, 4)})"))

# ── Train final model on all training data ────────────────────────────────────
print(glue("INFO — Training final model (nrounds = {best_nrounds})..."))
model <- xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = best_nrounds,
    verbose = 0
)

# ── Save model ────────────────────────────────────────────────────────────────
print(glue("INFO — Saving model to {opt$model_rds_file}"))
saveRDS(model, file = opt$model_rds_file)

# optionally save to json
# xgb.save(model, opt$model_rds_file) # change to .json when needed

# ── Save feature importance (gain) ───────────────────────────────────────────
# Same two-column layout as train_enpact.R: feature | <metadata>
# NOTE: these are importance scores, not linear weights.
importance <- xgb.importance(model = model)

# Pad with zero-importance features (features absent from any tree)
all_features <- colnames(X_train)
imp_dt <- data.table(feature = all_features)
imp_dt <- merge(imp_dt,
                importance[, .(Feature, Gain)],
                by.x = "feature", by.y = "Feature",
                all.x = TRUE)
imp_dt[is.na(Gain), Gain := 0]
setnames(imp_dt, "Gain", opt$metadata)
imp_dt <- imp_dt[order(-get(opt$metadata))]

print(glue("INFO — Non-zero importance features: {sum(imp_dt[[opt$metadata]] > 0)} / {nrow(imp_dt)}"))
print(glue("INFO — Saving feature importance to {opt$weights_file}"))
data.table::fwrite(imp_dt, file = opt$weights_file,
                   sep = "\t", row.names = FALSE, col.names = TRUE,
                   quote = FALSE, compress = "gzip")

print("INFO — Done.")
