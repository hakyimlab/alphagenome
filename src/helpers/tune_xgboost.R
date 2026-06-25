# Author: Temi
# Description: Random hyperparameter search for XGBoost TF binding model.
#              Samples n_configs parameter sets, evaluates each with xgb.cv,
#              then trains a final model on the best configuration.
#
# Usage:
#   Rscript tune_xgboost.R \
#       --train_data_file    train.tsv.gz \
#       --weights_file       weights.tsv.gz \
#       --model_rds_file     model.rds \
#       --tuning_results_file tuning_results.tsv \
#       --metadata           ag_xgb_tune \
#       [--n_configs         50] \
#       [--nfolds            5] \
#       [--ncores            12] \
#       [--seed              2023] \
#       [--features_group    embeddings_128bp]
#
# Output
# ------
# tuning_results_file : TSV with all configurations tried and their CV AUC
# model_rds_file      : xgb.Booster trained with the best configuration
# weights_file        : feature importance (Gain) for the best model

suppressPackageStartupMessages(library("optparse"))

option_list <- list(
    make_option("--train_data_file",     help = "Training data (TSV.GZ)"),
    make_option("--weights_file",        help = "Output: feature importance TSV.GZ"),
    make_option("--model_rds_file",      help = "Output: best trained xgb.Booster RDS"),
    make_option("--tuning_results_file", help = "Output: all CV results TSV"),
    make_option("--metadata",            help = "Column name written to weights file"),
    make_option("--n_configs",  type = "integer", default = 100L,
                help = "Number of random parameter configurations to try [default: 50]"),
    make_option("--nfolds",     type = "integer", default = 5L,
                help = "CV folds for each configuration [default: 5]"),
    make_option("--ncores",     type = "integer", default = 12L,
                help = "Threads for XGBoost [default: 12]"),
    make_option("--seed",       type = "integer", default = 2023L,
                help = "Random seed [default: 2023]"),
    make_option("--features_group", default = NULL,
                help = "Comma-separated feature-column prefix(es), e.g. embeddings_128bp")
)

opt <- parse_args(OptionParser(option_list = option_list))
print(opt)

suppressPackageStartupMessages({
    library(tidyverse)
    library(glue)
    library(data.table)
    library(xgboost)
})

set.seed(opt$seed)
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
n_pos   <- sum(y_train == 1)
n_neg   <- sum(y_train == 0)
print(glue("INFO — Training matrix: {nrow(X_train)} rows × {ncol(X_train)} features"))
print(glue("INFO — Class balance: {n_pos} positive / {n_neg} negative  (ratio {round(n_neg/n_pos, 2)})"))

dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# ── Parameter search space ────────────────────────────────────────────────────
# Log-uniform sampling for scale-sensitive params (learning rate, regularisation)
log_uniform <- function(n, lo, hi) exp(runif(n, log(lo), log(hi)))

n <- opt$n_configs

search_grid <- data.frame(
    # learning rate — small values need more rounds; early stopping compensates
    eta              = log_uniform(n, 0.01, 0.3),

    # tree structure
    max_depth        = sample(c(3L, 4L, 5L, 6L, 8L), n, replace = TRUE),
    min_child_weight = sample(c(1L, 3L, 5L, 10L, 20L), n, replace = TRUE),
    gamma            = log_uniform(n, 1e-3, 3.0),   # min gain to make a split

    # stochastic subsampling — reduces variance and speeds up training
    subsample        = runif(n, 0.5, 1.0),
    colsample_bytree = runif(n, 0.3, 1.0),   # features sampled per tree
    colsample_bylevel= runif(n, 0.5, 1.0),   # features sampled per level

    # regularisation
    lambda           = log_uniform(n, 0.01, 10.0),  # L2
    alpha            = log_uniform(n, 0.01, 10.0),  # L1

    # leaf output shrinkage (not the same as eta)
    max_delta_step   = sample(c(0L, 1L, 5L), n, replace = TRUE),

    stringsAsFactors = FALSE
)

print(glue("INFO — Starting random search over {n} configurations..."))

# ── Run CV for each configuration ─────────────────────────────────────────────
results <- vector("list", n)

for (i in seq_len(n)) {
    t0 <- proc.time()["elapsed"]

    params <- list(
        objective         = "binary:logistic",
        eval_metric       = "auc",
        eta               = search_grid$eta[i],
        max_depth         = search_grid$max_depth[i],
        min_child_weight  = search_grid$min_child_weight[i],
        gamma             = search_grid$gamma[i],
        subsample         = search_grid$subsample[i],
        colsample_bytree  = search_grid$colsample_bytree[i],
        colsample_bylevel = search_grid$colsample_bylevel[i],
        lambda            = search_grid$lambda[i],
        alpha             = search_grid$alpha[i],
        max_delta_step    = search_grid$max_delta_step[i],
        nthread           = opt$ncores
    )

    cv <- tryCatch(
        xgb.cv(
            params                = params,
            data                  = dtrain,
            nrounds               = 500,
            nfold                 = opt$nfolds,
            early_stopping_rounds = 50,
            verbose               = 0
        ),
        error = function(e) {
            message(glue("  config {i}: CV error — {conditionMessage(e)}"))
            NULL
        }
    )

    elapsed <- round(proc.time()["elapsed"] - t0, 1)

    if (is.null(cv)) {
        results[[i]] <- c(search_grid[i, ],
                          list(best_nrounds = NA_integer_,
                               cv_auc_mean  = NA_real_,
                               cv_auc_std   = NA_real_,
                               elapsed_sec  = elapsed))
        next
    }

    best_iter <- which.max(cv$evaluation_log$test_auc_mean)
    if (length(best_iter) == 0 || is.na(best_iter)) best_iter <- nrow(cv$evaluation_log)

    best_auc  <- cv$evaluation_log$test_auc_mean[best_iter]
    best_std  <- cv$evaluation_log$test_auc_std[best_iter]

    results[[i]] <- c(search_grid[i, ],
                      list(best_nrounds = best_iter,
                           cv_auc_mean  = best_auc,
                           cv_auc_std   = best_std,
                           elapsed_sec  = elapsed))

    print(glue(
        "  [{i}/{n}]  AUC={round(best_auc, 4)} ±{round(best_std, 4)}  ",
        "nrounds={best_iter}  eta={round(search_grid$eta[i],3)}  ",
        "depth={search_grid$max_depth[i]}  lambda={round(search_grid$lambda[i],3)}  ",
        "alpha={round(search_grid$alpha[i],3)}  mcw={search_grid$min_child_weight[i]}  ",
        "({elapsed}s)"
    ))
}

results_dt <- as.data.table(do.call(rbind, lapply(results, as.data.frame)))
results_dt <- results_dt[order(-cv_auc_mean)]

# ── Save all tuning results ───────────────────────────────────────────────────
print(glue("INFO — Saving tuning results to {opt$tuning_results_file}"))
data.table::fwrite(results_dt, file = opt$tuning_results_file,
                   sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

# ── Identify best configuration ───────────────────────────────────────────────
best_row <- results_dt[1, ]
print(glue("INFO — Best configuration:"))
print(as.data.frame(best_row))

best_params <- list(
    objective         = "binary:logistic",
    eval_metric       = "auc",
    eta               = best_row$eta,
    max_depth         = as.integer(best_row$max_depth),
    min_child_weight  = as.integer(best_row$min_child_weight),
    gamma             = best_row$gamma,
    subsample         = best_row$subsample,
    colsample_bytree  = best_row$colsample_bytree,
    colsample_bylevel = best_row$colsample_bylevel,
    lambda            = best_row$lambda,
    alpha             = best_row$alpha,
    max_delta_step    = as.integer(best_row$max_delta_step),
    nthread           = opt$ncores
)

best_nrounds <- as.integer(best_row$best_nrounds)
if (is.na(best_nrounds) || best_nrounds < 1L) best_nrounds <- 500L

# ── Train final model ─────────────────────────────────────────────────────────
print(glue("INFO — Training final model (nrounds={best_nrounds}, CV AUC={round(best_row$cv_auc_mean, 4)})..."))
model <- xgb.train(
    params  = best_params,
    data    = dtrain,
    nrounds = best_nrounds,
    verbose = 0
)

# ── Save model ────────────────────────────────────────────────────────────────
print(glue("INFO — Saving model to {opt$model_rds_file}"))
saveRDS(model, file = opt$model_rds_file)

# ── Save feature importance ───────────────────────────────────────────────────
importance  <- xgb.importance(model = model)
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