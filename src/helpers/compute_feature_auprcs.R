# Author: Temi
# Date: Tuesday May 26 2026
# Description: Compute per-feature AUPRC from a test data file
# Usage: Rscript compute_feature_auprcs.R [options]

suppressPackageStartupMessages(library("optparse"))

option_list <- list(
    make_option("--test_data_file", help = "path to test data TSV.GZ (chrom, start, end, interval_id, binding_class, features...)"),
    make_option("--output_file",    help = "output TSV.GZ file for per-feature AUPRCs"),
    make_option("--ncores",         type = "integer", default = 8L, help = "number of parallel workers [default: 8]")
)

opt <- parse_args(OptionParser(option_list = option_list))

print(opt)

library(data.table)
library(PRROC)
library(dplyr)
library(glue)
library(doParallel)
library(foreach)

metadata_cols <- c("chrom", "start", "end", "interval_id", "binding_class")

if(!file.exists(opt$test_data_file)){
    stop('ERROR - Test data file cannot be found.')
}

print(glue("INFO - Reading test data from {opt$test_data_file}"))
dt <- data.table::fread(opt$test_data_file)

feature_cols <- setdiff(colnames(dt), metadata_cols)
print(glue("INFO - {length(feature_cols)} features found, {nrow(dt)} intervals"))

# convert to matrix once — avoids repeated per-column lookups
feature_mat <- as.matrix(dt[, ..feature_cols])
pos_idx     <- which(dt$binding_class == 1)
neg_idx     <- which(dt$binding_class == 0)
print(glue("INFO - {length(pos_idx)} positive, {length(neg_idx)} negative intervals"))

doParallel::registerDoParallel(opt$ncores)
print(glue("INFO - Computing AUPRCs in parallel with {opt$ncores} workers"))

auprc_per_feature <- foreach::foreach(
    feat = feature_cols,
    .combine  = rbind,
    .packages = "PRROC"
) %dopar% {
    scores <- feature_mat[, feat]
    pr <- PRROC::pr.curve(
        scores.class0 = scores[pos_idx],
        scores.class1 = scores[neg_idx],
        curve = FALSE
    )
    data.frame(
        feature       = feat,
        feature_group = sub("_[0-9]+$", "", feat),
        auprc         = round(pr$auc.integral, 4)
    )
} |>
    dplyr::arrange(dplyr::desc(auprc))

doParallel::stopImplicitCluster()

print(glue("INFO - Done. Top features:"))
print(head(auprc_per_feature, 10))

dir.create(dirname(opt$output_file), recursive = TRUE, showWarnings = FALSE)
data.table::fwrite(auprc_per_feature, file = opt$output_file, sep = "\t",
                   row.names = FALSE, col.names = TRUE, quote = FALSE, compress = "gzip")
print(glue("INFO - Saved to {opt$output_file}"))
