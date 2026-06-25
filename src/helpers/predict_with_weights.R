#!/usr/bin/env Rscript
# predict_with_weights.R
#
# For each *.weights.tsv.gz in weights_dir, select the matching features from a
# 3D array (individual × features × gene), compute linear predictions, and
# write one output file per feature class.
#
# Usage:
#   Rscript predict_with_weights.R \
#       --array_rds   /path/to/ag_array_ifg.rds \
#       --weights_dir /path/to/weights/ \
#       --output_dir  /path/to/output/
#
# Output files: {output_dir}/{feature_class}.predictions.tsv.gz
#   Rows    = individuals (rownames of array)
#   Columns = genes / loci (dim-3 names of array)

suppressPackageStartupMessages({
    library(optparse)
    library(data.table)
})

option_list <- list(
    make_option("--array_rds",   type = "character", help = "Path to RDS file containing the 3D array (individual × features × gene)"),
    make_option("--weights_dir", type = "character", help = "Directory containing *.weights.tsv.gz files"),
    make_option("--output_dir",  type = "character", help = "Directory to write prediction files")
)
opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(
    !is.null(opt$array_rds),
    !is.null(opt$weights_dir),
    !is.null(opt$output_dir)
)
dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

message("Loading 3D array from: ", opt$array_rds)
ag_array <- readRDS(opt$array_rds)   # individual × features × gene
stopifnot(length(dim(ag_array)) == 3L)

ind_names     <- dimnames(ag_array)[[1]]
feature_names <- dimnames(ag_array)[[2]]
gene_names    <- dimnames(ag_array)[[3]]
message(sprintf("Array dimensions: %d individuals × %d features × %d genes",
                length(ind_names), length(feature_names), length(gene_names)))

weight_files <- list.files(opt$weights_dir, pattern = "\\.weights\\.tsv\\.gz$", full.names = TRUE)
if (length(weight_files) == 0L) stop("No *.weights.tsv.gz files found in: ", opt$weights_dir)
message(sprintf("Found %d weight files", length(weight_files)))

for (wf in weight_files) {
    feature_class <- sub("\\.weights\\.tsv\\.gz$", "", basename(wf))
    message("Processing: ", feature_class)

    wt <- fread(wf)
    stopifnot(c("feature", "ag_enpact_weight") %in% names(wt))

    intercept <- wt[feature == "intercept", ag_enpact_weight]
    wt        <- wt[feature != "intercept"]

    # match weights to array feature dimension
    matched <- intersect(wt$feature, feature_names)
    missing <- setdiff(wt$feature, feature_names)
    if (length(missing) > 0L)
        warning(sprintf("%s: %d weight features not found in array: %s ...",
                        feature_class, length(missing), paste(head(missing, 3), collapse = ", ")))
    if (length(matched) == 0L) {
        warning(feature_class, ": no matching features — skipping")
        next
    }
    wt <- wt[feature %in% matched]
    w  <- setNames(wt$ag_enpact_weight, wt$feature)

    # slice array to matched features, preserving order
    feat_idx <- match(names(w), feature_names)
    # sub-array: individual × matched_features × gene
    sub_arr  <- ag_array[, feat_idx, , drop = FALSE]

    # prediction: for each gene, individuals %*% weights + intercept
    # sub_arr[i, f, g]  →  pred[i, g] = sum_f(w[f] * sub_arr[i, f, g]) + intercept
    # Reshape to (individual × gene) via matrix multiply over the feature axis
    n_ind   <- length(ind_names)
    n_feat  <- length(w)
    n_genes <- length(gene_names)

    # reshape to (individual * gene) × features, multiply, reshape back
    mat   <- matrix(sub_arr, nrow = n_ind * n_genes, ncol = n_feat, byrow = FALSE)
    # sub_arr is stored [ind, feat, gene] — need ind × feat per gene slice
    # aperm to gene × individual × feature, then reshape
    tmp   <- aperm(sub_arr, c(3L, 1L, 2L))          # gene × individual × feature
    mat2  <- matrix(tmp, nrow = n_genes * n_ind, ncol = n_feat)  # (gene*ind) × feature
    preds <- matrix((mat2 %*% w) + intercept, nrow = n_genes, ncol = n_ind)  # gene × individual
    preds <- t(preds)                                 # individual × gene
    rownames(preds) <- ind_names
    colnames(preds) <- gene_names

    out_file <- file.path(opt$output_dir, paste0(feature_class, ".predictions.tsv.gz"))
    pred_dt  <- data.table(individual = ind_names, as.data.table(preds))
    data.table::fwrite(pred_dt, out_file, sep = "\t", compress = "gzip")
    message(sprintf("  Written: %s  [%d × %d]", basename(out_file), nrow(preds), ncol(preds)))
}
message("Done.")