#!/usr/bin/env Rscript
# gather_predictions.R
#
# Reads per-individual AlphaGenome HDF5 files, sums haplotype1 + haplotype2,
# builds a 3D array (individual × features × gene/locus), and saves as RDS.
#
# Usage:
#   Rscript gather_predictions.R \
#       --h5_dir    /path/to/h5/ \
#       --output    /path/to/ag_array_ifg.rds.gz \
#       --n_cores   10

suppressPackageStartupMessages({
    library(optparse)
    library(rhdf5)
    library(foreach)
    library(doSNOW)
})

option_list <- list(
    make_option("--h5_dir",   type = "character", help = "Directory containing per-individual *.h5 files"),
    make_option("--output",   type = "character", help = "Output RDS path (use .rds.gz for compression)"),
    make_option("--n_cores",  type = "integer",   default = NULL,
                help = "Number of parallel workers [default: detectCores() - 2]")
)
opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$h5_dir), !is.null(opt$output))
dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)

h5_files    <- list.files(opt$h5_dir, pattern = "\\.h5$", full.names = TRUE)
individuals <- tools::file_path_sans_ext(basename(h5_files))
if (length(h5_files) == 0L) stop("No .h5 files found in: ", opt$h5_dir)
message(sprintf("Found %d H5 files in %s", length(h5_files), opt$h5_dir))

feature_tracks <- c("ATAC", "CAGE", "CHIP_HISTONE", "CHIP_TF", "DNASE")
feature_dims   <- c(ATAC = 167L, CAGE = 546L, CHIP_HISTONE = 1116L, CHIP_TF = 1617L, DNASE = 305L)
feature_names  <- unlist(mapply(function(tr, n) paste0(tr, "_", seq_len(n)),
                                names(feature_dims), feature_dims, SIMPLIFY = FALSE))
n_features <- sum(feature_dims)

read_individual_h5 <- function(fpath, feature_tracks, n_features) {
    tryCatch({
        listing    <- rhdf5::h5ls(fpath, recursive = 1L)
        loci_names <- listing[listing$group == "/", "name"]
        if (length(loci_names) == 0L) return(NULL)

        mat <- vapply(loci_names, function(locus) {
            hap1 <- unlist(lapply(feature_tracks, function(tr)
                rhdf5::h5read(fpath, paste0("/", locus, "/haplotype1/", tr))))
            hap2 <- unlist(lapply(feature_tracks, function(tr)
                rhdf5::h5read(fpath, paste0("/", locus, "/haplotype2/", tr))))
            hap1 + hap2
        }, numeric(n_features))
        rhdf5::h5closeAll()
        t(mat)  # loci × features
    }, error = function(e) {
        message("Skipping ", basename(fpath), ": ", conditionMessage(e))
        NULL
    })
}

n_cores <- if (!is.null(opt$n_cores)) opt$n_cores else max(1L, parallel::detectCores() - 2L)
n_cores <- min(n_cores, length(h5_files))
message(sprintf("Using %d parallel workers", n_cores))

cl <- makeCluster(n_cores)
registerDoSNOW(cl)
pb       <- txtProgressBar(min = 0, max = length(h5_files), style = 3)
progress <- function(n) setTxtProgressBar(pb, n)

ind_list <- foreach(
    i             = seq_along(h5_files),
    .packages     = "rhdf5",
    .options.snow = list(progress = progress)
) %dopar% {
    read_individual_h5(h5_files[i], feature_tracks, n_features)
}

close(pb)
stopCluster(cl)

ind_data <- setNames(ind_list, individuals)
ind_data <- Filter(Negate(is.null), ind_data)

if (length(ind_data) == 0L) stop("No individuals loaded successfully.")

common_loci <- Reduce(intersect, lapply(ind_data, rownames))
message(sprintf("Individuals loaded : %d", length(ind_data)))
message(sprintf("Common loci        : %d", length(common_loci)))
message(sprintf("Features           : %d", n_features))

ag_array <- array(
    NA_real_,
    dim      = c(length(ind_data), length(common_loci), n_features),
    dimnames = list(
        individual = names(ind_data),
        locus      = common_loci,
        feature    = feature_names
    )
)
for (i in seq_along(ind_data)) {
    ag_array[i, , ] <- ind_data[[i]][common_loci, , drop = FALSE]
}

# permute to individual × features × gene
ag_array_ifg <- aperm(ag_array, perm = c(1L, 3L, 2L))
dimnames(ag_array_ifg) <- list(
    individual = dimnames(ag_array)$individual,
    feature    = dimnames(ag_array)$feature,
    gene       = dimnames(ag_array)$locus
)
message(sprintf("Array dims: %s", paste(dim(ag_array_ifg), collapse = " × ")))

compress <- grepl("\\.gz$", opt$output)
message("Saving to: ", opt$output)
saveRDS(ag_array_ifg, file = opt$output, compress = compress)
message("Done.")