# Author: Temi
# Date: Thursday July 27 2023
# Description: script to train elastic net TFPred models
# Usage: Rscript train_enet.R [options]

suppressPackageStartupMessages(library("optparse"))

option_list <- list(
    make_option("--train_data_file", help='data to train with enet'),
    make_option("--features_group", help='CHIP_TF,DNASE', default = NULL),
    make_option("--nfolds", type="integer", default=5L, help='How many cv folds?'),
    make_option("--metadata", help='extra information for file-naming'),
    make_option("--weights_file", help='output txt file containing the weights'),
    make_option("--model_rds_file", help='output rds file of the glmnet object'),
    make_option("--ncores", default = 40L, type = 'integer')
)

opt <- parse_args(OptionParser(option_list=option_list))

print(opt)

library(tidyverse)
library(glue)
library(R.utils)
library(data.table)
library(glmnet)
library(doParallel)
library(parallel)

# c('CHIP_TF', 'DNASE', 'CHIP_HISTONE', 'ATAC', 'CAGE')

#   chrom     start       end interval_id               CHIP_TF_0 CHIP_TF_1 CHIP_TF_2 CHIP_TF_3 CHIP_TF_4
# * <chr>     <dbl>     <dbl> <chr>                         <dbl>     <dbl>     <dbl>     <dbl>     <dbl>
# 1 chr10 101111775 101111784 chr10_101111775_101111784      53.2     110.       72.1     208.       74.6
# 2 chr10 103505193 103505202 chr10_103505193_103505202      55.7     101        71.2     100.       68.1
# 3 chr10 104885195 104885204 chr10_104885195_104885204      68.8     108.      103.      114.      104. 
# 4 chr10  10739044  10739053 chr10_10739044_10739053        50.6      78.8      65.2      61.8      59.6
# 5 chr10 111107394 111107409 chr10_111107394_111107409     126.      186.      111.      122.      113. 



metadata_columns <- c('chrom', 'start', 'end', 'interval_id', 'binding_class')



seed <- 2023
if(file.exists(opt$train_data_file)){
    dt_train <- data.table::fread(opt$train_data_file) #%>% dplyr::select(-all_of(remove_features))
} else {
    stop('ERROR - Training data cannot be found.')
}
# remove duplicates
dt_train <- dplyr::distinct(dt_train) #%>% dplyr::slice_sample(n = 2000)

# remove missing values
cc <- complete.cases(dt_train)
dt_train <- dt_train[cc, ]

# split the data
if(!is.null(opt$features_group)){
    features_group <- strsplit(opt$features_group, ',')[[1]]
    print(glue("INFO - Training using these groups of features: {paste(features_group, collapse=', ')}"))
    X_train <- dt_train %>% 
        dplyr::select(!any_of(metadata_columns)) %>% 
        dplyr::select(dplyr::starts_with(features_group)) %>% 
        as.matrix()
} else {
    X_train <- dt_train %>% dplyr::select(!any_of(metadata_columns)) %>% as.matrix()
}


y_meta <- dt_train %>% dplyr::select(any_of(metadata_columns)) %>% as.data.frame()
y_train <- y_meta$binding_class

print(glue('INFO - Read training data, number of features = {ncol(X_train)}'))

cl <- opt$ncores #parallel::makeCluster(5)
print(glue('INFO - Found {parallel::detectCores()} cores but using {cl}'))

set.seed(seed)

doParallel::registerDoParallel(cl)
print(glue('INFO - training enet model'))
cv_model <- glmnet::cv.glmnet(x=X_train, y=y_train, family = "binomial", type.measure = "auc", alpha = 0.5, keep=T, parallel=T, nfolds=opt$nfolds, trace.it=F)

print(cv_model)
print(glue('INFO - Saving the model to `{opt$model_rds_file}`'))
saveRDS(cv_model, file=opt$model_rds_file)
doParallel::stopImplicitCluster()

#cv_model <- readRDS('/beagle3/haky/users/temi/projects/TFXcan/experiments/auprc/AR_Prostate.no_prostate_features.glmnet_model.rds')

weights <- stats::coef(cv_model, s = 'lambda.1se')
fnames <- rownames(weights)
fnames[fnames == "(Intercept)"] <- "intercept"
weights <- weights %>% as.matrix() %>% as.data.table()
colnames(weights) <- opt$metadata  
weights <- weights %>% 
    dplyr::mutate(feature = fnames) %>% 
    dplyr::relocate(feature)

print(glue('INFO - Saving the weights'))
data.table::fwrite(weights, file=opt$weights_file, sep='\t', row.names=F, col.names=T, quote=F, compress = 'gzip')











# IGNORE -----------------------------------

# weights <- weights %>% tibble::column_to_rownames('feature') %>% as.matrix()
# weights.matrix <- weights %>% dplyr::select(-feature) %>% as.matrix()
# weights_intercepts <- weights.matrix[1, , drop = F]
# weights_features <- weights.matrix[-1, , drop = F]

# print(glue('INFO - Evaluating on test data'))
# X_test <- dt_test[, -c(2,3)] %>% tibble::column_to_rownames('locus') %>% as.matrix()
# y_test <- dt_test[, c(1,2,3)] |> as.data.frame()

# stopifnot(dim(X_test)[2] == dim(weights_features)[1])

# predictions <- as.matrix(X_test %*% weights_features[, , drop = FALSE])
# y_hat <- apply(predictions, 1, function(each_row){
#     weights_intercepts + each_row
# }) %>% t()

# colnames(y_hat) <- colnames(weights_features)

# y_hat <- y_hat[,,drop = FALSE] %>% as.data.table(keep.rownames = 'locus')
# predictions <- dplyr::inner_join(y_hat, y_test, by = 'locus')
# print(glue('INFO - Saving the predicted values'))
# data.table::fwrite(predictions, file=glue("{opt$predictions_file}"), sep='\t', row.names=F, col.names=T, quote=F, compress = 'gzip')



# Y_hats <- purrr:::map(.x=individual_enpact_features, .f = function(each_file){
#     dt <- data.table::fread(each_file)
#     X <- as.matrix(dt[, -c(1)])

#     stopifnot(dim(X)[2] == dim(weights_features)[1])

#     # prediction
#     y_hat_noIntercept <- X %*% weights_features[, , drop = FALSE]
#     y_hat <- apply(y_hat_noIntercept, 1, function(each_row){
#         weights_intercepts + each_row
#     }) %>% t()

#     if(!is.null(opt$for_models)){
#         y_hat <- t(y_hat)
#     }

#     colnames(y_hat) <- colnames(weights_features)
#     rownames(y_hat) <- dt$id
#     return(y_hat)

# }, .progress = TRUE)