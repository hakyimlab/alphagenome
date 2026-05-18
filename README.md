# alphagenome

Creating an inference pipeline for alphagenome

### Date
Fri May 15 2026

### Author
Temi


### Usage
bash launch.sh --config ../configs/run.yaml --n-gpus 20

# Override samples at launch time (e.g. a subset)
bash launch.sh --config configs/run.yaml --n-gpus 10 --samples files/EUR_subset.tsv

# Interactive
bash predict_run.sh --config configs/run.yaml

## To configure the slicing and aggregation, here are some examples
#### 1 — aggregated scalar (default, smallest)
aggregate:   true
full_output: false

#### 2 — sliced around TSS or the focal bp
aggregate:   false
full_output: false
pad_bins:    2      # 2 middle bins + 2 either side = 6 bins

#### 3 — full raw output (largest), save everything
full_output: true   # aggregate and pad_bins are ignored


