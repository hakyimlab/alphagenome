# alphagenome

Inference pipeline for generating epigenomic predictions on genomic intervals using
[AlphaGenome](https://deepmind.google/research/publications/alphagenomic/), locally with
pre-downloaded model weights. Supports two modes:

- **Personalized** — predictions on phased haplotype sequences per individual (uses VCF variants)
- **Reference** — predictions on the GRCh38 reference sequence (no VCF needed)

### Date
Fri May 15 2026

### Author
Temi

---

## Quick start

```bash
# Personalized: split 200 samples across 20 GPUs
bash src/alphagenomeInfer.sh --config configs/run.yaml --n-gpus 20

# Reference: split 40,000 intervals across 10 GPU shards
bash src/alphagenomeInfer.sh --config configs/run_reference.yaml --reference --n-gpus 10

# Subset samples at launch time (overrides samples_file in YAML)
bash src/alphagenomeInfer.sh --config configs/run.yaml --n-gpus 10 --samples files/EUR_subset.tsv

# Interactive (single GPU, inside srun / tmux)
bash src/predict_run.sh --config configs/run.yaml
```

After inference, merge the per-sample or per-shard HDF5 files:

```bash
# Merge reference shards → single HDF5
bash src/merge/mergeHDF5.sh \
    --h5-dir predictions/run_reference/h5 \
    --output predictions/run_reference/merged.h5

# Merge reference shards → parquet (rows = intervals, cols = tracks)
bash src/merge/mergeHDF5.sh \
    --h5-dir predictions/run_reference/h5 \
    --output predictions/run_reference/merged.parquet \
    --format parquet

# Submit merge as a SLURM job
bash src/merge/mergeHDF5.sh \
    --h5-dir predictions/run_reference/h5 \
    --output predictions/run_reference/merged.h5 \
    --slurm
```

## Configuration file reference

All inference settings live in a YAML file passed via `--config`. CLI flags in
`alphagenomeInfer.sh` (e.g. `--n-gpus`, `--samples`) override the values in the file.

### Data paths

| Key | Description |
|-----|-------------|
| `model_path` | Path to AlphaGenome model weights directory (e.g. `allfolds` model) |
| `fasta_file` | GRCh38 FASTA file (must be indexed with `.fai`) |
| `vcf_dir` | Directory containing per-chromosome phased VCF files (personalized mode only). VCFs chromosome column must have "chr" prefix e.g. chr1, ch19|
| `vcf_pattern` | Filename template with `{chrom}` placeholder, e.g. `ALL.{chrom}.vcf.gz` |

### Input files

| Key | Description |
|-----|-------------|
| `bed_file` | Tab-separated file of genomic intervals: columns `chrom` (e.g. chr1, chr19), `start`, `end` (0-based half-open). Each interval must be ≤ `window_size` bp. |
| `samples_file` | One sample ID per line. IDs must match the sample names inside the VCFs. Not used in reference mode. |

### Output

| Key | Description |
|-----|-------------|
| `output_dir` | Directory where per-sample HDF5 files (`{sample_id}.h5`) or reference shard files (`reference_shard{N}.h5`) are written |
| `profile_dir` | Directory for per-GPU timing TSVs (only written when `profile: true`) |

### Inference parameters

| Key | Values | Description |
|-----|--------|-------------|
| `window_size` | integer (default `131072`) | Sequence window fed to the model (128 Kbp = 1024 bins of 128 bp). Rarely needs changing. |
| `output_types` | list of one or more: `CHIP_TF`, `CHIP_HISTONE`, `ATAC`, `DNASE`, `CAGE`, `RNA_SEQ`, `PROCAP` | Which prediction heads to request and store. More types → larger files and longer write times. |
| `pad_bins` | integer ≥ 0 (default `0`) | Bins added on each side of the focal interval when slicing. With `aggregate: true` on a 1 bp interval: `pad_bins=0` → 2 center bins, `pad_bins=1` → 4 bins, `pad_bins=2` → 6 bins. Ignored when `full_output: true`. |
| `method` | `mean` or `sum` | Reduction applied across bins when `aggregate: true`. |
| `aggregate` | `true` / `false` | When `true`, bins are reduced to a single vector `(n_tracks,)` per interval. When `false`, returns the sliced bin matrix `(n_bins, n_tracks)`. |
| `full_output` | `true` / `false` | When `true`, saves the entire model output `(1024, n_tracks)` with no slicing. Overrides `aggregate` and `pad_bins`. Produces large files. |

**Output shape summary:**

| `full_output` | `aggregate` | Shape saved per haplotype per interval |
|---|---|---|
| `true` | — | `(1024, n_tracks)` — raw model output |
| `false` | `false` | `(n_bins, n_tracks)` — sliced to interval ± pad_bins |
| `false` | `true` | `(n_tracks,)` — mean/sum across TSS bins |

### Runtime

| Key | Values | Description |
|-----|--------|-------------|
| `n_workers` | integer (default `1`) | Parsl parallel workers for sequence building. Keep at `1` to avoid NFS contention on shared filesystems. |
| `skip_existing` | `true` / `false` | Skip samples whose output HDF5 already exists. Useful for resuming interrupted runs. |
| `profile` | `true` / `false` | Write per-interval inference and HDF5 write timings to `profile_dir`. |

---

## Example configs

### Aggregated scalar — smallest output, fastest downstream

```yaml
output_types: [CHIP_TF]
pad_bins:     1
method:       mean
aggregate:    true
full_output:  false
```

Stores one `(n_tracks,)` vector per interval (e.g. 1617 values for CHIP_TF).

### Sliced bins around focal site

```yaml
output_types: [CHIP_TF, ATAC]
pad_bins:     4
aggregate:    false
full_output:  false
```

Stores a `(n_bins, n_tracks)` matrix per interval, where `n_bins` depends on interval
length and `pad_bins`.

### Full raw output — largest, no slicing

```yaml
output_types: [CHIP_TF]
full_output:  true
```

Stores the complete `(1024, n_tracks)` model output for every interval. Use only when
bin-resolution predictions are needed downstream.

---

## HDF5 output structure

**Personalized** (`{sample_id}.h5`):
```
attrs: sample_id, window_size, output_types, pad_bins, method, aggregate, full_output
{interval_id}/
  attrs: chrom, start, end
  haplotype1/{output_type}   shape: see table above
  haplotype2/{output_type}
```

**Reference** (`reference_shard{N}.h5`):
```
attrs: sample_id="reference", shard_id, pad_bins, method, aggregate, output_types
{interval_id}/
  attrs: chrom, start, end
  haplotype1/{output_type}
```

---

## Other notes
- `n_workers=1` is recommended; higher values cause problems.
