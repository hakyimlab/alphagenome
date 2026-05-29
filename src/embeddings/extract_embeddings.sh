#!/bin/bash
# extract_embeddings.sh
# ─────────────────────────────────────────────────────────────────────────────
# Submit a SLURM array to extract AlphaGenome embeddings.
#
# Two modes
# ─────────
# reference (default)
#   Splits intervals across N GPU tasks.  Each task writes:
#     reference_emb_shard{task}.h5
#   Use --merge to auto-submit a merge job after all shards finish.
#
# personalized
#   Splits a samples list across N GPU tasks.  Each task processes a contiguous
#   chunk of samples and writes one file per sample:
#     {sample_id}_emb.h5
#   Requires --samples-file, --vcf-dir, --vcf-pattern.
#
# Usage
# ─────
#   # reference (default)
#   bash extract_embeddings.sh \
#       --bed-file   /path/to/intervals.bed \
#       --output-dir /scratch/midway2/temi/alphagenome/embeddings/ref_run \
#       [--fasta-file /path/to/genome.fasta] \
#       [--n-shards 4] [--aggregate mean] [--output-types embeddings_128bp] \
#       [--time 12:00:00] [--job-name my_job] [--merge]
#
#   # personalized
#   bash extract_embeddings.sh --mode personalized \
#       --bed-file      /path/to/intervals.bed \
#       --samples-file  /path/to/samples.txt \
#       --vcf-dir       /path/to/vcfs \
#       --vcf-pattern   'ALL.{chrom}.vcf.gz' \
#       --output-dir    /scratch/midway2/temi/alphagenome/embeddings/pers_run \
#       [--n-shards 8] [--aggregate mean] [--time 24:00:00]
#
#   # YAML config (CLI flags override individual keys)
#   bash extract_embeddings.sh --config embeddings.yaml
#   bash extract_embeddings.sh --config embeddings.yaml --n-shards 8
#
# YAML config keys
# ────────────────
#   mode:         reference          # reference | personalized
#   bed_file:     /path/to/intervals.bed
#   output_dir:   /scratch/midway2/temi/alphagenome/embeddings/my_run
#   fasta_file:   /path/to/genome.fasta  # default: hg38
#   n_shards:     4
#   aggregate:    mean               # mean | max | none
#   output_types: embeddings_128bp   # embeddings_128bp[,embeddings_1bp]
#   job_name:     my_job             # default: emb_<bed stem>
#   time:         12:00:00
#   merge:        false              # reference only: auto-submit merge job
#   # personalized only:
#   samples_file: /path/to/samples.txt
#   vcf_dir:      /path/to/vcfs
#   vcf_pattern:  'ALL.{chrom}.vcf.gz'
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICT_SCRIPT="${SCRIPT_DIR}/predict_embeddings.py"
MERGE_SCRIPT="$(dirname "${SCRIPT_DIR}")/merge/merge_embeddings.py"

# ── All variables start empty; CLI > YAML > hardcoded defaults ────────────────
CONFIG=""
MODE=""
BED_FILE=""
OUTPUT_DIR=""
FASTA_FILE=""
N_SHARDS=""
AGGREGATE=""
OUTPUT_TYPES=""
JOB_NAME=""
TIME=""
DO_MERGE=""
# personalized only
SAMPLES_FILE=""
VCF_DIR=""
VCF_PATTERN=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG=$2;       shift 2 ;;
        --mode)         MODE=$2;         shift 2 ;;
        --bed-file)     BED_FILE=$2;     shift 2 ;;
        --output-dir)   OUTPUT_DIR=$2;   shift 2 ;;
        --fasta-file)   FASTA_FILE=$2;   shift 2 ;;
        --n-shards)     N_SHARDS=$2;     shift 2 ;;
        --aggregate)    AGGREGATE=$2;    shift 2 ;;
        --output-types) OUTPUT_TYPES=$2; shift 2 ;;
        --job-name)     JOB_NAME=$2;     shift 2 ;;
        --time)         TIME=$2;         shift 2 ;;
        --merge)        DO_MERGE=true;   shift   ;;
        --samples-file) SAMPLES_FILE=$2; shift 2 ;;
        --vcf-dir)      VCF_DIR=$2;      shift 2 ;;
        --vcf-pattern)  VCF_PATTERN=$2;  shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep -v '^set' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1  (use --help for usage)"; exit 1 ;;
    esac
done

# ── Fill any unset values from YAML config ────────────────────────────────────
if [[ -n "$CONFIG" ]]; then
    [[ ! -f "$CONFIG" ]] && { echo "ERROR: config file not found: $CONFIG"; exit 1; }

    _yaml() {
        python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG')) or {}
v = c.get('$1')
if v is not None:
    print(str(v).strip())
" 2>/dev/null || true
    }

    [[ -z "$MODE" ]]         && { v="$(_yaml mode)";         [[ -n "$v" ]] && MODE="$v";         }
    [[ -z "$BED_FILE" ]]     && { v="$(_yaml bed_file)";     [[ -n "$v" ]] && BED_FILE="$v";     }
    [[ -z "$OUTPUT_DIR" ]]   && { v="$(_yaml output_dir)";   [[ -n "$v" ]] && OUTPUT_DIR="$v";   }
    [[ -z "$FASTA_FILE" ]]   && { v="$(_yaml fasta_file)";   [[ -n "$v" ]] && FASTA_FILE="$v";   }
    [[ -z "$N_SHARDS" ]]     && { v="$(_yaml n_shards)";     [[ -n "$v" ]] && N_SHARDS="$v";     }
    [[ -z "$AGGREGATE" ]]    && { v="$(_yaml aggregate)";    [[ -n "$v" ]] && AGGREGATE="$v";    }
    [[ -z "$OUTPUT_TYPES" ]] && { v="$(_yaml output_types)"; [[ -n "$v" ]] && OUTPUT_TYPES="$v"; }
    [[ -z "$JOB_NAME" ]]     && { v="$(_yaml job_name)";     [[ -n "$v" ]] && JOB_NAME="$v";     }
    [[ -z "$TIME" ]]         && { v="$(_yaml time)";         [[ -n "$v" ]] && TIME="$v";         }
    [[ -z "$DO_MERGE" ]]     && { v="$(_yaml merge)";        [[ -n "$v" ]] && DO_MERGE="$v";     }
    [[ -z "$SAMPLES_FILE" ]] && { v="$(_yaml samples_file)"; [[ -n "$v" ]] && SAMPLES_FILE="$v"; }
    [[ -z "$VCF_DIR" ]]      && { v="$(_yaml vcf_dir)";      [[ -n "$v" ]] && VCF_DIR="$v";      }
    [[ -z "$VCF_PATTERN" ]]  && { v="$(_yaml vcf_pattern)";  [[ -n "$v" ]] && VCF_PATTERN="$v";  }
fi

# ── Hardcoded defaults ────────────────────────────────────────────────────────
MODE="${MODE:-reference}"
N_SHARDS="${N_SHARDS:-4}"
AGGREGATE="${AGGREGATE:-mean}"
OUTPUT_TYPES="${OUTPUT_TYPES:-embeddings_128bp}"
DO_MERGE="${DO_MERGE:-false}"

# Mode-specific time defaults
if [[ -z "$TIME" ]]; then
    [[ "$MODE" == "personalized" ]] && TIME="02:00:00" || TIME="01:00:00"
fi

# ── Validate mode ─────────────────────────────────────────────────────────────
if [[ "$MODE" != "reference" && "$MODE" != "personalized" ]]; then
    echo "ERROR: --mode must be 'reference' or 'personalized' (got: $MODE)"; exit 1
fi

# ── Validate required fields ──────────────────────────────────────────────────
[[ -z "$BED_FILE" ]]   && { echo "ERROR: bed_file / --bed-file is required";  exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "ERROR: output_dir / --output-dir is required"; exit 1; }
[[ ! -f "$BED_FILE" ]] && { echo "ERROR: BED file not found: $BED_FILE"; exit 1; }
[[ ! -f "$PREDICT_SCRIPT" ]] && {
    echo "ERROR: predict_embeddings.py not found at ${PREDICT_SCRIPT}"; exit 1; }

if [[ "$MODE" == "personalized" ]]; then
    [[ -z "$SAMPLES_FILE" ]] && { echo "ERROR: --samples-file required for personalized mode"; exit 1; }
    [[ -z "$VCF_DIR" ]]      && { echo "ERROR: --vcf-dir required for personalized mode";     exit 1; }
    [[ -z "$VCF_PATTERN" ]]  && { echo "ERROR: --vcf-pattern required for personalized mode"; exit 1; }
    [[ ! -f "$SAMPLES_FILE" ]] && { echo "ERROR: samples file not found: $SAMPLES_FILE"; exit 1; }
    [[ ! -d "$VCF_DIR" ]]      && { echo "ERROR: VCF directory not found: $VCF_DIR";     exit 1; }
    if [[ "$DO_MERGE" == "true" ]]; then
        echo "WARNING: --merge is only meaningful for reference mode; ignoring for personalized."
        DO_MERGE=false
    fi
fi

# ── Default job name ──────────────────────────────────────────────────────────
if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="emb_${MODE}_$(basename "${BED_FILE}" | sed 's/\.[^.]*$//')"
fi

# ── Count items being processed ───────────────────────────────────────────────
if [[ "$MODE" == "reference" ]]; then
    N_ITEMS=$(python3 -c "
lines = open('${BED_FILE}').readlines()
data  = [l for l in lines if l.strip() and not l.startswith('#')]
if data and not data[0].split()[1].lstrip('-').isdigit():
    data = data[1:]
print(len(data))
" 2>/dev/null || echo "?")
    ITEM_LABEL="intervals"
else
    N_ITEMS=$(wc -l < "${SAMPLES_FILE}" | tr -d ' ')
    ITEM_LABEL="samples"
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

ARRAY_RANGE="0-$((N_SHARDS - 1))"

# ── Print summary ─────────────────────────────────────────────────────────────
echo "=============================="
[[ -n "$CONFIG" ]]     && echo "Config       : ${CONFIG}"
echo "Mode         : ${MODE}"
echo "BED file     : ${BED_FILE}"
echo "${ITEM_LABEL^}    : ~${N_ITEMS}"
echo "Output dir   : ${OUTPUT_DIR}"
[[ -n "$FASTA_FILE" ]] \
    && echo "FASTA        : ${FASTA_FILE}" \
    || echo "FASTA        : (default hg38)"
if [[ "$MODE" == "personalized" ]]; then
    echo "Samples file : ${SAMPLES_FILE}"
    echo "VCF dir      : ${VCF_DIR}"
    echo "VCF pattern  : ${VCF_PATTERN}"
fi
echo "N shards     : ${N_SHARDS}"
echo "Aggregate    : ${AGGREGATE}"
echo "Types        : ${OUTPUT_TYPES}"
echo "Job name     : ${JOB_NAME}"
echo "Time limit   : ${TIME}"
echo "Logs         : ${LOG_DIR}"
[[ "$MODE" == "reference" ]] && echo "Auto-merge   : ${DO_MERGE}"
echo "=============================="

# ── Build --export string ─────────────────────────────────────────────────────
EXPORT_VARS="PREDICT_SCRIPT=${PREDICT_SCRIPT}"
EXPORT_VARS+=",BED_FILE=${BED_FILE}"
EXPORT_VARS+=",OUTPUT_DIR=${OUTPUT_DIR}"
EXPORT_VARS+=",N_SHARDS=${N_SHARDS}"
EXPORT_VARS+=",AGGREGATE=${AGGREGATE}"
EXPORT_VARS+=",OUTPUT_TYPES=${OUTPUT_TYPES}"
[[ -n "$FASTA_FILE" ]] && EXPORT_VARS+=",FASTA_FILE=${FASTA_FILE}"

if [[ "$MODE" == "personalized" ]]; then
    EXPORT_VARS+=",SAMPLES_FILE=${SAMPLES_FILE}"
    EXPORT_VARS+=",VCF_DIR=${VCF_DIR}"
    EXPORT_VARS+=",VCF_PATTERN=${VCF_PATTERN}"
    SBATCH_SCRIPT="${SCRIPT_DIR}/embed_personalized.sbatch"
else
    SBATCH_SCRIPT="${SCRIPT_DIR}/embed_reference.sbatch"
fi

[[ ! -f "$SBATCH_SCRIPT" ]] && {
    echo "ERROR: sbatch script not found: ${SBATCH_SCRIPT}"; exit 1; }

# ── Submit array ──────────────────────────────────────────────────────────────
SHARD_JOB_ID=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --job-name="${JOB_NAME}" \
    --time="${TIME}" \
    --export="${EXPORT_VARS}" \
    -o "${LOG_DIR}/slurm_%A_%a.out" \
    -e "${LOG_DIR}/slurm_%A_%a.err" \
    "${SBATCH_SCRIPT}" \
    | awk '{print $NF}')

echo ""
echo "Submitted    : ${SHARD_JOB_ID}"
echo "Array        : ${ARRAY_RANGE}  (${N_SHARDS} tasks)"

# ── Optionally submit a dependent merge job (reference mode only) ─────────────
if [[ "$DO_MERGE" == "true" ]]; then
    [[ ! -f "$MERGE_SCRIPT" ]] && {
        echo "WARNING: merge script not found at ${MERGE_SCRIPT} — skipping merge submission"
    } || {
        MERGED_H5="${OUTPUT_DIR}/merged_embeddings.h5"
        CONDA_ENV=/beagle3/haky/users/shared_software/TFXcan-pipeline-tools

        MERGE_JOB_ID=$(sbatch \
            --dependency="afterok:${SHARD_JOB_ID}" \
            --job-name="${JOB_NAME}_merge" \
            --partition=caslake \
            --account=pi-haky \
            --nodes=1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=01:00:00 \
            -o "${LOG_DIR}/merge_%j.out" \
            -e "${LOG_DIR}/merge_%j.err" \
            --wrap="source /home/temi/miniconda3/etc/profile.d/conda.sh && \
                    conda activate ${CONDA_ENV} && \
                    python ${MERGE_SCRIPT} \
                        --h5-dir ${OUTPUT_DIR} \
                        --output ${MERGED_H5} \
                        --format hdf5" \
            | awk '{print $NF}')

        echo "Merge job    : ${MERGE_JOB_ID}  (runs after ${SHARD_JOB_ID})"
        echo "Merged HDF5  : ${MERGED_H5}"
    }
fi

# ── Monitoring hints ──────────────────────────────────────────────────────────
echo ""
echo "── Monitor ─────────────────────────────────────────────────────────────"
echo "  squeue -j ${SHARD_JOB_ID}"
echo "  Logs      : tail -f ${LOG_DIR}/slurm_${SHARD_JOB_ID}_0.out"
if [[ "$MODE" == "reference" ]]; then
    echo "  Progress  : ls ${OUTPUT_DIR}/reference_emb_shard*.h5 2>/dev/null | wc -l  (of ${N_SHARDS})"
    echo ""
    echo "── Merge (if not using --merge) ─────────────────────────────────────────"
    echo "  python ${MERGE_SCRIPT} \\"
    echo "      --h5-dir ${OUTPUT_DIR} \\"
    echo "      --output ${OUTPUT_DIR}/merged_embeddings.h5"
else
    echo "  Progress  : ls ${OUTPUT_DIR}/*_emb.h5 2>/dev/null | wc -l  (of ${N_ITEMS} samples)"
fi
echo "─────────────────────────────────────────────────────────────────────────"
