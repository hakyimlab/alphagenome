"""
AlphaGenome inference script — designed for SLURM array job execution.

Each invocation handles one batch of samples on one GPU:
  - Loads the model once
  - Builds haplotype sequences via Parsl ThreadPool (overlapped with GPU compute)
  - Predicts and aggregates per interval, writes per-sample HDF5
  - Skips samples whose HDF5 already exists (safe to rerun)

Usage:
    python predict.py \\
        --samples-file /path/to/samples.txt \\
        --bed-file     /path/to/intervals.bed \\
        --output-dir   /path/to/predictions \\
        [--device cuda:0] [--n-workers 4] [--pad-bins 0] [--method mean] [--profile]
"""

import os

# XLA flags — must be set before ANY JAX import
os.environ.setdefault("XLA_FLAGS", " ".join([
    "--xla_gpu_deterministic_ops",
    "--xla_gpu_enable_scatter_determinism_expander=True",
    "--xla_gpu_enable_triton_gemm=False",
]))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import argparse
import csv
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import parsl
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH               = "/beagle3/haky/data/alpha_genome/weights/fold0"
ALPHAGENOME_RESEARCH_DIR = "/beagle3/haky/data/alpha_genome/src"
FASTA_FILE               = "/project2/haky/Data/hg_sequences/hg38/Homo_sapiens_assembly38.fasta"
VCF_DIR                  = "/project2/haky/Data/1000G/vcf_snps_only"
VCF_PATTERN              = "ALL.{chrom}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz"
WINDOW_SIZE              = 131072
REQUESTED_OUTPUT_NAMES   = ["CHIP_TF"]

# add src/ to path so modules/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from modules.intervals   import build_intervals_table
from modules.sequences   import build_sequences_for_sample
from modules.predictions import predict_and_save, PRED_ATTR_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# TSV columns in the order they appear in the profile file
_PROFILE_FIELDS = ["event", "sample_id", "interval_id", "haplotype", "elapsed_s"]


def write_profile(timings, profile_file):
    with open(profile_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PROFILE_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timings)
    log.info(f"Profile written to {profile_file}  ({len(timings)} records)")


def parse_args():
    p = argparse.ArgumentParser(description="AlphaGenome batch inference")
    p.add_argument("--samples-file",  required=True,  help="One sample ID per line")
    p.add_argument("--bed-file",      required=True,  help="Intervals BED/TSV (0-based half-open)")
    p.add_argument("--output-dir",    required=True,  help="Directory for per-sample HDF5 output")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--n-workers",     type=int, default=4)
    p.add_argument("--pad-bins",      type=int, default=0)
    p.add_argument("--method",        default="mean", choices=["mean", "sum"])
    p.add_argument("--output-types",  nargs="+", default=REQUESTED_OUTPUT_NAMES,
                   choices=list(PRED_ATTR_MAP.keys()))
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--profile",       action="store_true", default=False,
                   help="Record per-stage timing to a TSV file")
    p.add_argument("--profile-dir",   default=None,
                   help="Directory for profile TSV (default: same as --output-dir)")
    return p.parse_args()


def main():
    args = parse_args()

    # AlphaGenome imports (after XLA env vars are set)
    if ALPHAGENOME_RESEARCH_DIR not in sys.path:
        sys.path.insert(0, ALPHAGENOME_RESEARCH_DIR)

    from alphagenome.models import dna_output
    from alphagenome.models import dna_model as dna_model_base
    from alphagenome_research.model import dna_model
    import jax

    output_type_map = {name: getattr(dna_output.OutputType, name) for name in PRED_ATTR_MAP}

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    samples = pd.read_csv(args.samples_file, header=None).iloc[:, 0].tolist()
    log.info(f"Samples      : {len(samples)}")

    if args.skip_existing:
        pending = [s for s in samples if not Path(args.output_dir, f"{s}.h5").exists()]
        skipped = len(samples) - len(pending)
        if skipped:
            log.info(f"Skipping     : {skipped} already-completed samples")
        samples = pending

    if not samples:
        log.info("All samples already complete. Exiting.")
        return

    intervals_df = build_intervals_table(args.bed_file)
    log.info(f"Intervals    : {len(intervals_df)}")

    intervals_by_chrom = {}
    for rec in intervals_df.to_dict("records"):
        intervals_by_chrom.setdefault(rec["chrom"], []).append(rec)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        device = jax.devices("gpu")[int(args.device.replace("cuda:", ""))]
    except Exception:
        log.warning("GPU not found, falling back to CPU")
        device = jax.devices("cpu")[0]

    log.info(f"Device       : {device}")
    log.info(f"Loading model from {MODEL_PATH} ...")
    t0 = time.time()
    organism_settings = {
        dna_model_base.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(fasta_path=FASTA_FILE),
    }
    model = dna_model.create(MODEL_PATH, organism_settings=organism_settings, device=device)
    model_load_elapsed = time.time() - t0
    log.info(f"Model loaded in {model_load_elapsed:.1f}s")

    all_timings = []
    if args.profile:
        all_timings.append({
            "event": "model_load", "sample_id": "", "interval_id": "",
            "haplotype": "", "elapsed_s": round(model_load_elapsed, 4),
        })

    # ── Parsl ─────────────────────────────────────────────────────────────────
    runinfo_dir = str(Path(args.output_dir).parent / "parsl_runinfo")
    os.makedirs(runinfo_dir, exist_ok=True)
    parsl.load(Config(
        executors=[ThreadPoolExecutor(label="io_pool", max_threads=args.n_workers,
                                      working_dir=runinfo_dir)],
        strategy=None,
        run_dir=runinfo_dir,
    ))
    log.info(f"Parsl loaded : {args.n_workers} I/O threads")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    log.info(f"Starting pipeline: {len(samples)} samples × {len(intervals_df)} intervals"
             + ("  [profiling ON]" if args.profile else ""))

    futures = [
        (sid, build_sequences_for_sample(sid, intervals_by_chrom, FASTA_FILE, VCF_DIR, VCF_PATTERN))
        for sid in samples
    ]

    completed, failed = [], []
    t_pipeline = time.time()

    try:
        for i, (sample_id, fut) in enumerate(futures):
            t_sample = time.time()
            try:
                t_seq = time.time()
                sequences_dict  = fut.result()
                seq_build_elapsed = time.time() - t_seq

                result = predict_and_save(
                    model=model,
                    sample_id=sample_id,
                    sequences_dict=sequences_dict,
                    intervals_df=intervals_df,
                    output_dir=args.output_dir,
                    requested_output_names=args.output_types,
                    output_type_map=output_type_map,
                    window_size=WINDOW_SIZE,
                    pad_bins=args.pad_bins,
                    method=args.method,
                    profile=args.profile,
                )
                sample_elapsed = time.time() - t_sample

                completed.append(result)
                n_errs = len(result["errors"])
                status = f"{n_errs} interval error(s)" if n_errs else "OK"
                log.info(f"[{i+1}/{len(futures)}] {sample_id}  {sample_elapsed:.1f}s  {status}")
                for err in result["errors"]:
                    log.warning(f"  {err['interval_id']}: {err['error'][:120]}")

                if args.profile:
                    all_timings.append({
                        "event": "seq_build", "sample_id": sample_id, "interval_id": "",
                        "haplotype": "", "elapsed_s": round(seq_build_elapsed, 4),
                    })
                    all_timings.append({
                        "event": "sample_total", "sample_id": sample_id, "interval_id": "",
                        "haplotype": "", "elapsed_s": round(sample_elapsed, 4),
                    })
                    all_timings.extend(result["timings"])

            except Exception as exc:
                failed.append({"sample_id": sample_id, "error": str(exc)})
                log.error(f"[{i+1}/{len(futures)}] {sample_id} FAILED: {exc}")

    finally:
        parsl.clear()

    pipeline_elapsed = time.time() - t_pipeline
    log.info(f"Done. Completed={len(completed)}  Failed={len(failed)}  "
             f"Total={pipeline_elapsed/60:.1f}min")

    if args.profile:
        all_timings.append({
            "event": "pipeline_total", "sample_id": "", "interval_id": "",
            "haplotype": "", "elapsed_s": round(pipeline_elapsed, 4),
        })
        profile_dir = Path(args.profile_dir if args.profile_dir else args.output_dir)
        os.makedirs(profile_dir, exist_ok=True)
        gpu_id = (os.environ.get("SLURM_ARRAY_TASK_ID")
                  or os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        profile_file = profile_dir / f"profile_gpu{gpu_id}.tsv"
        write_profile(all_timings, profile_file)

    if failed:
        failed_file = Path(args.output_dir) / "failed_samples.txt"
        failed_file.write_text("\n".join(f"{f['sample_id']}\t{f['error']}" for f in failed) + "\n")
        log.warning(f"Failed samples written to {failed_file}")


if __name__ == "__main__":
    main()
