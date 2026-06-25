"""
AlphaGenome inference — unified personalized and reference modes.

Personalized mode (default):
    Parallelises across samples. Each SLURM task handles a chunk of samples.
    Usage:  python predict.py --config run.yaml

Reference mode (--reference):
    No VCF / no personalisation. Parallelises across interval shards so all
    GPUs stay busy even with a single genome.
    Usage:  python predict.py --config run_reference.yaml --reference \\
                --shard-id $SLURM_ARRAY_TASK_ID --n-shards 20
"""

import os

# XLA flags — must be set before any JAX import
os.environ.setdefault("XLA_FLAGS", " ".join([
    "--xla_gpu_deterministic_ops",
    "--xla_gpu_enable_scatter_determinism_expander=True",
    "--xla_gpu_enable_triton_gemm=False",
    #"--xla_gpu_autotune_level=0",
]))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import argparse
import csv
import logging
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import parsl
import yaml
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

warnings.filterwarnings("ignore")

ALPHAGENOME_RESEARCH_DIR = "/beagle3/haky/data/alpha_genome/src"

# src/ is the parent of better_inference/ — add it so modules/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.intervals   import build_intervals_table
from modules.sequences   import build_sequences_for_sample
from modules.predictions import (
    predict_and_save, PRED_ATTR_MAP,
    slice_prediction, aggregate_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_PROFILE_FIELDS = ["event", "sample_id", "interval_id", "haplotype", "elapsed_s"]


# ── Reference sequence building ───────────────────────────────────────────────

def build_reference_sequences(intervals_df, fasta_file):
    """
    Extract reference genome sequences for every interval.
    Returns {interval_id: {"hap1": str} | {"error": str}}.
    No VCF overlay — pure hg38 reference.
    """
    from pyfaidx import Fasta

    def _clean(seq):
        return "".join(c if c.upper() in "ACGT" else "N" for c in seq).upper()

    fasta = Fasta(fasta_file)
    sequences = {}
    for _, row in intervals_df.iterrows():
        try:
            seq = _clean(str(fasta[row.chrom][row.padded_start:row.padded_end].seq))
            expected = int(row.padded_end) - int(row.padded_start)
            if len(seq) < expected:
                seq = seq + "N" * (expected - len(seq))
            sequences[row.interval_id] = {"hap1": seq}
        except Exception as exc:
            sequences[row.interval_id] = {"error": str(exc)}
    return sequences


# ── Reference prediction / HDF5 writer ───────────────────────────────────────

def predict_reference_and_save(
    model,
    sequences_dict,
    intervals_df,
    output_dir,
    requested_output_names,
    output_type_map,
    window_size=131072,
    pad_bins=0,
    pad_bp=0,
    method="mean",
    aggregate=True,
    full_output=False,
    shard_id=0,
    profile=False,
):
    """
    Run reference genome predictions for this shard of intervals.
    Single haplotype only (stored as haplotype1).
    Output: {output_dir}/reference_shard{shard_id}.h5
    """
    requested_outputs = [output_type_map[n] for n in requested_output_names]
    output_file = os.path.join(output_dir, f"reference_shard{shard_id}.h5")
    errors  = []
    timings = []

    with h5py.File(output_file, "w") as f:
        f.attrs["sample_id"]    = "reference"
        f.attrs["shard_id"]     = shard_id
        f.attrs["window_size"]  = window_size
        f.attrs["output_types"] = ",".join(requested_output_names)
        f.attrs["pad_bins"]     = pad_bins
        f.attrs["pad_bp"]       = pad_bp
        f.attrs["method"]       = method
        f.attrs["aggregate"]    = aggregate
        f.attrs["full_output"]  = full_output

        for _, row in intervals_df.iterrows():
            interval_id  = row["interval_id"]
            orig_start   = int(row["start"])
            orig_end     = int(row["end"])
            padded_start = int(row["padded_start"])

            seq_data = sequences_dict.get(interval_id)
            if seq_data is None or "error" in seq_data:
                err = (seq_data or {}).get("error", "missing sequence")
                errors.append({"interval_id": interval_id, "error": err})
                continue

            try:
                grp = f.create_group(interval_id)
                grp.attrs["chrom"] = row.chrom
                grp.attrs["start"] = orig_start
                grp.attrs["end"]   = orig_end

                hap_grp = grp.create_group("haplotype1")

                t_inf = time.time()
                pred  = model.predict_sequence(
                    seq_data["hap1"],
                    requested_outputs=requested_outputs,
                    ontology_terms=None,
                )
                inf_elapsed = time.time() - t_inf

                t_write = time.time()
                for out_name in requested_output_names:
                    track = getattr(pred, PRED_ATTR_MAP[out_name], None)
                    if track is not None:
                        if full_output:
                            data = track.values
                        elif aggregate:
                            data = aggregate_prediction(
                                track.values, orig_start, orig_end, padded_start,
                                window_size, pad_bins, pad_bp, method,
                            )
                        else:
                            data = slice_prediction(
                                track.values, orig_start, orig_end, padded_start,
                                window_size, pad_bins, pad_bp,
                            )
                        hap_grp.create_dataset(out_name, data=np.array(data), compression="gzip")
                write_elapsed = time.time() - t_write

                if profile:
                    timings.extend([
                        {"event": "inference",  "sample_id": "reference",
                         "interval_id": interval_id, "haplotype": "haplotype1",
                         "elapsed_s": round(inf_elapsed, 4)},
                        {"event": "hdf5_write", "sample_id": "reference",
                         "interval_id": interval_id, "haplotype": "haplotype1",
                         "elapsed_s": round(write_elapsed, 4)},
                    ])

            except Exception as exc:
                errors.append({"interval_id": interval_id, "error": str(exc)})

    return {"output_file": output_file, "errors": errors, "timings": timings}


# ── Config / arg parsing ──────────────────────────────────────────────────────

def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def parse_args():
    p = argparse.ArgumentParser(
        description="AlphaGenome batch inference — personalized and reference modes"
    )
    p.add_argument("--config",      default=None, help="YAML config; CLI flags override")
    p.add_argument("--reference",   action="store_true", default=False,
                   help="Reference genome mode: no VCF, parallelise by interval shard")
    p.add_argument("--shard-id",    type=int, default=None,
                   help="Shard index for this task (reference mode; default: SLURM_ARRAY_TASK_ID)")
    p.add_argument("--n-shards",    type=int, default=None,
                   help="Total number of interval shards (reference mode; default: 1)")

    # shared inference parameters
    p.add_argument("--samples-file",  default=None)
    p.add_argument("--bed-file",      default=None)
    p.add_argument("--output-dir",    default=None)
    p.add_argument("--profile-dir",   default=None)
    p.add_argument("--model-path",    default=None)
    p.add_argument("--fasta-file",    default=None)
    p.add_argument("--vcf-dir",       default=None)
    p.add_argument("--vcf-pattern",   default=None)
    p.add_argument("--window-size",   type=int, default=None)
    p.add_argument("--device",        default=None)
    p.add_argument("--n-workers",     type=int, default=None)
    p.add_argument("--pad-bins",      type=int, default=None)
    p.add_argument("--pad-bp",        type=int, default=None)
    p.add_argument("--method",        default=None, choices=["mean", "sum"])
    p.add_argument("--output-types",  nargs="+", default=None,
                   choices=list(PRED_ATTR_MAP.keys()))
    p.add_argument("--skip-existing",     action="store_true",  default=None)
    p.add_argument("--no-skip-existing",  dest="skip_existing", action="store_false")
    p.add_argument("--aggregate",         action="store_true",  default=None)
    p.add_argument("--no-aggregate",      dest="aggregate",     action="store_false")
    p.add_argument("--full-output",       action="store_true",  default=None)
    p.add_argument("--no-full-output",    dest="full_output",   action="store_false")
    p.add_argument("--profile",           action="store_true",  default=None)
    p.add_argument("--no-profile",        dest="profile",       action="store_false")

    args = p.parse_args()
    cfg  = load_config(args.config) if args.config else {}

    defaults = {
        "samples_file":  cfg.get("samples_file"),
        "bed_file":      cfg.get("bed_file"),
        "output_dir":    cfg.get("output_dir"),
        "profile_dir":   cfg.get("profile_dir"),
        "model_path":    cfg.get("model_path",   "/beagle3/haky/data/alpha_genome/weights/allfolds"),
        "fasta_file":    cfg.get("fasta_file",   "/project2/haky/Data/hg_sequences/hg38/Homo_sapiens_assembly38.fasta"),
        "vcf_dir":       cfg.get("vcf_dir",      "/project2/haky/Data/1000G/vcf_snps_only"),
        "vcf_pattern":   cfg.get("vcf_pattern",  "ALL.{chrom}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz"),
        "window_size":   cfg.get("window_size",  131072),
        "device":        cfg.get("device",       "cuda:0"),
        "n_workers":     cfg.get("n_workers",    1),
        "pad_bins":      cfg.get("pad_bins",     0),
        "pad_bp":        cfg.get("pad_bp",       0),
        "method":        cfg.get("method",       "mean"),
        "output_types":  cfg.get("output_types", ["CHIP_TF"]),
        "skip_existing": cfg.get("skip_existing", True),
        "aggregate":     cfg.get("aggregate",    True),
        "full_output":   cfg.get("full_output",  False),
        "profile":       cfg.get("profile",      False),
        "n_shards":      cfg.get("n_shards",     1),
        "shard_id":      cfg.get("shard_id",     None),
    }

    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    # resolve shard_id from SLURM env if not provided
    if args.shard_id is None:
        task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
        args.shard_id = int(task_env) if task_env else 0

    # validate required fields
    required = ["bed_file", "output_dir"]
    if not args.reference:
        required.append("samples_file")
    missing = [f for f in required if not getattr(args, f, None)]
    if missing:
        p.error(f"Missing required fields (set via CLI or YAML): {', '.join(missing)}")

    return args


# ── Profile writer ────────────────────────────────────────────────────────────

def write_profile(timings, profile_file):
    with open(profile_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PROFILE_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timings)
    log.info(f"Profile written: {profile_file}  ({len(timings)} records)")


def _save_profile(args, all_timings):
    profile_dir = Path(args.profile_dir if args.profile_dir else args.output_dir)
    os.makedirs(profile_dir, exist_ok=True)
    gpu_id = (os.environ.get("SLURM_ARRAY_TASK_ID") or
              os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    write_profile(all_timings, profile_dir / f"profile_gpu{gpu_id}.tsv")


# ── Mode implementations ──────────────────────────────────────────────────────

def _run_reference(args, model, intervals_df, output_type_map, all_timings):
    """Reference mode: shard intervals across GPUs, single haplotype."""
    shard_df = intervals_df.iloc[args.shard_id::args.n_shards].reset_index(drop=True)
    log.info(f"Reference mode  shard {args.shard_id}/{args.n_shards} "
             f"— {len(shard_df)} of {len(intervals_df)} intervals")

    out_file = Path(args.output_dir) / f"reference_shard{args.shard_id}.h5"
    if args.skip_existing and out_file.exists():
        log.info(f"Skipping — {out_file} already exists")
        return

    t_seq = time.time()
    log.info("Building reference sequences ...")
    sequences_dict = build_reference_sequences(shard_df, args.fasta_file)
    seq_elapsed = time.time() - t_seq
    log.info(f"Sequences built in {seq_elapsed:.1f}s")

    if args.profile:
        all_timings.append({
            "event": "seq_build", "sample_id": "reference", "interval_id": "",
            "haplotype": "", "elapsed_s": round(seq_elapsed, 4),
        })

    t_pipeline = time.time()
    result = predict_reference_and_save(
        model=model,
        sequences_dict=sequences_dict,
        intervals_df=shard_df,
        output_dir=args.output_dir,
        requested_output_names=args.output_types,
        output_type_map=output_type_map,
        window_size=args.window_size,
        pad_bins=args.pad_bins,
        pad_bp=args.pad_bp,
        method=args.method,
        aggregate=args.aggregate,
        full_output=args.full_output,
        shard_id=args.shard_id,
        profile=args.profile,
    )
    pipeline_elapsed = time.time() - t_pipeline

    n_errs = len(result["errors"])
    log.info(f"Done  errors={n_errs}  total={pipeline_elapsed/60:.1f}min")
    for err in result["errors"]:
        log.warning(f"  {err['interval_id']}: {err['error'][:120]}")

    if args.profile:
        all_timings.extend(result["timings"])
        all_timings.append({
            "event": "pipeline_total", "sample_id": "reference", "interval_id": "",
            "haplotype": "", "elapsed_s": round(pipeline_elapsed, 4),
        })
        _save_profile(args, all_timings)

    if result["errors"]:
        failed_file = Path(args.output_dir) / f"failed_shard{args.shard_id}.txt"
        failed_file.write_text(
            "\n".join(f"{e['interval_id']}\t{e['error']}" for e in result["errors"]) + "\n"
        )
        log.warning(f"Failed intervals written to {failed_file}")


def _run_personalized(args, model, intervals_df, output_type_map, all_timings):
    """Personalized mode: build per-sample haplotype sequences with VCF overlay."""
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

    intervals_by_chrom = {}
    for rec in intervals_df.to_dict("records"):
        intervals_by_chrom.setdefault(rec["chrom"], []).append(rec)

    task_id     = os.environ.get("SLURM_ARRAY_TASK_ID", str(os.getpid()))
    runinfo_dir = str(Path(args.output_dir).parent / "parsl_logs" / f"runinfo_{task_id}")
    os.makedirs(runinfo_dir, exist_ok=True)
    parsl.load(Config(
        executors=[ThreadPoolExecutor(label="io_pool", max_threads=args.n_workers,
                                      working_dir=runinfo_dir)],
        strategy=None,
        run_dir=runinfo_dir,
    ))
    log.info(f"Parsl        : {args.n_workers} I/O threads")
    log.info(f"Pipeline     : {len(samples)} samples × {len(intervals_df)} intervals"
             + ("  [profiling ON]" if args.profile else ""))

    futures = [
        (sid, build_sequences_for_sample(
            sid, intervals_by_chrom, args.fasta_file, args.vcf_dir, args.vcf_pattern))
        for sid in samples
    ]

    completed, failed = [], []
    t_pipeline = time.time()

    try:
        for i, (sample_id, fut) in enumerate(futures):
            t_sample = time.time()
            try:
                t_seq = time.time()
                sequences_dict    = fut.result()
                seq_build_elapsed = time.time() - t_seq

                result = predict_and_save(
                    model=model,
                    sample_id=sample_id,
                    sequences_dict=sequences_dict,
                    intervals_df=intervals_df,
                    output_dir=args.output_dir,
                    requested_output_names=args.output_types,
                    output_type_map=output_type_map,
                    window_size=args.window_size,
                    pad_bins=args.pad_bins,
                    pad_bp=args.pad_bp,
                    method=args.method,
                    aggregate=args.aggregate,
                    full_output=args.full_output,
                    profile=args.profile,
                )
                sample_elapsed = time.time() - t_sample
                completed.append(result)

                n_errs = len(result["errors"])
                log.info(f"[{i+1}/{len(futures)}] {sample_id}  {sample_elapsed:.1f}s  "
                         f"{'OK' if not n_errs else f'{n_errs} error(s)'}")
                for err in result["errors"]:
                    log.warning(f"  {err['interval_id']}: {err['error'][:120]}")

                if args.profile:
                    all_timings += [
                        {"event": "seq_build",    "sample_id": sample_id, "interval_id": "",
                         "haplotype": "", "elapsed_s": round(seq_build_elapsed, 4)},
                        {"event": "sample_total", "sample_id": sample_id, "interval_id": "",
                         "haplotype": "", "elapsed_s": round(sample_elapsed, 4)},
                    ]
                    all_timings.extend(result["timings"])

            except Exception as exc:
                failed.append({"sample_id": sample_id, "error": str(exc)})
                log.error(f"[{i+1}/{len(futures)}] {sample_id} FAILED: {exc}")

    finally:
        parsl.clear()

    pipeline_elapsed = time.time() - t_pipeline
    log.info(f"Done  completed={len(completed)}  failed={len(failed)}  "
             f"total={pipeline_elapsed/60:.1f}min")

    if args.profile:
        all_timings.append({
            "event": "pipeline_total", "sample_id": "", "interval_id": "",
            "haplotype": "", "elapsed_s": round(pipeline_elapsed, 4),
        })
        _save_profile(args, all_timings)

    if failed:
        failed_file = Path(args.output_dir) / "failed_samples.txt"
        failed_file.write_text(
            "\n".join(f"{f['sample_id']}\t{f['error']}" for f in failed) + "\n"
        )
        log.warning(f"Failed samples written to {failed_file}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.config:
        log.info(f"Config       : {args.config}")
    log.info(f"Mode         : {'reference' if args.reference else 'personalized'}")

    if ALPHAGENOME_RESEARCH_DIR not in sys.path:
        sys.path.insert(0, ALPHAGENOME_RESEARCH_DIR)

    from alphagenome.models import dna_output
    from alphagenome.models import dna_model as dna_model_base
    from alphagenome_research.model import dna_model
    import jax

    output_type_map = {name: getattr(dna_output.OutputType, name) for name in PRED_ATTR_MAP}

    os.makedirs(args.output_dir, exist_ok=True)
    intervals_df = build_intervals_table(args.bed_file)
    log.info(f"Intervals    : {len(intervals_df)}")

    try:
        device = jax.devices("gpu")[int(args.device.replace("cuda:", ""))]
    except Exception:
        log.warning("GPU not found — falling back to CPU")
        device = jax.devices("cpu")[0]

    log.info(f"Device       : {device}")
    log.info(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    organism_settings = {
        dna_model_base.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(fasta_path=args.fasta_file),
    }
    model = dna_model.create(args.model_path, organism_settings=organism_settings, device=device)
    model_load_elapsed = time.time() - t0
    log.info(f"Model loaded in {model_load_elapsed:.1f}s")

    all_timings = []
    if args.profile:
        all_timings.append({
            "event": "model_load", "sample_id": "", "interval_id": "",
            "haplotype": "", "elapsed_s": round(model_load_elapsed, 4),
        })

    if args.reference:
        _run_reference(args, model, intervals_df, output_type_map, all_timings)
    else:
        _run_personalized(args, model, intervals_df, output_type_map, all_timings)


if __name__ == "__main__":
    main()
