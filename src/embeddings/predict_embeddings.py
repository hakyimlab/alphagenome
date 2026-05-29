"""
predict_embeddings.py
---------------------
Batch extraction of AlphaGenome's internal trunk embeddings.

Two modes
---------
reference (default)
    Extract embeddings from the hg38 reference genome.
    Parallelise by sharding intervals across SLURM array tasks.
    Output: reference_emb_shard{N}.h5 (one per task)
    HDF5:   /{interval_id}/embeddings_128bp[_hap1]   shape (3072,) if aggregated

personalized
    Extract embeddings from phased VCF-personalised sequences.
    Parallelise by splitting a samples file across SLURM array tasks.
    Each task processes a contiguous chunk of samples.
    Output: {sample_id}_emb.h5 (one per sample, written by the task that owns it)
    HDF5:   /{interval_id}/embeddings_128bp_hap1
                          /embeddings_128bp_hap2   shape (3072,) if aggregated

Embedding types (--output-types, comma-separated)
    embeddings_128bp  — TransformerTower trunk,  dim 3072,  at 128bp resolution
    embeddings_1bp    — SequenceDecoder output,  dim 1536,  at 1bp resolution
    Both are computed in the same forward pass at no extra GPU cost.
    ALWAYS aggregate embeddings_1bp (full shape 131072×1536 ≈ 768 MB/interval).

Usage
-----
    # via extract_embeddings.sh (recommended)
    bash extract_embeddings.sh --config embeddings.yaml

    # direct reference run (one shard)
    python predict_embeddings.py \\
        --bed-file /path/to/intervals.bed --output-dir /scratch/... \\
        --shard-id 0 --n-shards 4 --aggregate mean

    # direct personalized run (one shard of samples)
    python predict_embeddings.py --mode personalized \\
        --bed-file /path/to/intervals.bed --output-dir /scratch/... \\
        --samples-file /path/to/samples.txt \\
        --vcf-dir /path/to/vcfs --vcf-pattern 'ALL.{chrom}.vcf.gz' \\
        --shard-id 0 --n-shards 4 --aggregate mean

Notes
-----
* XLA flags are set before any JAX import — do not reorder the imports.
* build_personalized_sequences_for_sample is a plain Python function (no Parsl)
  mirroring the logic in modules/sequences.py so it runs cleanly in SLURM jobs.
"""

import os

# ── XLA flags — must be set before any JAX import ─────────────────────────────
os.environ.setdefault("XLA_FLAGS", " ".join([
    "--xla_gpu_deterministic_ops",
    "--xla_gpu_enable_scatter_determinism_expander=True",
    "--xla_gpu_enable_triton_gemm=False",
    "--xla_gpu_autotune_level=0",
]))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import argparse
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import yaml

warnings.filterwarnings("ignore")

ALPHAGENOME_RESEARCH_DIR = "/beagle3/haky/data/alpha_genome/src"

# This file lives in src/embeddings/; parent.parent = src/, which contains modules/
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.intervals            import build_intervals_table
from modules.embeddings_extractor import build_embedding_apply_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builders
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_sequences(intervals_df, fasta_file):
    """
    Extract hg38 reference sequences for every interval.
    Returns {interval_id: {"hap1": str} | {"error": str}}.
    """
    from pyfaidx import Fasta

    def _clean(seq):
        return "".join(c if c.upper() in "ACGT" else "N" for c in seq).upper()

    fasta = Fasta(fasta_file)
    sequences = {}
    for _, row in intervals_df.iterrows():
        try:
            seq      = _clean(str(fasta[row.chrom][row.padded_start:row.padded_end].seq))
            expected = int(row.padded_end) - int(row.padded_start)
            if len(seq) < expected:
                seq = seq + "N" * (expected - len(seq))
            sequences[row.interval_id] = {"hap1": seq}
        except Exception as exc:
            sequences[row.interval_id] = {"error": str(exc)}
    return sequences


def build_personalized_sequences_for_sample(
    sample_id,
    intervals_by_chrom,   # {chrom: [interval_record_dict, ...]}
    fasta_file,
    vcf_dir,
    vcf_pattern,          # e.g. "ALL.{chrom}.vcf.gz"
):
    """
    Build phased haplotype sequences for one sample.

    Plain Python re-implementation of modules.sequences.build_sequences_for_sample
    (which is Parsl-decorated and cannot be used directly in SLURM jobs).

    Returns {interval_id: {"hap1": str, "hap2": str} | {"error": str}}

    Only SNPs (single-base REF and ALT) are applied; indels are skipped to keep
    the window length fixed at 131 072 bp.
    """
    from pyfaidx import Fasta
    import pysam

    def _clean(seq):
        return "".join(c if c.upper() in "ACGT" else "N" for c in seq).upper()

    def _ref_seq(fasta, chrom, padded_start, padded_end):
        seq = _clean(str(fasta[chrom][padded_start:padded_end].seq))
        expected = padded_end - padded_start
        if len(seq) < expected:
            seq = seq + "N" * (expected - len(seq))
        return seq

    fasta     = Fasta(fasta_file)
    sequences = {}

    for chrom, records in intervals_by_chrom.items():
        vcf_path = os.path.join(vcf_dir, vcf_pattern.format(chrom=chrom))
        vcf      = pysam.VariantFile(vcf_path) if os.path.exists(vcf_path) else None

        for rec in records:
            interval_id  = rec["interval_id"]
            padded_start = rec["padded_start"]
            padded_end   = rec["padded_end"]
            try:
                ref  = _ref_seq(fasta, chrom, padded_start, padded_end)
                hap1 = list(ref)
                hap2 = list(ref)

                if vcf is not None:
                    for vrec in vcf.fetch(chrom, padded_start, padded_end):
                        if sample_id not in vrec.samples:
                            continue
                        gt = vrec.samples[sample_id]["GT"]
                        if None in gt or len(gt) < 2:
                            continue
                        alleles = vrec.alleles
                        a1 = alleles[gt[0]] if gt[0] is not None and gt[0] < len(alleles) else None
                        a2 = alleles[gt[1]] if gt[1] is not None and gt[1] < len(alleles) else None
                        # skip indels — length must stay fixed
                        if a1 and len(a1) == 1 and a2 and len(a2) == 1:
                            idx = vrec.start - padded_start
                            if 0 <= idx < len(hap1):
                                hap1[idx] = a1
                                hap2[idx] = a2

                sequences[interval_id] = {"hap1": "".join(hap1), "hap2": "".join(hap2)}
            except Exception as exc:
                sequences[interval_id] = {"error": str(exc)}

        if vcf is not None:
            vcf.close()

    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# Shared aggregation helper
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(arr, aggregate):
    """Reduce (bins, dim) → (dim,) via mean or max, or return unchanged."""
    if aggregate == "mean":
        return arr.mean(axis=0)
    elif aggregate == "max":
        return arr.max(axis=0)
    return arr   # None → full spatial


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: reference mode
# ─────────────────────────────────────────────────────────────────────────────

def extract_reference_embeddings_and_save(
    *,
    model,
    apply_fn_emb,
    sequences_dict,
    intervals_df,
    output_dir,
    shard_id,
    aggregate,
    output_types=("embeddings_128bp",),
    window_size=131072,
):
    """
    Reference mode — one shard of intervals, single haplotype.
    Output: {output_dir}/reference_emb_shard{shard_id}.h5

    HDF5 structure:
        /{interval_id}/
            attrs: chrom, start, end
            embeddings_128bp : (3072,) or (1024, 3072)
            embeddings_1bp   : (1536,) or (131072, 1536)  [if requested]
    """
    import jax

    output_types = list(output_types)
    save_128bp   = "embeddings_128bp" in output_types
    save_1bp     = "embeddings_1bp"   in output_types

    if save_1bp and aggregate is None:
        log.warning(
            "embeddings_1bp with aggregate=None → full (131072, 1536) per interval ≈ 768 MB. "
            "Consider --aggregate mean."
        )

    output_file = os.path.join(output_dir, f"reference_emb_shard{shard_id}.h5")
    agg_str     = aggregate if aggregate is not None else "none"
    errors, n_ok = [], 0
    encoder = model._one_hot_encoder

    with h5py.File(output_file, "w") as f:
        f.attrs.update({
            "sample_id": "reference", "mode": "reference",
            "shard_id": shard_id, "window_size": window_size,
            "aggregate": agg_str, "output_types": ",".join(output_types),
        })

        for _, row in intervals_df.iterrows():
            interval_id = row["interval_id"]
            seq_data    = sequences_dict.get(interval_id)
            if seq_data is None or "error" in seq_data:
                errors.append({"interval_id": interval_id,
                                "error": (seq_data or {}).get("error", "missing sequence")})
                continue
            try:
                # Guard only covers the forward pass — indexing and aggregation
                # outside it to avoid blocking JAX's implicit int scalar transfers.
                with model._device_context as dev, jax.transfer_guard("disallow"):
                    seq_arr = jax.device_put(
                        np.asarray(encoder.encode(seq_data["hap1"]))[np.newaxis], dev
                    )  # (1, W, 4)
                    org_arr = jax.device_put(np.full((1,), 0, dtype=np.int32), dev)
                    _, embeddings = apply_fn_emb(model._params, model._state, seq_arr, org_arr)

                # JAX arrays remain on GPU outside the guard; index and aggregate here.
                results = {}
                if save_128bp:
                    results["embeddings_128bp"] = _aggregate(
                        embeddings.embeddings_128bp[0], aggregate)
                if save_1bp:
                    results["embeddings_1bp"] = _aggregate(
                        embeddings.embeddings_1bp[0], aggregate)

                arrays = {k: np.array(jax.device_get(v)).astype(np.float32)
                          for k, v in results.items()}

                grp = f.create_group(interval_id)
                grp.attrs.update({"chrom": row.chrom,
                                   "start": int(row.start), "end": int(row.end)})
                for ds_name, arr in arrays.items():
                    grp.create_dataset(ds_name, data=arr, compression="gzip")
                n_ok += 1

            except Exception as exc:
                errors.append({"interval_id": interval_id, "error": str(exc)})

    return {"output_file": output_file, "errors": errors, "n_ok": n_ok}


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: personalized mode
# ─────────────────────────────────────────────────────────────────────────────

def extract_personalized_embeddings_and_save(
    *,
    model,
    apply_fn_emb,
    sample_id,
    sequences_dict,   # {interval_id: {"hap1": str, "hap2": str} | {"error": str}}
    intervals_df,
    output_dir,
    aggregate,
    output_types=("embeddings_128bp",),
    window_size=131072,
):
    """
    Personalized mode — both haplotypes for one sample.
    Output: {output_dir}/{sample_id}_emb.h5

    hap1 and hap2 are batched into a single forward pass (batch size 2),
    so GPU compute cost equals one reference interval, not two.

    HDF5 structure:
        /{interval_id}/
            attrs: chrom, start, end
            embeddings_128bp_hap1 : (3072,) or (1024, 3072)
            embeddings_128bp_hap2 : (3072,) or (1024, 3072)
            embeddings_1bp_hap1   : (1536,) or (131072, 1536)  [if requested]
            embeddings_1bp_hap2   : (1536,) or (131072, 1536)  [if requested]
    """
    import jax

    output_types = list(output_types)
    save_128bp   = "embeddings_128bp" in output_types
    save_1bp     = "embeddings_1bp"   in output_types

    if save_1bp and aggregate is None:
        log.warning(
            "embeddings_1bp with aggregate=None → full (131072, 1536) per haplotype ≈ 768 MB. "
            "Consider --aggregate mean."
        )

    output_file = os.path.join(output_dir, f"{sample_id}_emb.h5")
    agg_str     = aggregate if aggregate is not None else "none"
    errors, n_ok = [], 0
    encoder = model._one_hot_encoder

    with h5py.File(output_file, "w") as f:
        f.attrs.update({
            "sample_id": sample_id, "mode": "personalized",
            "window_size": window_size, "aggregate": agg_str,
            "output_types": ",".join(output_types),
        })

        for _, row in intervals_df.iterrows():
            interval_id = row["interval_id"]
            seq_data    = sequences_dict.get(interval_id)
            if seq_data is None or "error" in seq_data:
                errors.append({"interval_id": interval_id,
                                "error": (seq_data or {}).get("error", "missing sequence")})
                continue
            if "hap2" not in seq_data:
                # fall back: treat as reference (hap1 only), duplicate for hap2
                seq_data = {"hap1": seq_data["hap1"], "hap2": seq_data["hap1"]}
            try:
                # Guard only covers the forward pass — indexing and aggregation
                # outside it to avoid blocking JAX's implicit int scalar transfers.
                with model._device_context as dev, jax.transfer_guard("disallow"):
                    # Batch hap1 + hap2 in a single forward pass — no extra GPU cost
                    seq_arr = jax.device_put(
                        np.stack([
                            np.asarray(encoder.encode(seq_data["hap1"])),
                            np.asarray(encoder.encode(seq_data["hap2"])),
                        ]), dev
                    )  # (2, W, 4)
                    org_arr = jax.device_put(np.full((2,), 0, dtype=np.int32), dev)
                    _, embeddings = apply_fn_emb(model._params, model._state, seq_arr, org_arr)

                # JAX arrays remain on GPU outside the guard; index and aggregate here.
                results = {}
                if save_128bp:
                    results["embeddings_128bp_hap1"] = _aggregate(
                        embeddings.embeddings_128bp[0], aggregate)
                    results["embeddings_128bp_hap2"] = _aggregate(
                        embeddings.embeddings_128bp[1], aggregate)
                if save_1bp:
                    results["embeddings_1bp_hap1"] = _aggregate(
                        embeddings.embeddings_1bp[0], aggregate)
                    results["embeddings_1bp_hap2"] = _aggregate(
                        embeddings.embeddings_1bp[1], aggregate)

                arrays = {k: np.array(jax.device_get(v)).astype(np.float32)
                          for k, v in results.items()}

                grp = f.create_group(interval_id)
                grp.attrs.update({"chrom": row.chrom,
                                   "start": int(row.start), "end": int(row.end)})
                for ds_name, arr in arrays.items():
                    grp.create_dataset(ds_name, data=arr, compression="gzip")
                n_ok += 1

            except Exception as exc:
                errors.append({"interval_id": interval_id, "error": str(exc)})

    return {"output_file": output_file, "errors": errors, "n_ok": n_ok}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def parse_args():
    p = argparse.ArgumentParser(
        description="AlphaGenome embedding extraction — reference and personalized modes"
    )
    # ── mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--mode", choices=["reference", "personalized"], default=None,
                   help="reference: extract from hg38 (default). "
                        "personalized: apply VCF variants to get hap1/hap2 sequences.")
    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--config",       default=None,
                   help="YAML config file; CLI flags override individual keys")
    p.add_argument("--bed-file",     default=None)
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--model-path",   default=None)
    p.add_argument("--fasta-file",   default=None)
    p.add_argument("--shard-id",     type=int, default=None)
    p.add_argument("--n-shards",     type=int, default=None)
    p.add_argument("--aggregate",    default=None,
                   choices=["mean", "max", "none"],
                   help="Aggregate spatial bins: mean, max, or none (full)")
    p.add_argument("--output-types", default=None,
                   help="Comma-separated: embeddings_128bp[,embeddings_1bp] "
                        "(default: embeddings_128bp). Both computed in one pass.")
    p.add_argument("--device",       default=None)
    p.add_argument("--skip-existing",    action="store_true",  default=None)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    # ── personalized only ─────────────────────────────────────────────────────
    p.add_argument("--samples-file", default=None,
                   help="[personalized] One sample ID per line")
    p.add_argument("--vcf-dir",      default=None,
                   help="[personalized] Directory containing per-chromosome VCFs")
    p.add_argument("--vcf-pattern",  default=None,
                   help="[personalized] Filename pattern with {chrom} placeholder, "
                        "e.g. 'ALL.{chrom}.vcf.gz'")

    args = p.parse_args()
    cfg  = load_config(args.config) if args.config else {}

    defaults = {
        "mode":          cfg.get("mode",         "reference"),
        "bed_file":      cfg.get("bed_file"),
        "output_dir":    cfg.get("output_dir"),
        "model_path":    cfg.get("model_path",
                                 "/beagle3/haky/data/alpha_genome/weights/allfolds"),
        "fasta_file":    cfg.get("fasta_file",
                                 "/project2/haky/Data/hg_sequences/hg38/"
                                 "Homo_sapiens_assembly38.fasta"),
        "n_shards":      cfg.get("n_shards",     1),
        "shard_id":      cfg.get("shard_id",     None),
        "aggregate":     cfg.get("aggregate",    "mean"),
        "output_types":  cfg.get("output_types", "embeddings_128bp"),
        "device":        cfg.get("device",       "cuda:0"),
        "skip_existing": cfg.get("skip_existing", True),
        # personalized
        "samples_file":  cfg.get("samples_file"),
        "vcf_dir":       cfg.get("vcf_dir"),
        "vcf_pattern":   cfg.get("vcf_pattern"),
    }

    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    # "none" string → Python None for aggregate
    if args.aggregate == "none":
        args.aggregate = None

    # Parse comma-separated output_types → list
    if isinstance(args.output_types, str):
        args.output_types = [t.strip() for t in args.output_types.split(",")]
    valid = {"embeddings_128bp", "embeddings_1bp"}
    unknown = set(args.output_types) - valid
    if unknown:
        p.error(f"Unknown --output-types: {unknown}.  Valid: {valid}")

    # Resolve shard_id from SLURM env if not set explicitly
    if args.shard_id is None:
        task_env = os.environ.get("SLURM_ARRAY_TASK_ID")
        args.shard_id = int(task_env) if task_env else 0

    # Validate required fields
    missing = [f for f in ["bed_file", "output_dir"] if not getattr(args, f, None)]
    if missing:
        p.error(f"Missing required: {', '.join(missing)}")

    if args.mode == "personalized":
        pers_missing = [f for f in ["samples_file", "vcf_dir", "vcf_pattern"]
                        if not getattr(args, f, None)]
        if pers_missing:
            p.error(f"Personalized mode requires: {', '.join(pers_missing)}")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info(f"Mode         : {args.mode}")
    log.info(f"Output types : {args.output_types}")
    log.info(f"Aggregate    : {args.aggregate!r}")
    if args.config:
        log.info(f"Config       : {args.config}")

    if ALPHAGENOME_RESEARCH_DIR not in sys.path:
        sys.path.insert(0, ALPHAGENOME_RESEARCH_DIR)

    from alphagenome.models import dna_model as dna_model_base
    from alphagenome_research.model import dna_model
    import jax

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model (shared by both modes) ─────────────────────────────────────
    try:
        device = jax.devices("gpu")[int(args.device.replace("cuda:", ""))]
    except Exception:
        log.warning("GPU not found — falling back to CPU")
        device = jax.devices("cpu")[0]

    log.info(f"Device       : {device}")
    log.info(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    model = dna_model.create(
        args.model_path,
        organism_settings={
            dna_model_base.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
                fasta_path=args.fasta_file
            ),
        },
        device=device,
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    log.info("Building embedding apply fn ...")
    t0 = time.time()
    apply_fn_emb = build_embedding_apply_fn(model._metadata)
    log.info(f"Apply fn built in {time.time() - t0:.3f}s  (JIT traces on first call)")

    # ── Load intervals ─────────────────────────────────────────────────────────
    intervals_df = build_intervals_table(args.bed_file)
    log.info(f"Intervals    : {len(intervals_df)} total")

    # ═════════════════════════════════════════════════════════════════════════
    # REFERENCE MODE
    # ═════════════════════════════════════════════════════════════════════════
    if args.mode == "reference":
        out_file = Path(args.output_dir) / f"reference_emb_shard{args.shard_id}.h5"
        if args.skip_existing and out_file.exists():
            log.info(f"Skipping — {out_file} already exists")
            return

        shard_df = intervals_df.iloc[args.shard_id::args.n_shards].reset_index(drop=True)
        log.info(f"Shard        : {args.shard_id}/{args.n_shards} → {len(shard_df)} intervals")

        log.info("Extracting reference sequences ...")
        t0 = time.time()
        sequences_dict = build_reference_sequences(shard_df, args.fasta_file)
        log.info(f"Sequences built in {time.time() - t0:.1f}s")

        log.info(f"Extracting embeddings → {out_file}")
        t0 = time.time()
        result = extract_reference_embeddings_and_save(
            model=model,
            apply_fn_emb=apply_fn_emb,
            sequences_dict=sequences_dict,
            intervals_df=shard_df,
            output_dir=args.output_dir,
            shard_id=args.shard_id,
            aggregate=args.aggregate,
            output_types=args.output_types,
            window_size=131072,
        )
        _log_result(result, time.time() - t0, args.output_dir,
                    f"failed_emb_shard{args.shard_id}.txt")

    # ═════════════════════════════════════════════════════════════════════════
    # PERSONALIZED MODE
    # ═════════════════════════════════════════════════════════════════════════
    else:
        import pandas as pd

        all_samples = pd.read_csv(args.samples_file, header=None)[0].tolist()
        log.info(f"Samples      : {len(all_samples)} total")

        # Contiguous chunk for this task
        chunk_size   = math.ceil(len(all_samples) / args.n_shards)
        start        = args.shard_id * chunk_size
        end          = min(start + chunk_size, len(all_samples))
        task_samples = all_samples[start:end]
        log.info(f"Shard        : {args.shard_id}/{args.n_shards} → "
                 f"samples [{start}:{end}]  ({len(task_samples)} samples)")

        # Pre-build intervals_by_chrom (re-used for every sample in this chunk)
        intervals_by_chrom = {}
        for _, row in intervals_df.iterrows():
            intervals_by_chrom.setdefault(row.chrom, []).append(row.to_dict())

        for s_idx, sample_id in enumerate(task_samples):
            out_file = Path(args.output_dir) / f"{sample_id}_emb.h5"

            if args.skip_existing and out_file.exists():
                log.info(f"[{s_idx+1}/{len(task_samples)}] Skipping {sample_id} — exists")
                continue

            log.info(f"[{s_idx+1}/{len(task_samples)}] Building sequences for {sample_id} ...")
            t0 = time.time()
            sequences_dict = build_personalized_sequences_for_sample(
                sample_id        = sample_id,
                intervals_by_chrom = intervals_by_chrom,
                fasta_file       = args.fasta_file,
                vcf_dir          = args.vcf_dir,
                vcf_pattern      = args.vcf_pattern,
            )
            log.info(f"  Sequences built in {time.time() - t0:.1f}s")

            log.info(f"  Extracting embeddings → {out_file}")
            t0 = time.time()
            result = extract_personalized_embeddings_and_save(
                model=model,
                apply_fn_emb=apply_fn_emb,
                sample_id=sample_id,
                sequences_dict=sequences_dict,
                intervals_df=intervals_df,
                output_dir=args.output_dir,
                aggregate=args.aggregate,
                output_types=args.output_types,
                window_size=131072,
            )
            _log_result(result, time.time() - t0, args.output_dir,
                        f"failed_{sample_id}.txt")


def _log_result(result, elapsed, output_dir, failed_filename):
    n_ok  = result["n_ok"]
    n_err = len(result["errors"])
    log.info(
        f"Done  ok={n_ok}  errors={n_err}  "
        f"total={elapsed/60:.1f}min  "
        f"({elapsed/max(n_ok, 1):.2f}s/interval)"
    )
    for err in result["errors"]:
        log.warning(f"  {err['interval_id']}: {err['error'][:120]}")

    if result["errors"]:
        failed_file = Path(output_dir) / failed_filename
        failed_file.write_text(
            "\n".join(f"{e['interval_id']}\t{e['error']}" for e in result["errors"]) + "\n"
        )
        log.warning(f"Failed intervals written to {failed_file}")


if __name__ == "__main__":
    main()
