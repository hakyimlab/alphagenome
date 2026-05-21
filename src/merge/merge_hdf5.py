"""
Merge AlphaGenome HDF5 prediction files.

Two modes (auto-detected from file names, or set with --mode):

  reference     reference_shard*.h5 → one merged output per output type
  personalized  {sample_id}.h5      → stacked (n_samples × n_tracks) per interval

Output formats:
  hdf5    (default) single merged .h5
  parquet one .parquet per interval (personalized) or one flat .parquet (reference)

Usage examples:
  # personalized, HDF5 output
  python merge_hdf5.py --h5-dir run/h5 --output merged.h5

  # personalized, parquet per interval
  python merge_hdf5.py --h5-dir run/h5 --output parquet_dir --format parquet

  # reference shards → single HDF5
  python merge_hdf5.py --h5-dir run_reference/h5 --output merged_ref.h5 --mode reference

  # reference shards → flat parquet
  python merge_hdf5.py --h5-dir run_reference/h5 --output ref.parquet --mode reference --format parquet
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Readers ───────────────────────────────────────────────────────────────────

def read_sample_h5(h5_path, output_types=None):
    """
    Read one personalized sample HDF5.
    Returns nested dict compatible with reference readers:
    {
        'meta': {...},
        'interval_id': {
            'coords':     {'chrom', 'start', 'end'},
            'haplotype1': {'CHIP_TF': array, ...},
            'haplotype2': {'CHIP_TF': array, ...},
        }, ...
    }
    """
    result = {}
    with h5py.File(h5_path, "r") as f:
        result["meta"] = dict(f.attrs)
        stored_types = result["meta"].get("output_types", "").split(",")
        if output_types is None:
            output_types = stored_types

        for iid in f.keys():
            grp = f[iid]
            entry = {
                "coords": {
                    "chrom": grp.attrs["chrom"],
                    "start": int(grp.attrs["start"]),
                    "end":   int(grp.attrs["end"]),
                }
            }
            for hap in ("haplotype1", "haplotype2"):
                if hap not in grp:
                    continue
                entry[hap] = {
                    ot: grp[hap][ot][()]
                    for ot in output_types
                    if ot in grp[hap]
                }
            result[iid] = entry
    return result


def read_reference_shard(h5_path, output_types=None):
    """Read one reference shard (reference_shard{N}.h5)."""
    result = {}
    with h5py.File(h5_path, "r") as f:
        result["meta"] = dict(f.attrs)
        stored_types = result["meta"].get("output_types", "").split(",")
        if output_types is None:
            output_types = stored_types

        for iid in f.keys():
            grp = f[iid]
            entry = {
                "coords": {
                    "chrom": grp.attrs["chrom"],
                    "start": int(grp.attrs["start"]),
                    "end":   int(grp.attrs["end"]),
                }
            }
            if "haplotype1" in grp:
                entry["haplotype1"] = {
                    ot: grp["haplotype1"][ot][()]
                    for ot in output_types
                    if ot in grp["haplotype1"]
                }
            result[iid] = entry
    return result


# ── Mergers ───────────────────────────────────────────────────────────────────

def merge_personalized(h5_dir, output_types=None, haplotype="sum", sample_ids=None):
    """
    Read all {sample_id}.h5 files and stack into per-interval matrices.

    Returns:
        matrices : {interval_id: np.ndarray (n_samples, n_tracks)}  per output_type
        samples  : list of sample IDs (row order)
        coords   : {interval_id: {'chrom', 'start', 'end'}}
        meta     : dict of file-level attrs from the first sample
    """
    h5_dir = Path(h5_dir)
    paths  = sorted(p for p in h5_dir.glob("*.h5")
                    if not p.stem.startswith("reference_shard"))

    if sample_ids is not None:
        paths = [h5_dir / f"{s}.h5" for s in sample_ids]

    if not paths:
        raise FileNotFoundError(f"No sample HDF5 files found in {h5_dir}")

    log.info(f"Found {len(paths)} sample file(s) in {h5_dir}")

    # determine output_types from first file if not specified
    if output_types is None:
        with h5py.File(paths[0], "r") as f:
            output_types = dict(f.attrs).get("output_types", "").split(",")

    # accumulate: {output_type: {interval_id: [array per sample]}}
    accum  = {ot: {} for ot in output_types}
    coords = {}
    sample_list = []
    first_meta  = None

    for path in paths:
        data = read_sample_h5(path, output_types=output_types)
        sid  = data["meta"].get("sample_id", path.stem)
        sample_list.append(sid)
        if first_meta is None:
            first_meta = data["meta"]

        for iid, entry in data.items():
            if iid == "meta":
                continue
            if iid not in coords:
                coords[iid] = entry["coords"]

            h1 = entry.get("haplotype1", {})
            h2 = entry.get("haplotype2", {})

            for ot in output_types:
                a1 = h1.get(ot)
                a2 = h2.get(ot)

                if haplotype == "haplotype1":
                    arr = a1
                elif haplotype == "haplotype2":
                    arr = a2
                elif haplotype == "sum":
                    arr = (a1 + a2) if (a1 is not None and a2 is not None) else (a1 or a2)
                elif haplotype == "mean":
                    arr = (a1 + a2) / 2 if (a1 is not None and a2 is not None) else (a1 or a2)
                else:
                    raise ValueError(f"Unknown haplotype mode: {haplotype}")

                if arr is not None:
                    accum[ot].setdefault(iid, []).append(arr)

        log.info(f"  read {sid}")

    # stack arrays per interval per output_type
    matrices = {
        ot: {iid: np.stack(arrs) for iid, arrs in iid_dict.items()}
        for ot, iid_dict in accum.items()
    }

    log.info(f"Merged: {len(sample_list)} samples × "
             f"{len(coords)} intervals × {output_types}")
    return matrices, sample_list, coords, first_meta or {}


def merge_reference(h5_dir, output_types=None):
    """
    Merge all reference_shard*.h5 files in shard_id order.

    Returns:
        matrices : {output_type: {interval_id: np.ndarray (n_tracks,)}}
        coords   : {interval_id: {'chrom', 'start', 'end'}}
        meta     : dict of file-level attrs from shard 0
    """
    h5_dir = Path(h5_dir)
    shard_files = sorted(
        h5_dir.glob("reference_shard*.h5"),
        key=lambda p: int(p.stem.replace("reference_shard", "")),
    )

    if not shard_files:
        raise FileNotFoundError(f"No reference_shard*.h5 files in {h5_dir}")

    log.info(f"Found {len(shard_files)} shard file(s) in {h5_dir}")

    if output_types is None:
        with h5py.File(shard_files[0], "r") as f:
            output_types = dict(f.attrs).get("output_types", "").split(",")

    accum  = {ot: {} for ot in output_types}
    coords = {}
    first_meta = None

    for path in shard_files:
        shard = read_reference_shard(path, output_types=output_types)
        if first_meta is None:
            first_meta = shard["meta"]

        for iid, entry in shard.items():
            if iid == "meta":
                continue
            if iid not in coords:
                coords[iid] = entry["coords"]
            for ot in output_types:
                arr = entry.get("haplotype1", {}).get(ot)
                if arr is not None:
                    accum[ot][iid] = arr

        log.info(f"  read {path.name}  ({len(shard) - 1} intervals)")

    log.info(f"Merged: {len(coords)} intervals × {output_types} (reference)")
    return accum, coords, first_meta or {}


# ── Writers ───────────────────────────────────────────────────────────────────

def write_hdf5_personalized(output_path, matrices, sample_list, coords, meta):
    """
    Write merged personalized data to HDF5.

    Structure:
        /{interval_id}/
          attrs: chrom, start, end
          /{output_type}: ndarray (n_samples, n_tracks)
            attrs: samples (comma-separated)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_str = ",".join(sample_list)

    with h5py.File(output_path, "w") as f:
        f.attrs["mode"]         = "personalized"
        f.attrs["samples"]      = samples_str
        f.attrs["n_samples"]    = len(sample_list)
        f.attrs["output_types"] = ",".join(matrices.keys())
        for k, v in meta.items():
            if k not in ("sample_id",):
                try:
                    f.attrs[k] = v
                except Exception:
                    pass

        # collect all interval IDs across output types
        all_iids = sorted({iid for ot_dict in matrices.values() for iid in ot_dict})

        for iid in all_iids:
            grp = f.create_group(iid)
            c = coords[iid]
            grp.attrs["chrom"] = c["chrom"]
            grp.attrs["start"] = c["start"]
            grp.attrs["end"]   = c["end"]
            for ot, iid_dict in matrices.items():
                if iid in iid_dict:
                    ds = grp.create_dataset(ot, data=iid_dict[iid], compression="gzip")
                    ds.attrs["samples"] = samples_str

    log.info(f"Written: {output_path}  "
             f"({len(sample_list)} samples, {len(all_iids)} intervals)")


def write_hdf5_reference(output_path, matrices, coords, meta):
    """
    Write merged reference data to HDF5.

    Structure:
        /{interval_id}/
          attrs: chrom, start, end
          /{output_type}: ndarray (n_tracks,)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.attrs["mode"]         = "reference"
        f.attrs["output_types"] = ",".join(matrices.keys())
        for k, v in meta.items():
            if k not in ("shard_id",):
                try:
                    f.attrs[k] = v
                except Exception:
                    pass

        all_iids = sorted({iid for ot_dict in matrices.values() for iid in ot_dict})

        for iid in all_iids:
            grp = f.create_group(iid)
            c = coords[iid]
            grp.attrs["chrom"] = c["chrom"]
            grp.attrs["start"] = c["start"]
            grp.attrs["end"]   = c["end"]
            for ot, iid_dict in matrices.items():
                if iid in iid_dict:
                    grp.create_dataset(ot, data=iid_dict[iid], compression="gzip")

    log.info(f"Written: {output_path}  ({len(all_iids)} intervals)")


def write_parquet_personalized(output_dir, matrices, sample_list):
    """
    Write one parquet per interval, each with shape (n_samples × n_tracks).
    Index = sample_id; columns = 0..n_tracks-1 (per output_type sub-directory).
    """
    output_dir = Path(output_dir)

    for ot, iid_dict in matrices.items():
        ot_dir = output_dir / ot
        ot_dir.mkdir(parents=True, exist_ok=True)

        for iid, matrix in iid_dict.items():
            df = pd.DataFrame(matrix, index=sample_list)
            df.index.name = "sample_id"
            df.to_parquet(ot_dir / f"{iid}.parquet")

    n_iids = len(next(iter(matrices.values())))
    log.info(f"Written: {output_dir}/{{output_type}}/{{interval_id}}.parquet  "
             f"({len(sample_list)} samples, {n_iids} intervals)")


def write_parquet_reference(output_path, matrices, coords):
    """
    Write one flat parquet per output_type: rows = intervals, cols = tracks.
    Index = interval_id; first three columns are chrom, start, end.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for ot, iid_dict in matrices.items():
        iids = sorted(iid_dict.keys())
        rows = [iid_dict[iid] for iid in iids]

        df = pd.DataFrame(np.stack(rows), index=iids)
        df.index.name = "interval_id"
        df.insert(0, "chrom", [coords[iid]["chrom"] for iid in iids])
        df.insert(1, "start", [coords[iid]["start"] for iid in iids])
        df.insert(2, "end",   [coords[iid]["end"]   for iid in iids])

        # one file per output_type
        if len(matrices) == 1:
            dest = output_path
        else:
            dest = output_path.with_stem(f"{output_path.stem}_{ot}")

        df.to_parquet(dest)
        log.info(f"Written: {dest}  ({len(iids)} intervals × {df.shape[1] - 3} tracks)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def detect_mode(h5_dir):
    h5_dir = Path(h5_dir)
    if list(h5_dir.glob("reference_shard*.h5")):
        return "reference"
    if list(h5_dir.glob("*.h5")):
        return "personalized"
    raise FileNotFoundError(f"No .h5 files found in {h5_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge AlphaGenome HDF5 prediction files")
    p.add_argument("--h5-dir",       required=True,
                   help="Directory containing HDF5 files to merge")
    p.add_argument("--output",       required=True,
                   help="Output path (.h5 file, or directory for parquet)")
    p.add_argument("--mode",         default=None, choices=["personalized", "reference"],
                   help="Merge mode (auto-detected if not set)")
    p.add_argument("--output-types", nargs="+", default=None,
                   help="Output types to include (default: all stored)")
    p.add_argument("--haplotype",    default="sum",
                   choices=["haplotype1", "haplotype2", "sum", "mean"],
                   help="Haplotype combination for personalized mode (default: sum)")
    p.add_argument("--format",       default="hdf5", choices=["hdf5", "parquet"],
                   help="Output format (default: hdf5)")
    p.add_argument("--samples",      default=None,
                   help="Optional file with sample IDs to include (one per line)")
    return p.parse_args()


def main():
    args = parse_args()

    mode = args.mode or detect_mode(args.h5_dir)
    log.info(f"Mode     : {mode}")
    log.info(f"H5 dir   : {args.h5_dir}")
    log.info(f"Output   : {args.output}")
    log.info(f"Format   : {args.format}")

    sample_ids = None
    if args.samples:
        sample_ids = pd.read_csv(args.samples, header=None).iloc[:, 0].tolist()
        log.info(f"Samples  : {len(sample_ids)} (from {args.samples})")

    if mode == "personalized":
        matrices, sample_list, coords, meta = merge_personalized(
            args.h5_dir,
            output_types=args.output_types,
            haplotype=args.haplotype,
            sample_ids=sample_ids,
        )
        if args.format == "hdf5":
            write_hdf5_personalized(args.output, matrices, sample_list, coords, meta)
        else:
            write_parquet_personalized(args.output, matrices, sample_list)

    else:  # reference
        matrices, coords, meta = merge_reference(
            args.h5_dir,
            output_types=args.output_types,
        )
        if args.format == "hdf5":
            write_hdf5_reference(args.output, matrices, coords, meta)
        else:
            write_parquet_reference(args.output, matrices, coords)


if __name__ == "__main__":
    main()
