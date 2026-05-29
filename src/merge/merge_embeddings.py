"""
merge_embeddings.py
-------------------
Merge per-shard embedding HDF5 files produced by predict_embeddings.py into a
single file.

Each shard HDF5 has the structure:
    /{interval_id}/
        attrs: chrom, start, end
        embeddings_128bp : ndarray  (3072,) or (1024, 3072)
        embeddings_1bp   : ndarray  (1536,) or (131072, 1536)   [optional]

Output formats
--------------
hdf5    : merged.h5 with the same group/dataset structure as the shards.
parquet : one parquet file per embedding type found in the shards.
          Columns: chrom, start, end, [emb_0 … emb_{D-1}]
          Index  : interval_id

Usage
-----
    python merge_embeddings.py \\
        --h5-dir /path/to/shard_dir \\
        --output /path/to/merged_embeddings.h5 \\
        [--format hdf5|parquet]
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shard_files(h5_dir: Path) -> list[Path]:
    """Return all reference_emb_shard*.h5 files sorted by shard index."""
    files = sorted(
        h5_dir.glob("reference_emb_shard*.h5"),
        key=lambda p: int(re.search(r"shard(\d+)", p.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No reference_emb_shard*.h5 files found in {h5_dir}")
    return files


# ── Merge → HDF5 ─────────────────────────────────────────────────────────────

def merge_to_hdf5(shard_files: list[Path], output_path: Path):
    """Concatenate all shard HDF5 files into a single HDF5."""
    n_intervals = 0
    with h5py.File(output_path, "w") as fout:
        for shard_path in shard_files:
            with h5py.File(shard_path, "r") as fin:
                for iid in fin.keys():
                    if iid in fout:
                        log.warning(f"Duplicate interval_id '{iid}' — skipping")
                        continue
                    grp = fout.create_group(iid)
                    grp.attrs.update(fin[iid].attrs)
                    for ds_name in fin[iid].keys():
                        data = fin[iid][ds_name][()]
                        grp.create_dataset(ds_name, data=data, compression="gzip")
                    n_intervals += 1
        # Copy file-level attributes from the first shard (minus shard_id)
        with h5py.File(shard_files[0], "r") as fin:
            for k, v in fin.attrs.items():
                if k != "shard_id":
                    fout.attrs[k] = v
        fout.attrs["n_intervals"] = n_intervals
    log.info(f"Written {output_path}  ({n_intervals} intervals)")


# ── Merge → Parquet ───────────────────────────────────────────────────────────

def _detect_embedding_types(shard_files: list[Path]) -> list[str]:
    """Return the list of embedding dataset names present in the shards."""
    with h5py.File(shard_files[0], "r") as f:
        # inspect the first interval group
        for iid in f.keys():
            return list(f[iid].keys())
    return ["embeddings_128bp"]


def merge_to_parquet(shard_files: list[Path], output_path: Path):
    """
    Build a pandas DataFrame per embedding type and write as parquet.

    For each embedding type found in the shards, writes one parquet file:
        <output_path_stem>.<type>.parquet
        e.g. merged_embeddings.embeddings_128bp.parquet
             merged_embeddings.embeddings_1bp.parquet

    Columns: chrom, start, end, emb_0 … emb_{D-1}
    Index:   interval_id
    """
    import pandas as pd

    emb_types = _detect_embedding_types(shard_files)
    log.info(f"Embedding types found: {emb_types}")

    for emb_type in emb_types:
        records = []
        for shard_path in shard_files:
            with h5py.File(shard_path, "r") as f:
                for iid in f.keys():
                    grp = f[iid]
                    if emb_type not in grp:
                        continue
                    chrom = grp.attrs["chrom"]
                    start = int(grp.attrs["start"])
                    end   = int(grp.attrs["end"])
                    emb   = grp[emb_type][()]
                    if emb.ndim == 2:
                        emb = emb.mean(axis=0)  # fallback: mean if not pre-aggregated
                    records.append((iid, chrom, start, end, emb))

        if not records:
            log.warning(f"No intervals found for {emb_type} — skipping")
            continue

        iids, chroms, starts, ends, embs = zip(*records)
        emb_arr  = np.stack(embs).astype(np.float32)
        D        = emb_arr.shape[1]
        emb_cols = [f"emb_{i}" for i in range(D)]

        df = pd.DataFrame(emb_arr, columns=emb_cols, index=list(iids))
        df.insert(0, "chrom", list(chroms))
        df.insert(1, "start", list(starts))
        df.insert(2, "end",   list(ends))
        df.index.name = "interval_id"

        # deduplicate (keep first)
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < n_before:
            log.warning(f"[{emb_type}] Dropped {n_before - len(df)} duplicate interval_id(s)")

        # Write alongside the HDF5 path, tagged with the embedding type
        out = output_path.with_suffix("").with_suffix(f".{emb_type}.parquet")
        df.to_parquet(out)
        log.info(f"Written {out}  ({len(df)} intervals × {D} dims)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge AlphaGenome embedding shards")
    p.add_argument("--h5-dir",  required=True,
                   help="Directory containing reference_emb_shard*.h5 files")
    p.add_argument("--output",  required=True,
                   help="Output file path (.h5 or .parquet)")
    p.add_argument("--format",  choices=["hdf5", "parquet"], default="hdf5",
                   help="Output format (default: hdf5)")
    return p.parse_args()


def main():
    args   = parse_args()
    h5_dir = Path(args.h5_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    shard_files = _shard_files(h5_dir)
    log.info(f"Found {len(shard_files)} shard file(s) in {h5_dir}")

    if args.format == "hdf5":
        merge_to_hdf5(shard_files, output)
    else:
        merge_to_parquet(shard_files, output)


if __name__ == "__main__":
    main()
