import os
import time

import h5py
import numpy as np

from .intervals import WINDOW_SIZE

PRED_ATTR_MAP = {
    "ATAC":         "atac",
    "CAGE":         "cage",
    "CHIP_TF":      "chip_tf",
    "CHIP_HISTONE": "chip_histone",
    "DNASE":        "dnase",
    "RNA_SEQ":      "rna_seq",
    "PROCAP":       "procap",
}


def slice_prediction(arr, orig_start, orig_end, padded_start,
                     window_size=WINDOW_SIZE, pad_bins=0):
    """
    Slice model output back to the original interval.

    Binned outputs (e.g. CHIP_TF): always returns at least 2 bins (the two middle
    bins straddling the TSS), then expands by pad_bins on each side.
    Nucleotide-resolution outputs: pad_bins treated as bp.
    """
    n_rows = arr.shape[0]
    offset = orig_start - padded_start
    length = orig_end   - orig_start
    if n_rows == window_size:
        start_idx = max(0, offset - pad_bins)
        end_idx   = min(n_rows, offset + length + pad_bins)
        return arr[start_idx:end_idx, :]
    else:
        bin_size  = window_size // n_rows
        bin_start = offset // bin_size
        bin_end   = (offset + length + bin_size - 1) // bin_size
        if bin_end - bin_start < 2:
            bin_end = bin_start + 2
        bin_start = max(0, bin_start - pad_bins)
        bin_end   = min(n_rows, bin_end + pad_bins)
        return arr[bin_start:bin_end, :]


def aggregate_prediction(arr, orig_start, orig_end, padded_start,
                         window_size=WINDOW_SIZE, pad_bins=0, method="mean"):
    """Slice then reduce across bins. Returns 1D array of shape (n_tracks,)."""
    sliced = slice_prediction(arr, orig_start, orig_end, padded_start, window_size, pad_bins)
    if method == "mean":
        return sliced.mean(axis=0)
    elif method == "sum":
        return sliced.sum(axis=0)
    else:
        raise ValueError(f"method must be 'mean' or 'sum', got '{method}'")


def predict_and_save(
    model,
    sample_id,
    sequences_dict,
    intervals_df,
    output_dir,
    requested_output_names,
    output_type_map,
    window_size=WINDOW_SIZE,
    pad_bins=0,
    method="mean",
    aggregate=True,
    full_output=False,
    profile=False,
):
    """
    Run predictions for all intervals of one sample and write HDF5.

    HDF5 structure:
        {sample_id}.h5
        ├── attrs: sample_id, window_size, output_types, pad_bins, method, aggregate, full_output
        └── {interval_id}/
            ├── attrs: chrom, start, end
            ├── haplotype1/  {output_name: shape depends on mode — see below}
            └── haplotype2/

    Output shape per dataset:
        full_output=True  → (n_bins, n_tracks)  raw model output, no slicing
        aggregate=True    → (n_tracks,)          mean/sum across TSS bins
        aggregate=False   → (n_bins, n_tracks)   sliced to TSS ± pad_bins

    When profile=True, returns a 'timings' list with per-interval inference
    and HDF5 write times.
    """
    requested_outputs = [output_type_map[n] for n in requested_output_names]
    output_file = os.path.join(output_dir, f"{sample_id}.h5")
    errors  = []
    timings = []   # populated only when profile=True

    with h5py.File(output_file, "w") as f:
        f.attrs["sample_id"]    = sample_id
        f.attrs["window_size"]  = window_size
        f.attrs["output_types"] = ",".join(requested_output_names)
        f.attrs["pad_bins"]     = pad_bins
        f.attrs["method"]       = method
        f.attrs["aggregate"]    = aggregate
        f.attrs["full_output"]  = full_output

        for _, row in intervals_df.iterrows():
            interval_id  = row["interval_id"]
            chrom        = row["chrom"]
            orig_start   = int(row["start"])
            orig_end     = int(row["end"])
            padded_start = int(row["padded_start"])

            if interval_id not in sequences_dict:
                errors.append({"interval_id": interval_id, "error": "missing from sequence builder"})
                continue

            seq_data = sequences_dict[interval_id]
            if "error" in seq_data:
                errors.append({"interval_id": interval_id, "error": seq_data["error"]})
                continue

            try:
                grp = f.create_group(interval_id)
                grp.attrs["chrom"] = chrom
                grp.attrs["start"] = orig_start
                grp.attrs["end"]   = orig_end

                for hap_name, seq in [("haplotype1", seq_data["hap1"]),
                                      ("haplotype2", seq_data["hap2"])]:
                    t_inf = time.time()
                    pred  = model.predict_sequence(
                        seq,
                        requested_outputs=requested_outputs,
                        ontology_terms=None,
                    )
                    inf_elapsed = time.time() - t_inf

                    t_write = time.time()
                    hap_grp = grp.create_group(hap_name)
                    for out_name in requested_output_names:
                        track = getattr(pred, PRED_ATTR_MAP[out_name], None)
                        if track is not None:
                            if full_output:
                                data = track.values
                            elif aggregate:
                                data = aggregate_prediction(
                                    track.values, orig_start, orig_end, padded_start,
                                    window_size, pad_bins, method,
                                )
                            else:
                                data = slice_prediction(
                                    track.values, orig_start, orig_end, padded_start,
                                    window_size, pad_bins,
                                )
                            hap_grp.create_dataset(out_name, data=np.array(data), compression="gzip")
                    write_elapsed = time.time() - t_write

                    if profile:
                        timings.append({
                            "event":       "inference",
                            "sample_id":   sample_id,
                            "interval_id": interval_id,
                            "haplotype":   hap_name,
                            "elapsed_s":   round(inf_elapsed, 4),
                        })
                        timings.append({
                            "event":       "hdf5_write",
                            "sample_id":   sample_id,
                            "interval_id": interval_id,
                            "haplotype":   hap_name,
                            "elapsed_s":   round(write_elapsed, 4),
                        })

            except Exception as exc:
                errors.append({"interval_id": interval_id, "error": str(exc)})

    return {"sample_id": sample_id, "output_file": output_file,
            "errors": errors, "timings": timings}
