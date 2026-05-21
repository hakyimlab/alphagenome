import pandas as pd

WINDOW_SIZE = 131072   # 128 KB; 1024 bins of 128 bp


def pad_interval_to_window(chrom, start, end, window_size=WINDOW_SIZE):
    interval_len = end - start
    if interval_len > window_size:
        raise ValueError(
            f"{chrom}:{start}-{end} ({interval_len} bp) exceeds window {window_size} bp"
        )
    pad_total    = window_size - interval_len
    pad_left     = pad_total // 2
    padded_start = max(0, start - pad_left)
    padded_end   = padded_start + window_size
    return int(padded_start), int(padded_end)


def build_intervals_table(bed_file, window_size=WINDOW_SIZE):
    df = pd.read_csv(
        bed_file, sep="\t",
        usecols=['chrom', 'start', 'end'],
        dtype={"chrom": str, "start": int, "end": int},
    )
    df["interval_id"] = (
        df["chrom"] + "_" + df["start"].astype(str) + "_" + df["end"].astype(str)
    )
    padded = df.apply(
        lambda r: pad_interval_to_window(r.chrom, r.start, r.end, window_size), axis=1
    )
    df["padded_start"] = [p[0] for p in padded]
    df["padded_end"]   = [p[1] for p in padded]
    return df[["interval_id", "chrom", "start", "end", "padded_start", "padded_end"]]
