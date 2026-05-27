import os
from parsl.app.app import python_app


@python_app
def build_sequences_for_sample(
    sample_id,
    intervals_by_chrom,   # dict: {chrom: [record, ...]}
    fasta_file,
    vcf_dir,
    vcf_pattern,
):
    """
    Parsl I/O task: build phased haplotype sequences for one sample.

    Intervals are grouped by chromosome so each VCF is opened once per chrom.
    Returns: {interval_id: {'hap1': str, 'hap2': str} | {'error': str}}
    """
    import os
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
        vcf_open = os.path.exists(vcf_path)

        vcf = pysam.VariantFile(vcf_path) if vcf_open else None

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
