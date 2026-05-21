import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import itertools, tqdm
import gtfparse
import polars as pl
from cyvcf2 import VCF
# import kipoiseq
import pybedtools
import pickle 

import sys
sys.path.append('/beagle3/haky/users/temi/projects/alphagenome/src')

import personalization_modules

# import argparse

# needed arguments 
# parser = argparse.ArgumentParser()
# parser.add_argument("--metadata_file", help="Path to the metadata file", type=str)
# parser.add_argument("--agg_types", nargs='+', help='<Required> aggregation type to be used', required=True)
# parser.add_argument("--output_directory", help='<Required> folder where aggregation will be saved', required=True)
# parser.add_argument("--hpc", required=True)
# parser.add_argument("--parsl_executor", required=True)
# parser.add_argument("--delete_enformer_outputs", action='store_true')
# args = parser.parse_args()


# I want to predict for these 4 genes 
# target_genes = ['ERAP2', 'ERAP1', 'NUDT2', 'PEX6']

geuvadis_rpkm = pd.read_table('/beagle3/haky/users/temi/projects/alphagenome/files/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt')
geuvadis_ids = list(geuvadis_rpkm.columns[4:])

print(f"INFO - Found {len(geuvadis_ids)} individuals")

target_genes_metadata = pl.read_csv("/beagle3/haky/users/temi/projects/alphagenome/files/genes_metadata.tsv", separator='\t')

target_genes = target_genes_metadata.get_column('external_gene_name').to_list()
print(f"INFO - Found {len(target_genes)} genes")

fasta_file = "/project2/haky/Data/hg_sequences/hg38/Homo_sapiens_assembly38.fasta"

samples = geuvadis_ids #['HG00157', 'HG00171']

for g, gene in enumerate(target_genes):
    # create the dictionary of variants for a gene
    chromosome = target_genes_metadata.item(g, 'chrom')
    gene_start = target_genes_metadata.item(g, 'expanded_start')
    print(f"INFO - Processing {gene} on {chromosome}")
    
    # one way to get DNA sequence quickly from fasta file
    gene_dna_sequence = pybedtools.BedTool.seq(target_genes_metadata.item(g, 'expanded_coord'), fasta_file)

    chromosome_vcf = VCF(f"/project2/haky/Data/1000G/vcf_snps_only/ALL.{chromosome}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz", samples = samples)

    print(f"INFO - Found this vcf file /project2/haky/Data/1000G/vcf_snps_only/ALL.{chromosome}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz")

    var_dict = dict()
    query = f"{chromosome}:{target_genes_metadata.item(g, 'expanded_start')}-{target_genes_metadata.item(g, 'expanded_end')}" # {'chr': coord_interval.chr, 'start': coord_interval.start, 'end': coord_interval.end}
    var_dict['positions'] = tuple(variant.POS for variant in chromosome_vcf(query))

    for i, sample in enumerate(samples):
        if sample in chromosome_vcf.samples:
            try:
                var_dict[sample] = [tuple(variant.gt_bases[i].split('|')) for variant in chromosome_vcf(query)]
            except IndexError:
                print(f"ERROR - with {sample} at base index {i}")
                continue

    chromosome_vcf.close()

    samples_mappings = personalization_modules.create_personalized_mappings(samples = samples, gene_start = gene_start, var_dict=var_dict, dna_sequence=gene_dna_sequence)

    st = time.time()
    p_sequences = personalization_modules.create_personalized_sequences(samples_mappings = samples_mappings, reference_dna_sequence=gene_dna_sequence, gene_start=gene_start)
    en = time.time()

    print(f"INFO - Time taken to process {en - st}")

    with open(f'/beagle3/haky/users/temi/projects/alphagenome/files/{gene}.personalized_sequences.geuvadis.pkl', 'wb') as f:
        pickle.dump(p_sequences, f)

print(f"INFO - Finished.")