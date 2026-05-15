


def create_personalized_mappings(samples, gene_start, var_dict, dna_sequence):

    # you need to check the allele at the same position in the reference
    dna_sequence_list = list(dna_sequence)

    samples_dict = {}
    for sample in samples:
        if sample in list(var_dict.keys()):
            samples_dict[sample] = dict()
            samples_dict[sample]['position'] = list()
            samples_dict[sample]['alleles'] = list()
            zipped_data = zip(var_dict['positions'], var_dict[sample])
            for pos, alleles in zipped_data:
                num_alleles = set(alleles) # the alleles
                condition_1 = (len(num_alleles) != 1) # are they different alleles? (A,A)->False, (A,T)->True
                if not condition_1: # check if this allele is the reference or alternate
                    condition_2_index = (pos - gene_start)
                    #print(condition_2_index)
                    if condition_2_index >= 0:
                        try:
                            condition_2 = ((next(iter(num_alleles)) != dna_sequence_list[condition_2_index])) # are they alternate alleles?
                        except IndexError as ie:
                            print(f"ERROR - gene start {gene_start}, {len(dna_sequence_list)}; {condition_2_index}")
                if condition_1 or condition_2: 
                    samples_dict[sample]['position'].append(pos)
                    samples_dict[sample]['alleles'].append(alleles)
    return(samples_dict)



    # this function creates personalized sequences (both haplotypes)
def create_personalized_sequences(samples_mappings, reference_dna_sequence, gene_start):
    import copy
    # now remove for each individual if the alt is the same as the reference
    # filter the mappings and create personalized mappings
    samples = list(samples_mappings.keys())
    sequence_list = list(reference_dna_sequence)
    personalized_sequences = dict()
    for sample in samples:
        sample_data = samples_mappings[sample]
        indices = [sample_data['position'][i] - gene_start for i in range(len(sample_data['position']))]
        personalized_sequences[sample] = list()
        for i, haplotype in enumerate(['haplotype1', 'haplotype2']):
            ref_i = copy.deepcopy(sequence_list) # i.e. the reference genome
            for j, ind in enumerate(indices):
                ref_i[ind] = sample_data['alleles'][j][i]
            ref_ind = ''.join(ref_i)
            personalized_sequences[sample].append(ref_ind)

    return(personalized_sequences)