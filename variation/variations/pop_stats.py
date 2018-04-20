
import numpy

from variation.variations.filters import SampleFilter, FLT_VARS
from variation import GT_FIELD, SNPS_PER_CHUNK

STAT_FUNCTION_METADATA = {'calc_number_of_alleles': {'required_fields': [GT_FIELD],
                                                     'stat_name': 'number_of_alleles'},
                          'calc_number_of_private_alleles': {'required_fields': [GT_FIELD],
                                                             'stat_name': 'number_of_private_alleles'}}


def calc_number_of_alleles(variations, populations=None, pop_sample_filters=None):
    if pop_sample_filters is None:
        pop_sample_filters = {pop_id: SampleFilter(pop_samples) for pop_id, pop_samples in populations.items()}

    num_alleles_per_snp = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]
        num_of_allele_counts_per_allele_and_snp = vars_for_pop.allele_count
        if num_of_allele_counts_per_allele_and_snp is None:
            num_of_allele_counts_per_allele_and_snp = numpy.zeros((variations.num_variations, 1),
                                                                  dtype=int)
        chunk_num_alleles_per_snp = numpy.sum(num_of_allele_counts_per_allele_and_snp != 0, axis=1)
        num_alleles_per_snp[pop_id] = chunk_num_alleles_per_snp
    return num_alleles_per_snp


def wip_calc_number_of_private_alleles(variations, populations=None, pop_sample_filters=None):
    pass


def calc_pop_stats(variations, populations, pop_stat_functions,
                   chunk_size=SNPS_PER_CHUNK):

    pop_sample_filters = {pop_id: SampleFilter(pop_samples) for pop_id, pop_samples in populations.items()}

    pop_stat_functions = {funct.__name__: funct for funct in pop_stat_functions}

    kept_fields = set([req_field for funct in pop_stat_functions for req_field in STAT_FUNCTION_METADATA[funct]['required_fields']])
    chunks = variations.iterate_chunks(kept_fields=kept_fields,
                                       chunk_size=chunk_size)
    results_per_stat = {}

    for chunk in chunks:
        for funct_name, funct in pop_stat_functions.items():
            stat_per_pop = funct(chunk, pop_sample_filters=pop_sample_filters)

            if funct_name in results_per_stat:
                accumulated_results_per_pop_so_far = results_per_stat[funct_name]
                accumulated_results = {}
                for pop_id, pop_result in stat_per_pop.items():
                    accumulated_results_so_far = accumulated_results_per_pop_so_far[pop_id]
                    accumulated_results[pop_id] = numpy.append(accumulated_results_so_far,
                                                               pop_result)

            else:
                accumulated_results = stat_per_pop
            results_per_stat[funct_name] = accumulated_results
    return results_per_stat
