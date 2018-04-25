
import functools

import numpy

from variation.variations.filters import SampleFilter, FLT_VARS
from variation import (GT_FIELD, SNPS_PER_CHUNK, MISSING_INT,
                       MIN_NUM_GENOTYPES_FOR_POP_STAT, DP_FIELD,
                       MIN_CALL_DP_FOR_HET)
from variation.matrix.stats import counts_and_allels_by_row
from variation.variations.stats import calc_maf as calc_maf_in_pop
from variation.variations.stats import calc_obs_het as calc_obs_het_in_pop
from variation.variations.stats import calc_unbias_expected_het

STAT_FUNCTION_METADATA = {'calc_number_of_alleles': {'required_fields': [GT_FIELD],
                                                     'stat_name': 'number_of_alleles'},
                          'calc_number_of_private_alleles': {'required_fields': [GT_FIELD],
                                                             'stat_name': 'number_of_private_alleles'},
                          'calc_major_allele_freq': {'required_fields': [GT_FIELD],
                                                     'stat_name': 'major_allele_freq'},
                          'calc_exp_het': {'required_fields': [GT_FIELD],
                                           'stat_name': 'expected_heterozigosity'},
                          'calc_obs_het': {'required_fields': [GT_FIELD],
                                           'optional_fields': {'min_call_dp': [DP_FIELD]},
                                           'stat_name': 'observed_heterozigosity'}}


def _prepare_pop_sample_filters(populations, pop_sample_filters=None):
    if pop_sample_filters is None:
        pop_sample_filters = {pop_id: SampleFilter(pop_samples) for pop_id, pop_samples in populations.items()}
    return pop_sample_filters


def calc_number_of_alleles(variations, populations=None,
                           pop_sample_filters=None):
    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    num_alleles_per_snp = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]

        allele_counts_per_allele_and_snp = _count_alleles_per_allele_and_snp(vars_for_pop)

        chunk_num_alleles_per_snp = numpy.sum(allele_counts_per_allele_and_snp != 0, axis=1)
        num_alleles_per_snp[pop_id] = chunk_num_alleles_per_snp
    return num_alleles_per_snp


def _count_alleles_per_allele_and_snp(variations, alleles=None):

    allele_counts_per_allele_and_snp = counts_and_allels_by_row(variations[GT_FIELD],
                                                                missing_value=MISSING_INT,
                                                                alleles=alleles)[0]

    if allele_counts_per_allele_and_snp is None:
        allele_counts_per_allele_and_snp = numpy.zeros((variations.num_variations, 1),
                                                        dtype=int)
    return allele_counts_per_allele_and_snp


def calc_number_of_private_alleles(variations, populations=None,
                                   pop_sample_filters=None):

    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    if len(pop_sample_filters) < 2:
        raise ValueError('At least two populations are required')

    different_alleles = numpy.sort(numpy.unique(variations[GT_FIELD]))
    if different_alleles[0] == MISSING_INT:
        different_alleles = different_alleles[1:]

    allele_counts_per_pop_per_allele_and_snp = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]
        allele_counts = _count_alleles_per_allele_and_snp(vars_for_pop,
                                                          different_alleles)
        allele_counts_per_pop_per_allele_and_snp[pop_id] = allele_counts

    private_alleles = {}
    for pop_id in pop_sample_filters:
        other_pops_allele_counts = None
        for other_pop in pop_sample_filters:
            if other_pop == pop_id:
                continue
            pop_counts = allele_counts_per_pop_per_allele_and_snp[other_pop]
            if other_pops_allele_counts is None:
                other_pops_allele_counts = pop_counts
            else:
                other_pops_allele_counts = numpy.add(other_pops_allele_counts, pop_counts)
        this_pop_allele_counts = allele_counts_per_pop_per_allele_and_snp[pop_id]

        alleles_present_in_this_pop = this_pop_allele_counts > 0
        alleles_not_present_in_other_pops = other_pops_allele_counts == 0
        alleles_present_in_this_pop_not_in_others = numpy.logical_and(alleles_present_in_this_pop,
                                                                      alleles_not_present_in_other_pops)
        private_alleles[pop_id] = numpy.sum(alleles_present_in_this_pop_not_in_others, axis=1)
    return private_alleles


def _get_original_function_name(funct):
    if isinstance(funct, functools.partial):
        original_funct = funct.func
        funct_name = original_funct.__name__
    else:
        funct_name = funct.__name__
    return funct_name


def _get_partial_funct_kwargs(funct):
    if isinstance(funct, functools.partial):
        return funct.keywords
    else:
        return []


def calc_pop_stats(variations, populations, pop_stat_functions,
                   chunk_size=SNPS_PER_CHUNK):

    pop_sample_filters = _prepare_pop_sample_filters(populations)

    pop_stat_functions_orig_list = pop_stat_functions
    pop_stat_functions = {}
    for funct in pop_stat_functions_orig_list:
        funct_name = _get_original_function_name(funct)
        pop_stat_functions[funct_name] = {'function': funct}

    kept_fields = set()
    for funct_name, funct_info in pop_stat_functions.items():
        funct_metadata = STAT_FUNCTION_METADATA[funct_name]
        kept_fields.update(funct_metadata['required_fields'])
        if 'optional_fields' in funct_metadata:
            funct = funct_info['function']
            kwargs = _get_partial_funct_kwargs(funct)
            for kwarg in kwargs:
                if kwarg in funct_metadata['optional_fields']:
                    kept_fields.update(funct_metadata['optional_fields'][kwarg])

    chunks = variations.iterate_chunks(kept_fields=kept_fields,
                                       chunk_size=chunk_size)
    results_per_stat = {}

    for chunk in chunks:
        for funct_name, funct_info in pop_stat_functions.items():
            funct = funct_info['function']
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


def calc_major_allele_freq(variations, populations=None,
                           pop_sample_filters=None,
                           min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    allele_freq = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]
        allele_freq[pop_id] = calc_maf_in_pop(vars_for_pop,
                                              min_num_genotypes=min_num_genotypes,
                                              chunk_size=None)
    return allele_freq


def calc_obs_het(variations, populations=None, pop_sample_filters=None,
                 min_call_dp=MIN_CALL_DP_FOR_HET,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    obs_het = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]
        obs_het[pop_id] = calc_obs_het_in_pop(vars_for_pop,
                                              min_call_dp=min_call_dp,
                                              min_num_genotypes=min_num_genotypes)
    return obs_het


def calc_exp_het(variations, populations=None, pop_sample_filters=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    het = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]
        het[pop_id] = calc_unbias_expected_het(vars_for_pop,
                                               min_num_genotypes=min_num_genotypes)
    return het

