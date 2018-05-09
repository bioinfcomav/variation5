
import functools
import os
from collections import defaultdict
import math
import itertools

import numpy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from variation.variations.filters import SampleFilter, FLT_VARS
from variation import (GT_FIELD, SNPS_PER_CHUNK, MISSING_INT,
                       MIN_NUM_GENOTYPES_FOR_POP_STAT, DP_FIELD,
                       MIN_CALL_DP_FOR_HET)
from variation.matrix.stats import counts_and_allels_by_row
from variation.variations.stats import calc_maf as calc_maf_in_pop
from variation.variations.stats import calc_obs_het as calc_obs_het_in_pop
from variation.variations.stats import (calc_unbias_expected_het,
                                        _mask_stats_with_few_samples,
                                   7     calc_called_gt,
                                        calc_allele_freq)

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
                           pop_sample_filters=None,
                           min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    num_alleles_per_snp = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]

        allele_counts_per_allele_and_snp = _count_alleles_per_allele_and_snp(vars_for_pop)

        chunk_num_alleles_per_snp = numpy.sum(allele_counts_per_allele_and_snp != 0, axis=1)
        chunk_num_alleles_per_snp = _mask_stats_with_few_samples(chunk_num_alleles_per_snp,
                                                                 vars_for_pop,
                                                                 min_num_genotypes,
                                                                 masking_value=0)
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
                                   pop_sample_filters=None,
                                   min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    pop_sample_filters = _prepare_pop_sample_filters(populations,
                                                     pop_sample_filters)

    if len(pop_sample_filters) < 2:
        raise ValueError('At least two populations are required')

    different_alleles = numpy.sort(numpy.unique(variations[GT_FIELD]))
    if different_alleles[0] == MISSING_INT:
        different_alleles = different_alleles[1:]

    tot_num_called_gts = calc_called_gt(variations, rates=False)

    allele_counts_per_pop_per_allele_and_snp = {}
    not_enough_genotypes_masks = {}
    for pop_id, pop_sample_filter in pop_sample_filters.items():
        vars_for_pop = pop_sample_filter(variations)[FLT_VARS]
        allele_counts = _count_alleles_per_allele_and_snp(vars_for_pop,
                                                          different_alleles)
        if min_num_genotypes:
            num_called_gts_in_pop = calc_called_gt(vars_for_pop, rates=False)
            num_called_gts_in_other_pops = tot_num_called_gts - num_called_gts_in_pop
            mask = numpy.logical_or(num_called_gts_in_pop < min_num_genotypes,
                                    num_called_gts_in_other_pops < min_num_genotypes)
            not_enough_genotypes_masks[pop_id] = mask
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

        private_alleles_for_pop = numpy.sum(alleles_present_in_this_pop_not_in_others, axis=1)

        if min_num_genotypes:
            mask = not_enough_genotypes_masks[pop_id]
            private_alleles_for_pop[mask] = 0

        private_alleles[pop_id] = private_alleles_for_pop
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
                if ((kwarg == 'min_call_dp' and kwargs['min_call_dp']) or
                    (kwarg != 'min_call_dp' and kwarg in funct_metadata['optional_fields'])):
                    kept_fields.update(funct_metadata['optional_fields'][kwarg])

    chunks = variations.iterate_chunks(kept_fields=kept_fields,
                                       chunk_size=chunk_size)
    results_per_stat = {}

    for chunk in chunks:
        for funct_name, funct_info in pop_stat_functions.items():
            stat_name = STAT_FUNCTION_METADATA[funct_name]['stat_name']
            funct = funct_info['function']
            stat_per_pop = funct(chunk, pop_sample_filters=pop_sample_filters)

            if stat_name in results_per_stat:
                accumulated_results_per_pop_so_far = results_per_stat[stat_name]
                accumulated_results = {}
                for pop_id, pop_result in stat_per_pop.items():
                    accumulated_results_so_far = accumulated_results_per_pop_so_far[pop_id]
                    accumulated_results[pop_id] = numpy.append(accumulated_results_so_far,
                                                               pop_result)

            else:
                accumulated_results = stat_per_pop

            results_per_stat[stat_name] = accumulated_results
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


def _format_num(num):
    if isinstance(num, int):
        return str(num)
    elif isinstance(num, str):
        return num
    else:
        return '{:.2f}'.format(num)


def _draw_pop_stat_violins(pop_stats, plot_fpath, ylimits):

    stats = sorted(pop_stats.keys())
    pop_names = sorted(set(pop for pop_stat in pop_stats.values() for pop in pop_stat))

    size = 3
    fig = Figure(figsize=(2 * size * len(pop_names) // 4 , (len(pop_stats) - 1) * 1.5 * size))
    FigureCanvas(fig) # Don't remove it or savefig will fail later

    xtick_pos = [idx + 1 for idx in range(len(pop_names))]

    first_axes = None
    for stat_idx, stat_name in enumerate(stats):
        values_per_snp_per_pop = pop_stats[stat_name]
        subplot_position = len(pop_stats), 1, len(pop_stats) - stat_idx

        if first_axes is None:
            axes = fig.add_subplot(*subplot_position)
            first_axes = axes
            axes.set_xticklabels(pop_names, rotation='vertical')
            axes.set_xticks(xtick_pos)
        else:
            axes = fig.add_subplot(*subplot_position, sharex=first_axes)
            axes.tick_params(axis='x', which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

        if stat_name in ylimits:
            axes.set_ylim(**ylimits[stat_name])
        values_per_snp_for_pops = [values_per_snp_per_pop[pop] for pop in pop_names]
        axes.violinplot(values_per_snp_for_pops)

        y_label = stat_name.replace('_', '\n')
        axes.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(plot_fpath)


def _draw_pop_stat_bars(pop_means, plot_fpath):
    stats = sorted(pop_means.keys())
    pop_names = sorted(set(pop for pop_stat in pop_means.values() for pop in pop_stat))

    size = 3
    fig = Figure(figsize=(2 * size * len(pop_names) // 4 , (len(pop_means) - 1) * 1.5 * size))
    FigureCanvas(fig) # Don't remove it or savefig will fail later

    xtick_pos = [idx + 1 for idx in range(len(pop_names))]

    first_axes = None
    for stat_idx, stat_name in enumerate(stats):
        means_per_pop = pop_means[stat_name]
        subplot_position = len(pop_means), 1, len(pop_means) - stat_idx

        if first_axes is None:
            axes = fig.add_subplot(*subplot_position)
            first_axes = axes
            axes.set_xticklabels(pop_names, rotation='vertical')
            axes.set_xticks(xtick_pos)
        else:
            axes = fig.add_subplot(*subplot_position, sharex=first_axes)
            axes.tick_params(axis='x', which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

        means = [means_per_pop[pop] for pop in pop_names]
        axes.bar(xtick_pos, means)

        if stat_name == 'number_of_alleles':
            _, top = axes.get_ylim()
            axes.set_ylim(bottom=1, top=top)

        y_label = stat_name.replace('_', '\n')
        if stat_name in ('almost_fixed', 'highly_variable'):
            y_label = '% ' + y_label
        axes.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(plot_fpath)


def create_pop_stats_report(variations, populations, out_dir_fpath,
                            min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                            min_call_dp_for_obs_het=MIN_CALL_DP_FOR_HET,
                            violin_ylimits=None):
    funct1 = functools.partial(calc_number_of_alleles,
                               min_num_genotypes=min_num_genotypes)
    funct2 = functools.partial(calc_number_of_private_alleles,
                               min_num_genotypes=min_num_genotypes)
    funct3 = functools.partial(calc_major_allele_freq,
                               min_num_genotypes=min_num_genotypes)
    funct4 = functools.partial(calc_obs_het,
                               min_num_genotypes=min_num_genotypes,
                               min_call_dp=min_call_dp_for_obs_het)
    funct5 = functools.partial(calc_exp_het,
                               min_num_genotypes=min_num_genotypes)
    pop_stat_functions = [funct1, funct2, funct3, funct4, funct5]

    pop_stats = calc_pop_stats(variations, populations,
                               pop_stat_functions)

    sep = '\t'

    stats_csv_fpath = os.path.join(out_dir_fpath, 'pop_stats.csv')
    stats_csv = open(stats_csv_fpath, 'wt')

    stats_csv.write(sep)
    items_to_write = []
    for stat_name in pop_stats:
        items_to_write.extend([stat_name.replace('_', ' ')] * 6)
    stats_csv.write(sep.join(items_to_write))
    stats_csv.write('\n')

    stats_csv.write('Populations')
    stats_csv.write(sep)
    items_to_write = []
    for _ in pop_stats:
        items_to_write.extend(['mean', 'min', 'q25', 'median', 'q75', 'max'])
    stats_csv.write(sep.join(items_to_write))
    stats_csv.write('\n')

    pop_names = sorted(populations.keys())
    means = defaultdict(dict)
    maf_stats = defaultdict(dict)
    for pop in pop_names:
        items_to_write = [pop]
        for stat_name in pop_stats:
            values_for_pop_per_snp = pop_stats[stat_name][pop]
            min_, q25, median, q75, max_ = numpy.nanpercentile(values_for_pop_per_snp,
                                                               [0, 25, 50, 75, 100])
            mean = numpy.nanmean(values_for_pop_per_snp)
            items_to_write.extend([mean, min_, q25, median, q75, max_])
            if stat_name != 'major_allele_freq':
                means[stat_name][pop] = mean
            else:
                num_values = numpy.count_nonzero(~numpy.isnan(values_for_pop_per_snp))
                almost_fixed = numpy.sum(values_for_pop_per_snp >= 0.95) / num_values * 100
                highly_variable = numpy.sum(values_for_pop_per_snp < 0.7) / num_values * 100
                maf_stats['almost_fixed'][pop] = almost_fixed
                maf_stats['highly_variable'][pop] = highly_variable
        stats_csv.write(sep.join([_format_num(num) for num in items_to_write]))
        stats_csv.write('\n')
    stats_csv.close()

    violins_fpath = os.path.join(out_dir_fpath, 'pop_stats_violin_plots.svg')
    _draw_pop_stat_violins(pop_stats, violins_fpath, violin_ylimits)
    bars_fpath = os.path.join(out_dir_fpath, 'pop_stats_bar_plots.svg')
    _draw_pop_stat_bars(means, bars_fpath)
    bars_fpath = os.path.join(out_dir_fpath, 'pop_maf_stats_bar_plots.svg')
    _draw_pop_stat_bars(maf_stats, bars_fpath)
