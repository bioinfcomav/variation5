#!/usr/bin/env python
from os.path import join
import os
import argparse
import numpy
from pandas import *

from variation.variations.vars_matrices import VariationsH5
from variation import (SNPS_PER_CHUNK, STATS_DEPTHS, MAX_DEPTH,
                       SNP_DENSITY_WINDOW_SIZE, MIN_N_GENOTYPES,
                       MAX_N_ALLELES)
from variation.variations.stats import (_calc_stat, _MafCalculator,
                                        _MissingGTCalculator,
                                        _CalledGTCalculator,
                                        _ObsHetCalculatorBySnps,
                                        _ObsHetCalculatorBySample,
                                        calc_snp_density,
                                        calc_depth_cumulative_distribution_per_sample,
                                        calc_gq_cumulative_distribution_per_sample,
                                        _AlleleFreqCalculator,
                                        calc_expected_het,
                                        calc_inbreeding_coeficient,
                                        _is_het, _is_hom,
                                        calc_snv_density_distribution,
                                        GenotypeStatsCalculator,
                                        calc_called_gts_distrib_per_depth,
                                        calc_quality_by_depth,
                                        calculate_maf_depth_distribution,
                                        calc_allele_obs_distrib_2D,
    _remove_nans)
from variation.plot import plot_histogram, plot_pandas_barplot, plot_boxplot,\
    _print_figure, plot_barplot
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import sys


def _setup_argparse(**kwargs):
    'It prepares the command line argument parsing.'
    parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument('input', help='Input HDF5 file')
    parser.add_argument('-o', '--output_prefix', required=True,
                        help='Output files prefix')
    parser.add_argument('-n', '--chunk_size', default=SNPS_PER_CHUNK,
                        help='Number of SNPs per chunk')
    parser.add_argument('-nc', '--no_chunks', default=False,
                        action='store_true',
                        help='Do not divide HDF5 in chunks to calculate stats')
    help_msg = 'Comma separated list of depths to calculate missing gts'
    help_msg += ' distributions. Use : separator for range (ex 1:50)'
    parser.add_argument('-d', '--depths', default=STATS_DEPTHS, help=help_msg)
    help_msg = 'Max depth for depth distributions ({})'.format(MAX_DEPTH)
    parser.add_argument('-md', '--max_depth', default=MAX_DEPTH,
                        help=help_msg)
    help_msg = 'Window size to calculate SNP density ({})'
    help_msg = help_msg.format(SNP_DENSITY_WINDOW_SIZE)
    parser.add_argument('-w', '--window_size', default=SNP_DENSITY_WINDOW_SIZE,
                        help=help_msg)
    help_msg = 'Min number of called genotypes to calculate stats ({})'
    help_msg = help_msg.format(MIN_N_GENOTYPES)
    parser.add_argument('-m', '--min_n_gts', default=MIN_N_GENOTYPES,
                        help=help_msg)
    help_msg = 'Max num alleles to calculate allele frequency ({})'
    help_msg = help_msg.format(MAX_N_ALLELES)
    parser.add_argument('-ma', '--max_num_alleles', default=MAX_N_ALLELES,
                        help=help_msg)
    return parser


def _parse_args(parser):
    parsed_args = parser.parse_args()
    args = {}
    args['in_fpath'] = parsed_args.input
    args['out_prefix'] = parsed_args.output_prefix
    args['chunk_size'] = parsed_args.chunk_size
    args['by_chunk'] = not parsed_args.no_chunks
    if ':' in parsed_args.depths:
        start, stop = [int(x) for x in parsed_args.depths.split(':')]
        args['depths'] = range(start, stop)
    else:
        args['depths'] = [int(x) for x in parsed_args.depths.split(',')]
    args['max_depth'] = parsed_args.max_depth
    args['window_size'] = parsed_args.window_size
    args['min_num_genotypes'] = parsed_args.min_n_gts
    args['max_num_alleles'] = parsed_args.max_num_alleles
    return args


def main():
    description = 'Calculates basic stats of a HDF5 file'
    parser = _setup_argparse(description=description)
    args = _parse_args(parser)
    data_dir = args['out_prefix']
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    h5 = VariationsH5(args['in_fpath'], mode='r')
    mafs = _calc_stat(h5, _MafCalculator(), by_chunk=args['by_chunk'])

    # MAF distribution
    fpath = join(data_dir, 'mafs.png')
    title = 'Maximum allele frequency (MAF) distribution'
    plot_histogram(mafs, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['MAF'], 'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   range_=(0, 1))
    df_mafs = DataFrame(mafs)
    _save(fpath.strip('.png') + '.csv', df_mafs)

    # MAF (depth based) distribution
    try:
        maf_depths_distrib = calculate_maf_depth_distribution(h5)
        fpath = join(data_dir, 'mafs_depths_distribution.png')
        title = 'Depth based Maximum allele frequency (MAF) distribution'
        mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                      'set_ylabel': {'args': ['MAF (depth)'], 'kwargs': {}},
                      'set_title': {'args': [title], 'kwargs': {}}}
        if h5.samples is not None:
            mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs': {}}
        plot_boxplot(maf_depths_distrib, fhand=open(fpath, 'w'), figsize=(40, 10),
                     mpl_params=mpl_params)
        df_maf_depths_distrib = DataFrame(maf_depths_distrib)
        _save(fpath.strip('.png') + '.csv', df_maf_depths_distrib)
    except ValueError:
        sys.stderr.write('MAF depth could not be calculated\n')

    # Missing genotype rate per SNP distribution
    rates = _calc_stat(h5, _MissingGTCalculator(), by_chunk=args['by_chunk'])
    fpath = join(data_dir, 'missing_gt_rate.png')
    title = 'Missing Genotype rates per SNP distribution'
    plot_histogram(rates, bins=100, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['Missing GT rate'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}})
    df_rates = DataFrame(rates)
    _save(fpath.strip('.png') + '.csv', df_rates)

#     counts = _calc_stat(h5, _CalledGTCalculator(rate=False), by_chunk=args['by_chunk'])
#     fpath = join(data_dir, 'called_gt_counts.csv')
#     df_counts = DataFrame(counts)
#     _save(fpath.strip('.png') + '.csv', df_counts)

    # Observed heterozygosity distributions
    fpath = join(data_dir, 'obs_het.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    axes = fig.add_subplot(211)
    het = _calc_stat(h5, _ObsHetCalculatorBySnps(), by_chunk=args['by_chunk'])
    title = 'SNP observed Heterozygosity distribution'
    plot_histogram(het, bins=100, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['Heterozygosity'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   axes=axes)
    df_het = DataFrame(het)
    _save(fpath.strip('.png') + '.csv', df_het)

    axes = fig.add_subplot(212)
    het_sample = _calc_stat(h5, _ObsHetCalculatorBySample(),
                            by_chunk=args['by_chunk'])
    title = 'Sample observed Heterozygosity distribution'
    het_sample = _remove_nans(het_sample)
    plot_histogram(het_sample, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['Heterozygosity'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   axes=axes)
    fpath = join(data_dir, 'obs_het_per_sample.csv')
    df_het_sample = DataFrame(het_sample)
    _save(fpath, df_het_sample)
    canvas.print_figure(fhand)


#     gt_freq = _calc_stat(h5, _CalledGTCalculator(),
#                          by_chunk=args['by_chunk'])
#     fpath = join(data_dir, 'gt_frequency.csv')
#     df_gt_freq = DataFrame(gt_freq)
#     _save(fpath.strip('.png') + '.csv', df_gt_freq)

    # SNP density distribution
    density = calc_snp_density(h5, args['window_size'])
    fpath = join(data_dir, 'snps_density.csv')
    title = 'SNP density distribution per {} bp windows'
    title = title.format(args['window_size'])
    plot_histogram(rates, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['SNP density'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}})
    df_dens = DataFrame(density)
    _save(fpath.strip('.png') + '.csv', df_dens)

    # DP distribution per sample
    distrib_dp, cum_dp = calc_depth_cumulative_distribution_per_sample(h5,
                                                                by_chunk=args['by_chunk'],
                                                                max_depth=args['max_depth'])
    fpath = join(data_dir, 'depth_distribution_per_sample.png')
    title = 'Depth distribution per sample'
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    if h5.samples is not None:
        mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs': {}}
    plot_boxplot(distrib_dp, fhand=open(fpath, 'w'), figsize=(40, 10),
                 mpl_params=mpl_params)
    df_distrib_dp = DataFrame(distrib_dp)
    _save(fpath.strip('.png') + '.csv', df_distrib_dp)

    # DP distribution all sample
    distrib_dp_all = numpy.sum(distrib_dp, axis=0)
    fpath = join(data_dir, 'depth_distribution.png')
    title = 'Depth distribution all samples'
    plot_barplot(numpy.arange(0, distrib_dp_all.shape[0]), distrib_dp_all,
                 mpl_params={'set_xlabel': {'args': ['Depth'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 fhand=open(fpath, 'w'))

#     fpath = join(data_dir, 'depth_cumulative.csv')
#     df_cum_dp = DataFrame(cum_dp)
#     _save(fpath.strip('.png') + '.csv', df_cum_dp)

    # GQ distribution per sample
    distrib_gq, cum_gq = calc_gq_cumulative_distribution_per_sample(h5, by_chunk=args['by_chunk'])
    fpath = join(data_dir, 'gq_distribution_per_sample.png')
    title = 'Genotype Quality (QG) distribution per sample'
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': ['GQ'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    if h5.samples is not None:
        mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs': {}}
    plot_boxplot(distrib_gq, fhand=open(fpath, 'w'), figsize=(40, 10),
                 mpl_params=mpl_params)
    df_distrib_gq = DataFrame(distrib_gq)
    _save(fpath.strip('.png') + '.csv', df_distrib_gq)

    # GQ distribution all sample
    distrib_gq_all = numpy.sum(distrib_gq, axis=0)
    fpath = join(data_dir, 'gq_distribution.png')
    title = 'Genotype Quality (GQ) distribution all samples'
    plot_barplot(numpy.arange(0, distrib_gq_all.shape[0]), distrib_gq_all,
                 mpl_params={'set_xlabel': {'args': ['GQ'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 fhand=open(fpath, 'w'))

#     fpath = join(data_dir, 'gq_cumulative.csv')
#     df_cum_gq = DataFrame(cum_gq)
#     _save(fpath.strip('.png') + '.csv', df_cum_gq)

#     max_n_alleles = args['max_num_alleles']
#     allele_freq = _calc_stat(h5,
#                              _AlleleFreqCalculator(max_num_allele=max_n_alleles),
#                              by_chunk=True)
#     fpath = join(data_dir, 'allele_frequency.csv')
#     df_allele_freq = DataFrame(allele_freq)
#     _save(fpath.strip('.png') + '.csv', df_allele_freq)
#
#     exp_het = calc_expected_het(allele_freq)
#     fpath = join(data_dir, 'het_expected.csv')
#     df_exp_het = DataFrame(exp_het)
#     _save(fpath.strip('.png') + '.csv', df_exp_het)
#
#     inbreeding_coef = calc_inbreeding_coeficient(het, exp_het)
#     fpath = join(data_dir, 'inbreeding_coeficient.csv')
#     df_inbreeding_coef = DataFrame(inbreeding_coef)
#     _save(fpath.strip('.png') + '.csv', df_inbreeding_coef)

    # Depth distribution per genotype
    fpath = join(data_dir, 'depth_distribution_per_gt.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    # Heterozygous
    result = calc_depth_cumulative_distribution_per_sample(h5,
                                                           max_depth=args['max_depth'],
                                                           mask_function=_is_het,
                                                           mask_field='/calls/GT',
                                                           by_chunk=args['by_chunk'])
    distrib_dp_het, cum_dp_het = result
    distrib_dp_het = numpy.sum(distrib_dp_het, axis=0)
    title = 'Depth distribution Heterozygous'
    axes = fig.add_subplot(211)
    plot_barplot(numpy.arange(0, distrib_dp_het.shape[0]), distrib_dp_het,
                 mpl_params={'set_xlabel': {'args': ['Depth'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    fpath = join(data_dir, 'distribution_het_by_depth.csv')
    df_distrib_dp_het = DataFrame(distrib_dp_het)
    _save(fpath, df_distrib_dp_het)
#     fpath = join(data_dir, 'cumulative_het_by_depth.csv')
#     df_cum_dp_het = DataFrame(cum_dp_het)
#     _save(fpath.strip('.png') + '.csv', df_cum_dp_het)

    # Homozygous
    result2 = calc_depth_cumulative_distribution_per_sample(h5,
                                                            max_depth=args['max_depth'],
                                                            mask_function=_is_hom,
                                                            mask_field='/calls/GT',
                                                            by_chunk=args['by_chunk'])
    distrib_dp_hom, cum_dp_hom = result2
    distrib_dp_hom = numpy.sum(distrib_dp_hom, axis=0)
    title = 'Depth distribution Homozygous'
    axes = fig.add_subplot(212)
    plot_barplot(numpy.arange(0, distrib_dp_hom.shape[0]), distrib_dp_hom,
                 mpl_params={'set_xlabel': {'args': ['Depth'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    fpath = join(data_dir, 'distribution_hom_by_depth.csv')
    df_distrib_dp_hom = DataFrame(distrib_dp_hom)
    _save(fpath, df_distrib_dp_hom)
#     fpath = join(data_dir, 'cumulative_hom_by_depth.csv')
#     df_cum_dp_hom = DataFrame(cum_dp_hom)
#     _save(fpath.strip('.png') + '.csv', df_cum_dp_hom)
    canvas.print_figure(fhand)

    # GQ distribution per genotype
    fpath = join(data_dir, 'gq_distribution_per_gt.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    # Heterozygous
    result = calc_gq_cumulative_distribution_per_sample(h5,
                                                        mask_function=_is_het,
                                                        mask_field='/calls/GT',
                                                        by_chunk=args['by_chunk'])
    distrib_gq_het, cum_gq_het = result
    distrib_gq_het = numpy.sum(distrib_gq_het, axis=0)
    title = 'Genotype Quality (GQ) distribution Heterozygous'
    axes = fig.add_subplot(211)
    plot_barplot(numpy.arange(0, distrib_gq_het.shape[0]), distrib_gq_het,
                 mpl_params={'set_xlabel': {'args': ['GQ'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    fpath = join(data_dir, 'distribution_gq_het.csv')
    df_distrib_gq_het = DataFrame(distrib_gq_het)
    _save(fpath, df_distrib_gq_het)
#     fpath = join(data_dir, 'cumulative_het_by_depth.csv')
#     df_cum_gq_het = DataFrame(cum_gq_het)
#     _save(fpath.strip('.png') + '.csv', df_cum_gq_het)

    # Homozygous
    result2 = calc_gq_cumulative_distribution_per_sample(h5,
                                                         mask_function=_is_hom,
                                                         mask_field='/calls/GT',
                                                         by_chunk=args['by_chunk'])
    distrib_gq_hom, cum_gq_hom = result2
    distrib_gq_hom = numpy.sum(distrib_gq_hom, axis=0)
    title = 'Genotype Quality (GQ) distribution Homozygous'
    axes = fig.add_subplot(212)
    plot_barplot(numpy.arange(0, distrib_gq_hom.shape[0]), distrib_gq_hom,
                 mpl_params={'set_xlabel': {'args': ['GQ'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    fpath = join(data_dir, 'distribution_gq_hom.csv')
    df_distrib_gq_hom = DataFrame(distrib_gq_hom)
    _save(fpath, df_distrib_gq_hom)
#     fpath = join(data_dir, 'cumulative_hom_by_depth.csv')
#     df_cum_gq_hom = DataFrame(cum_gq_hom)
#     _save(fpath.strip('.png') + '.csv', df_cum_gq_hom)
    canvas.print_figure(fhand)

    # GT stats per sample
    fpath = join(data_dir, 'genotype_basic_stats.png')
    gt_stats = _calc_stat(h5, GenotypeStatsCalculator(), reduce_funct=numpy.add,
                          by_chunk=args['by_chunk'])
    gt_stats = gt_stats.transpose()
    title = 'Genotypes counts per sample'
    plot_pandas_barplot(gt_stats, ['Ref Homozygous', 'Heterozygous',
                                   'Alt Homozygous', 'Missing GT'],
                        mpl_params={'set_xlabel': {'args': ['Samples'],
                                                   'kwargs': {}},
                                    'set_ylabel': {'args': ['Counts'],
                                                   'kwargs': {}},
                                    'set_title': {'args': [title],
                                                  'kwargs': {}}},
                        color=['darkslategrey', 'c', 'paleturquoise',
                               'cadetblue'],
                        fpath=fpath, stacked=True)
    df_gt_stats = DataFrame(gt_stats)
    _save(fpath.strip('.png') + '.csv', df_gt_stats)

    # Distribution of the number of samples with a depth higher than
    # given values
    fpath = join(data_dir, 'gts_distribution_per_depth.png')
    distrib, cum = calc_called_gts_distrib_per_depth(h5, depths=args['depths'])
    title = 'Distribution of the number of samples with a depth higher than'
    title += ' given values'
    mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Number of samples'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    if h5.samples is not None:
        mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs': {}}
    plot_boxplot(distrib, fhand=open(fpath, 'w'), figsize=(15, 10),
                 mpl_params=mpl_params)
    df_distrib = DataFrame(distrib)
    _save(fpath.strip('.png') + '.csv', df_distrib)
#     fpath = join(data_dir, 'gts_cumulative_per_depth.csv')
#     df_cum = DataFrame(cum)
#     _save(fpath.strip('.png') + '.csv', df_cum)

    # GQ distribution per depth
    fpath = join(data_dir, 'gq_distrig_per_snp.png')
    distrib_gq, cum_gq = calc_quality_by_depth(h5, depths=args['depths'])
    title = 'Genotype Quality (GQ) distribution per depth'
    mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_ylabel': {'args': ['GQ'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}},
                  'set_xticklabels': {'args': [args['depths']], 'kwargs': {}}}
    plot_boxplot(distrib_gq, fhand=open(fpath, 'w'), figsize=(15, 10),
                 mpl_params=mpl_params)
    df_cum_gq = DataFrame(cum_gq)
    _save(fpath.strip('.png') + '.csv', df_cum_gq)

#     #
#     try:
#         allele_distrib_2D = calc_allele_obs_distrib_2D(h5, by_chunk=False)
#         fpath = join(data_dir, 'allele_obs_distribution_2D.csv')
#         df_allele_distrib_2D = DataFrame(allele_distrib_2D)
#         _save(fpath.strip('.png') + '.csv', df_allele_distrib_2D)
#     except ValueError:
#         sys.stderr('Allele distribution 2D could not be calculated\n')


#Pandas dataframe(data, index, columns, dtype, copy)
def _save(path, dataframe):
    file = open(path, mode='w')
    dataframe.to_csv(file)

if __name__ == '__main__':
    main()
