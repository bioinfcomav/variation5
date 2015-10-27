#!/usr/bin/env python
from os.path import join
import os
import argparse
import numpy
from pandas import DataFrame
from matplotlib import gridspec
from variation.variations.vars_matrices import VariationsH5
from variation import (SNPS_PER_CHUNK, STATS_DEPTHS, MAX_DEPTH,
                       SNP_DENSITY_WINDOW_SIZE, MIN_N_GENOTYPES,
                       MAX_N_ALLELES)
from variation.variations.stats import (_calc_stat, calc_maf_depth_distrib,
                                        _MissingGTCalculator,
                                        _ObsHetCalculatorBySnps,
                                        _ObsHetCalculatorBySample,
                                        calc_snp_density,
                                        calc_depth_cumulative_distribution_per_sample,
                                        calc_gq_cumulative_distribution_per_sample,
                                        _is_het, _is_hom,
                                        GenotypeStatsCalculator,
                                        calc_called_gts_distrib_per_depth,
                                        calc_quality_by_depth_distrib,
                                        calc_allele_obs_distrib_2D,
                                        _remove_nans, _is_hom_ref, _is_hom_alt,
                                        calc_allele_obs_gq_distrib_2D,
                                        _MafCalculator,
    calc_inbreeding_coeficient_distrib, HWECalcualtor)
from variation.plot import (plot_histogram, plot_pandas_barplot,
                            plot_boxplot, plot_barplot, plot_hist2d, qqplot)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import sys
from itertools import combinations_with_replacement


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
                        help=help_msg, type=int)
    help_msg = 'Window size to calculate SNP density ({})'
    help_msg = help_msg.format(SNP_DENSITY_WINDOW_SIZE)
    parser.add_argument('-w', '--window_size', default=SNP_DENSITY_WINDOW_SIZE,
                        help=help_msg, type=int)
    help_msg = 'Min number of called genotypes to calculate stats ({})'
    help_msg = help_msg.format(MIN_N_GENOTYPES)
    parser.add_argument('-m', '--min_n_gts', default=MIN_N_GENOTYPES,
                        help=help_msg, type=int)
    help_msg = 'Max num alleles to calculate allele frequency ({})'
    help_msg = help_msg.format(MAX_N_ALLELES)
    parser.add_argument('-ma', '--max_num_alleles', default=MAX_N_ALLELES,
                        help=help_msg, type=int)
    help_msg = 'Max GQ value for distributions (Speeds up calculations)'
    parser.add_argument('-mq', '--max_gq', default=None, help=help_msg,
                        type=int)
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
    args['max_gq'] = parsed_args.max_gq
    return args


def create_plots():
    description = 'Calculates basic stats of a HDF5 file'
    parser = _setup_argparse(description=description)
    args = _parse_args(parser)
    data_dir = args['out_prefix']
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    h5 = VariationsH5(args['in_fpath'], mode='r',
                      vars_in_chunk=args['chunk_size'])
    by_chunk = args['by_chunk']
    plot_maf(h5, by_chunk, data_dir)
    plot_maf_dp(h5, by_chunk, data_dir)
    plot_missing_gt_rate_per_snp(h5, by_chunk, data_dir)
    plot_het_obs_distrib(h5, by_chunk, data_dir)
    plot_snp_dens_distrib(h5, by_chunk, args['window_size'], args['max_depth'],
                          data_dir)
    plot_dp_distrib_all_sample(h5, by_chunk, args['max_depth'], data_dir)
    plot_gq_distrib_per_sample(h5, by_chunk, data_dir,
                               max_value=args['max_gq'])
    plot_gq_distrib_all_sample(h5, by_chunk, data_dir,
                               max_value=args['max_gq'])
    plot_dp_distrib_per_gt(h5, by_chunk, args['max_depth'], data_dir)
    plot_gq_distrib_per_gt(h5, by_chunk, data_dir, max_value=args['max_gq'])
    plot_gt_stats_per_sample(h5, by_chunk, data_dir)
    plot_ditrib_num_samples_hi_dp(h5, by_chunk, args['depths'], data_dir)
    plot_gq_distrib_per_dp(h5, by_chunk, args['depths'], data_dir,
                           max_value=args['max_gq'])
    plot_allele_obs_distrib_2D(h5, by_chunk, data_dir)
    plot_inbreeding_coeficient(h5, args['max_num_alleles'], by_chunk, data_dir)
    plot_hwe_chi2_qqplot(h5, args['max_num_alleles'], by_chunk, data_dir,
                         ploidy=2)


def plot_maf(h5, by_chunk, data_dir):
    mafs = _calc_stat(h5, _MafCalculator(), by_chunk=by_chunk)
    fpath = join(data_dir, 'mafs.png')
    title = 'Maximum allele frequency (MAF) distribution'
    plot_histogram(mafs, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['MAF'], 'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   range_=(0, 1))
    df_mafs = DataFrame(mafs)
    _save(fpath.strip('.png') + '.csv', df_mafs)


def plot_maf_dp(h5, by_chunk, data_dir):
    try:
        maf_depths_distrib = calc_maf_depth_distrib(h5, by_chunk)
        fpath = join(data_dir, 'mafs_depths_distribution.png')
        title = 'Depth based Maximum allele frequency (MAF) distribution'
        mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                      'set_ylabel': {'args': ['MAF (depth)'], 'kwargs': {}},
                      'set_title': {'args': [title], 'kwargs': {}}}
        if h5.samples is not None:
            mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs': {}}
        plot_boxplot(maf_depths_distrib[:, 1:], fhand=open(fpath, 'w'),
                     figsize=(40, 10),
                     mpl_params=mpl_params)
        df_maf_depths_distrib = DataFrame(maf_depths_distrib)
        _save(fpath.strip('.png') + '.csv', df_maf_depths_distrib)
    except ValueError:
        sys.stderr.write('MAF depth could not be calculated\n')


def plot_missing_gt_rate_per_snp(h5, by_chunk, data_dir):
        # Missing genotype rate per SNP distribution
    rates = _calc_stat(h5, _MissingGTCalculator(), by_chunk=by_chunk)
    fpath = join(data_dir, 'missing_gt_rate.png')
    title = 'Missing Genotype rates per SNP distribution'
    plot_histogram(rates, bins=100, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['Missing GT rate'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}})
    df_rates = DataFrame(rates)
    _save(fpath.strip('.png') + '.csv', df_rates)


def plot_het_obs_distrib(h5, by_chunk, data_dir):
        # Observed heterozygosity distributions
    fpath = join(data_dir, 'obs_het.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    axes = fig.add_subplot(211)
    het = _calc_stat(h5, _ObsHetCalculatorBySnps(), by_chunk=by_chunk)
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
                            by_chunk=by_chunk)
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


def plot_snp_dens_distrib(h5, by_chunk, window_size, max_depth, data_dir):
        # SNP density distribution
    density = calc_snp_density(h5, window_size)
    fpath = join(data_dir, 'snps_density.csv')
    title = 'SNP density distribution per {} bp windows'
    title = title.format(window_size)
    plot_histogram(density, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['SNP density'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}})
    df_dens = DataFrame(density)
    _save(fpath.strip('.png') + '.csv', df_dens)

    # DP distribution per sample
    distrib_dp, cum_dp = calc_depth_cumulative_distribution_per_sample(h5,
                                                                by_chunk=by_chunk,
                                                                max_depth=max_depth)
    fpath = join(data_dir, 'depth_distribution_per_sample.png')
    title = 'Depth distribution per sample'
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    if h5.samples is not None:
        mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs':
                                         {'rotation':90}}
    plot_boxplot(distrib_dp, fhand=open(fpath, 'w'), figsize=(40, 10),
                 mpl_params=mpl_params)
    df_distrib_dp = DataFrame(distrib_dp)
    _save(fpath.strip('.png') + '.csv', df_distrib_dp)


def plot_dp_distrib_all_sample(h5, by_chunk, max_depth, data_dir):
        # DP distribution all sample
    distrib_dp, cum_dp = calc_depth_cumulative_distribution_per_sample(h5,
                                                                by_chunk=by_chunk,
                                                                max_depth=max_depth)
    distrib_dp_all = numpy.sum(distrib_dp, axis=0)
    cum_dp = numpy.sum(cum_dp, axis=0)
    fpath = join(data_dir, 'depth_distribution.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 20))
    canvas = FigureCanvas(fig)
    axes = fig.add_subplot(211)
    title = 'Depth distribution all samples'
    plot_barplot(numpy.arange(0, distrib_dp_all.shape[0]), distrib_dp_all,
                 mpl_params={'set_xlabel': {'args': ['Depth'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    cum_dp = cum_dp/cum_dp[0] * 100
    axes = fig.add_subplot(212)
    title = 'Depth cumulative distribution all samples'
    plot_barplot(numpy.arange(0, cum_dp.shape[0]), cum_dp,
                 mpl_params={'set_xlabel': {'args': ['Depth'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['% calls > Depth '], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    canvas.print_figure(fhand)


def plot_gq_distrib_per_sample(h5, by_chunk, data_dir, max_value):
        # GQ distribution per sample
    distrib_gq, _ = calc_gq_cumulative_distribution_per_sample(h5,
                                                               by_chunk=by_chunk,
                                                               max_value=max_value)
    fpath = join(data_dir, 'gq_distribution_per_sample.png')
    title = 'Genotype Quality (QG) distribution per sample'
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': ['GQ'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    if h5.samples is not None:
        mpl_params['set_xticklabels'] = {'args': [h5.samples],
                                         'kwargs': {'rotation': 90}}
    plot_boxplot(distrib_gq, fhand=open(fpath, 'w'), figsize=(40, 10),
                 mpl_params=mpl_params)
    df_distrib_gq = DataFrame(distrib_gq)
    _save(fpath.strip('.png') + '.csv', df_distrib_gq)


def plot_gq_distrib_all_sample(h5, by_chunk, data_dir, max_value):

    distrib_gq, cum_gq = calc_gq_cumulative_distribution_per_sample(h5,
                                                                by_chunk=by_chunk,
                                                                max_value=max_value)
    distrib_gq_all = numpy.sum(distrib_gq, axis=0)
    cum_gq = numpy.sum(cum_gq, axis=0)
    fpath = join(data_dir, 'gq_distribution.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 20))
    canvas = FigureCanvas(fig)
    title = 'Genotype Quality (GQ) distribution all samples'
    axes = fig.add_subplot(211)
    plot_barplot(numpy.arange(0, distrib_gq_all.shape[0]), distrib_gq_all,
                 mpl_params={'set_xlabel': {'args': ['GQ'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    title = 'Genotype Quality (GQ) cumulative distribution all samples'
    axes = fig.add_subplot(212)
    cum_gq = cum_gq/cum_gq[0] * 100
    plot_barplot(numpy.arange(0, cum_gq.shape[0]), cum_gq,
                 mpl_params={'set_xlabel': {'args': ['GQ'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['% calls > GQ'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes)
    canvas.print_figure(fhand)


def plot_dp_distrib_per_gt(h5, by_chunk, max_depth, data_dir):
        # Depth distribution per genotype
    fpath = join(data_dir, 'depth_distribution_per_gt.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    masks = [_is_het, _is_hom]
    names = ['Heterozygous', 'Homozygous']
    for i, (mask, name) in enumerate(zip(masks, names)):
        result = calc_depth_cumulative_distribution_per_sample(h5,
                                                               max_depth=max_depth,
                                                               mask_function=mask,
                                                               mask_field='/calls/GT',
                                                               by_chunk=by_chunk)
        distrib_dp, cum_dp = result
        distrib_dp = numpy.sum(distrib_dp, axis=0)
        title = 'Depth distribution {}'.format(name)
        axes = fig.add_subplot(211+i)
        plot_barplot(numpy.arange(0, distrib_dp.shape[0]), distrib_dp,
                     mpl_params={'set_xlabel': {'args': ['Depth'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}},
                     axes=axes)
        fpath = join(data_dir, 'distribution_{}_by_depth.csv'.format(name))
        df_distrib_dp = DataFrame(distrib_dp)
        _save(fpath, df_distrib_dp)
    canvas.print_figure(fhand)


def plot_gq_distrib_per_gt(h5, by_chunk, data_dir, max_value=None):

    # GQ distribution per genotype
    fpath = join(data_dir, 'gq_distribution_per_gt.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    masks = [_is_het, _is_hom]
    names = ['Heterozygous', 'Homozygous']
    for i, (mask, name) in enumerate(zip(masks, names)):
        result = calc_gq_cumulative_distribution_per_sample(h5,
                                                            mask_function=mask,
                                                            mask_field='/calls/GT',
                                                            by_chunk=by_chunk,
                                                            max_value=max_value)
        distrib_gq, _ = result
        distrib_gq = numpy.sum(distrib_gq, axis=0)
        title = 'Genotype Quality (GQ) distribution {}'.format(name)
        axes = fig.add_subplot(211+i)
        plot_barplot(numpy.arange(0, distrib_gq.shape[0]), distrib_gq,
                     mpl_params={'set_xlabel': {'args': ['GQ'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}},
                     axes=axes)
        fpath = join(data_dir, 'distribution_gq_{}.csv'.format(name))
        df_distrib_gq = DataFrame(distrib_gq)
        _save(fpath, df_distrib_gq)
    canvas.print_figure(fhand)


def plot_gt_stats_per_sample(h5, by_chunk, data_dir):
    # GT counts per sample
    fpath = join(data_dir, 'genotype_counts_per_sample.png')
    gt_stats = _calc_stat(h5, GenotypeStatsCalculator(), reduce_funct=numpy.add,
                          by_chunk=by_chunk)
    gt_stats = gt_stats.transpose()
    title = 'Genotypes counts per sample'
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    if h5.samples is not None:
        mpl_params['set_xticklabels'] = {'args': [h5.samples], 'kwargs': {}}
    plot_pandas_barplot(gt_stats, ['Ref Homozygous', 'Heterozygous',
                                   'Alt Homozygous', 'Missing GT'],
                        mpl_params=mpl_params,
                        color=['darkslategrey', 'c', 'paleturquoise',
                               'cadetblue'],
                        fpath=fpath, stacked=True)
    
    # Missing per sample
    fpath = join(data_dir, 'missing_per_sample.png')
    title = 'Missing genotypes counts per sample'
    mpl_params['set_ylabel'] = {'args': ['Missing Genotypes Number'],
                                 'kwargs': {}}
    mpl_params['set_title'] = {'args': [title], 'kwargs': {}}
    plot_pandas_barplot(gt_stats[:, -1], ['Missing GT'],
                        mpl_params=mpl_params,
                        fpath=fpath, stacked=True)
    
    # Heterozygous per sample
    fpath = join(data_dir, 'het_per_sample.png')
    title = 'Heterozygous counts per sample'
    mpl_params['set_ylabel'] = {'args': ['Heterozygous Number'],
                                 'kwargs': {}}
    mpl_params['set_title'] = {'args': [title], 'kwargs': {}}
    plot_pandas_barplot(gt_stats[:, 1], ['Heterozygous'],
                        mpl_params=mpl_params,
                        fpath=fpath, stacked=True)
    
    # GT percentage without missing values
    fpath = join(data_dir, 'gt_perc_per_sample.png')
    title = 'Genotypes percentage per sample'
    mpl_params['set_ylabel'] = {'args': ['% Genotypes'],
                                 'kwargs': {}}
    mpl_params['set_title'] = {'args': [title], 'kwargs': {}}
    gt_perc = gt_stats[:, :-1] / gt_stats[:, :-1].sum(axis=1,
                                                      keepdims=True) * 100
    plot_pandas_barplot(gt_perc, ['Ref Homozygous', 'Heterozygous',
                                   'Alt Homozygous'],
                        mpl_params=mpl_params, fpath=fpath)
    
    df_gt_stats = DataFrame(gt_stats)
    _save(fpath.strip('.png') + '.csv', df_gt_stats)


def plot_ditrib_num_samples_hi_dp(h5, by_chunk, depths, data_dir):
        # Distribution of the number of samples with a depth higher than
    # given values
    fpath = join(data_dir, 'gts_distribution_per_depth.png')
    distrib, cum = calc_called_gts_distrib_per_depth(h5, depths=depths)
    title = 'Distribution of the number of samples with a depth higher than'
    title += ' given values'
    mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Number of samples'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}},
                  'set_xticklabels': {'args': [depths],
                                      'kwargs': {'rotation': 90}}}

    plot_boxplot(distrib, fhand=open(fpath, 'w'), figsize=(15, 10),
                 mpl_params=mpl_params)
    df_distrib = DataFrame(distrib)
    _save(fpath.strip('.png') + '.csv', df_distrib)


def plot_gq_distrib_per_dp(h5, by_chunk, depths, data_dir, max_value):
        # GQ distribution per depth
    fpath = join(data_dir, 'gq_distrig_per_snp.png')
    distrib_gq, cum_gq = calc_quality_by_depth_distrib(h5, depths=depths,
                                                       max_value=max_value)
    title = 'Genotype Quality (GQ) distribution per depth'
    mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_ylabel': {'args': ['GQ'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}},
                  'set_xticklabels': {'args': [depths],
                                      'kwargs': {'rotation': 90}}}
    plot_boxplot(distrib_gq, fhand=open(fpath, 'w'), figsize=(15, 10),
                 mpl_params=mpl_params)
    df_cum_gq = DataFrame(cum_gq)
    _save(fpath.strip('.png') + '.csv', df_cum_gq)


def plot_allele_obs_distrib_2D(h5, by_chunk, data_dir):
    # Allele observation distribution 2D
    if '/calls/AO' in h5.keys() and '/calls/RO' in h5.keys():
        masks = [_is_het, _is_hom_alt, _is_hom_ref]
        names = ['Heterozygous', 'Alt Homozygous', 'Ref Homozygous']
        fig = Figure(figsize=(22, 25))
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(3, 2)
        fpath = join(data_dir, 'allele_obs_distrib_per_gt.png')
        fhand = open(fpath, 'w')
        for i, (mask_func, name) in enumerate(zip(masks, names)):
            axes = fig.add_subplot(gs[i, 0])
            allele_distrib_2D = calc_allele_obs_distrib_2D(h5, by_chunk=False,
                                                           mask_function=mask_func,
                                                           mask_field='/calls/GT')
            title = 'Allele counts distribution 2D {}'.format(name)
            plot_hist2d(numpy.log10(allele_distrib_2D), axes=axes, fig=fig,
                        mpl_params={'set_xlabel': {'args': ['Alt allele counts'],
                                                   'kwargs': {}},
                                    'set_ylabel': {'args': ['Ref allele counts'],
                                                   'kwargs': {}},
                                    'set_title': {'args': [title], 'kwargs': {}}},
                        colorbar_label='log10(counts)')

            axes = fig.add_subplot(gs[i, 1])
            allele_distrib_gq_2D = calc_allele_obs_gq_distrib_2D(h5,
                                                                 by_chunk=False,
                                                                 mask_function=mask_func,
                                                                 mask_field='/calls/GT')
            title = 'Allele counts GQ distribution 2D {}'.format(name)
            plot_hist2d(allele_distrib_gq_2D, axes=axes, fig=fig,
                        mpl_params={'set_xlabel': {'args': ['Alt allele counts'],
                                                   'kwargs': {}},
                                    'set_ylabel': {'args': ['Ref allele counts'],
                                                   'kwargs': {}},
                                    'set_title': {'args': [title], 'kwargs': {}}},
                        colorbar_label='Genotype Quality (GQ)')
            fpath = join(data_dir, 'allele_obs_distrib_{}.csv'.format(name))
            df_allele_distrib_2D = DataFrame(allele_distrib_2D)
            _save(fpath, df_allele_distrib_2D)
            fpath = join(data_dir, 'allele_obs_gq_distrib_{}.csv'.format(name))
            df_allele_distrib_gq_2D = DataFrame(allele_distrib_gq_2D)
            _save(fpath, df_allele_distrib_gq_2D)
        canvas.print_figure(fhand)
    else:
        print('Allele distribution 2D could not be calculated\n')


def plot_inbreeding_coeficient(h5, max_num_allele, by_chunk, data_dir):
    distrib = calc_inbreeding_coeficient_distrib(h5,
                                                 max_num_allele=max_num_allele,
                                                 by_chunk=by_chunk)
    fpath = join(data_dir, 'inbreeding_coef_distribution.png')
    fhand = open(fpath, 'w')
    title = 'Inbreeding coefficient distribution all samples'
    x = numpy.arange(-100, 101)/100
    plot_barplot(x, distrib, width=0.01,
                 mpl_params={'set_xlabel': {'args': ['Inbreeding coefficient'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Counts'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 fhand=fhand)


def plot_hwe_chi2_qqplot(h5, max_num_allele, by_chunk, data_dir,
                         ploidy=2):
    fpath = join(data_dir, 'hwe_chi2_qqplot.png')
    fhand = open(fpath, 'w')
    df = len(list(combinations_with_replacement(range(max_num_allele),
                                                ploidy)))
    hwe_test = _calc_stat(h5, HWECalcualtor(max_num_allele, ploidy),
                          by_chunk=by_chunk)
    hwe_chi2 = _remove_nans(hwe_test[:, 0])
    
    fig = Figure(figsize=(10, 20))
    canvas = FigureCanvas(fig)
    axes = fig.add_subplot(211)
    title = 'QQplot for chi2 with df={} using {} alleles'.format(df,
                                                                 max_num_allele)
    qqplot(hwe_chi2, distrib='chi2', distrib_params=(df,), axes=axes,
           mpl_params={'set_title': {'args': [title], 'kwargs': {}}})
    
    title = 'Chi2 df={} statistic values distribution'.format(df)
    axes = fig.add_subplot(212)
    plot_histogram(hwe_chi2, bins=50, axes=axes,
                   mpl_params={'set_xlabel': {'args': ['Chi2 statistic'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['SNP number'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}})
    # TODO: add expected chi2 distribution density
    canvas.print_figure(fhand)
    

def _save(path, dataframe):
    file = open(path, mode='w')
    dataframe.to_csv(file)

if __name__ == '__main__':
    create_plots()
