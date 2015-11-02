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
                       MAX_N_ALLELES, MAX_ALLELE_COUNTS)
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
                                        HWECalcualtor,
                                        PositionalStatsCalculator, calc_inbreeding_coeficient, calc_exp_het,
                                        _CalledGTCalculator)
from variation.plot import (plot_histogram, plot_pandas_barplot,
                            qqplot, plot_boxplot, plot_barplot, plot_hist2d,
                            plot_lines,
    manhattan_plot)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from itertools import combinations_with_replacement
from scipy.stats._continuous_distns import chi2
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
    help_msg = 'Max allele counts for 2D distribution (default {})'
    parser.add_argument('-mc', '--max_allele_counts', default=MAX_ALLELE_COUNTS,
                        help=help_msg.format(MAX_ALLELE_COUNTS), type=int)
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
    args['max_allele_counts'] = parsed_args.max_allele_counts
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
    print('plot_maf')
    plot_maf(h5, by_chunk, data_dir)
    print('plot_maf_dp')
    plot_maf_dp(h5, by_chunk, data_dir)
    print('plot_missing_gt_rate_per_snp')
    plot_missing_gt_rate_per_snp(h5, by_chunk, data_dir)
    print('plot_het_obs_distrib')
    plot_het_obs_distrib(h5, by_chunk, data_dir)
    print('plot_snp_dens_distrib')
    plot_snp_dens_distrib(h5, by_chunk, args['window_size'], data_dir)
    print('plot_dp_distrib_per_sample')
    plot_dp_distrib_per_sample(h5, by_chunk, args['max_depth'], data_dir)
    print('plot_dp_distrib_all_sample')
    plot_dp_distrib_all_sample(h5, by_chunk, args['max_depth'], data_dir)
    print('plot_gq_distrib_per_sample')
    plot_gq_distrib_per_sample(h5, by_chunk, data_dir,
                               max_value=args['max_gq'])
    print('plot_gq_distrib_all_sample')
    plot_gq_distrib_all_sample(h5, by_chunk, data_dir,
                               max_value=args['max_gq'])
    print('plot_dp_distrib_per_gt')
    plot_dp_distrib_per_gt(h5, by_chunk, args['max_depth'], data_dir)
    print('plot_gq_distrib_per_gt')
    plot_gq_distrib_per_gt(h5, by_chunk, data_dir, max_value=args['max_gq'])
    print('plot_gt_stats_per_sample')
    plot_gt_stats_per_sample(h5, by_chunk, data_dir)
    print('plot_ditrib_num_samples_hi_dp')
    plot_ditrib_num_samples_hi_dp(h5, by_chunk, args['depths'], data_dir)
    print('plot_gq_distrib_per_dp')
    plot_gq_distrib_per_dp(h5, by_chunk, args['depths'], data_dir,
                           max_value=args['max_gq'],
                           max_depth=args['max_depth'])
    print('plot_hwe_stats')
    plot_hwe(h5, args['max_num_alleles'], by_chunk, data_dir, ploidy=2)
    print('plot_nucleotide_diversity')
    plot_nucleotide_diversity_measures(h5, args['max_num_alleles'],
                                       args['window_size'], data_dir, by_chunk)
    print('plot_allele_obs_distrib_2D')
    plot_allele_obs_distrib_2D(h5, by_chunk, data_dir,
                                args['max_allele_counts'])
    print('plot_inbreeding_coeficient')
    plot_inbreeding_coeficient(h5, args['max_num_alleles'], by_chunk, data_dir)


def plot_maf(h5, by_chunk, data_dir):
    mafs = _calc_stat(h5, _MafCalculator(), by_chunk=by_chunk)
    fpath = join(data_dir, 'mafs.png')
    title = 'Maximum allele frequency (MAF) distribution'
    plot_histogram(mafs, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['MAF'], 'kwargs': {}},
                               'set_ylabel': {'args': ['SNP number'],
                                              'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   range_=(0, 1))
    df_mafs = DataFrame(mafs)
    _save(fpath.strip('.png') + '.csv', df_mafs)
    # Manhattan plot for SNP density
    fpath = join(data_dir, 'maf_manhattan.png')
    fhand = open(fpath, 'w')
    title = 'Max Allele Freq (MAF) along the genome'
    manhattan_plot(h5['/variations/chrom'], h5['/variations/pos'], mafs,
                   mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['MAF'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                   fhand=fhand, figsize=(15, 7.5))
    bg_fhand = open(join(data_dir, 'maf.bg'), 'w')
    pos_maf = PositionalStatsCalculator(h5['/variations/chrom'],
                                            h5['/variations/pos'], mafs)
    pos_maf.write(bg_fhand, 'MAF', 'Maximum allele frequency',
                      track_type='bedgraph')
    

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
                               'set_ylabel': {'args': ['SNP number'],
                                              'kwargs': {}},
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
                               'set_ylabel': {'args': ['SNP number'], 'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}},
                               'set_yscale': {'args': ['log'], 'kwargs': {}}},
                   axes=axes)
    df_het = DataFrame(het)
    _save(fpath.strip('.png') + '.csv', df_het)

    axes = fig.add_subplot(212)
    het_sample = _calc_stat(h5, _ObsHetCalculatorBySample(),
                            by_chunk=by_chunk, reduce_funct=numpy.add)
    called_gts_sample = _calc_stat(h5, _CalledGTCalculator(axis=0),
                                   reduce_funct=numpy.add, by_chunk=by_chunk)
    het_sample = het_sample / called_gts_sample
    title = 'Sample observed Heterozygosity distribution'
    het_sample = _remove_nans(het_sample)
    plot_histogram(het_sample, bins=100, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['Heterozygosity'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Sample number'],
                                              'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   axes=axes)
    fpath = join(data_dir, 'obs_het_per_sample.csv')
    df_het_sample = DataFrame(het_sample)
    _save(fpath, df_het_sample)
    canvas.print_figure(fhand)


def plot_snp_dens_distrib(h5, by_chunk, window_size, data_dir):
        # SNP density distribution
    density = calc_snp_density(h5, window_size)
    fpath = join(data_dir, 'snps_density.png')
    title = 'SNP density distribution per {} bp windows'
    title = title.format(window_size)
    plot_histogram(density, bins=50, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['SNP density'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['SNP number'],
                                              'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}},
                               'set_yscale': {'args': ['log'], 'kwargs': {}}})
    df_dens = DataFrame(density)
    _save(fpath.strip('.png') + '.csv', df_dens)
    
    # Manhattan plot for SNP density
    fpath = join(data_dir, 'snps_density_manhattan.png')
    fhand = open(fpath, 'w')
    title = 'SNP denisity along the genome'
    manhattan_plot(h5['/variations/chrom'], h5['/variations/pos'], density,
                   mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['SNP per {} bp'.format(window_size)],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}},
                             'set_yscale':{'args': ['log'], 'kwargs': {}}},
                   fhand=fhand, figsize=(15, 7.5), ylim=1)
    bg_fhand = open(join(data_dir, 'snp_density.bg'), 'w')
    pos_dens = PositionalStatsCalculator(h5['/variations/chrom'],
                                         h5['/variations/pos'], density)
    pos_dens.write(bg_fhand, 'snp_density',
                   'SNP number in {} bp around'.format(window_size),
                   track_type='bedgraph')


def plot_dp_distrib_per_sample(h5, by_chunk, max_depth, data_dir):
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
                             'set_ylabel': {'args': ['Number of GTs'],
                                            'kwargs': {}},
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
                             'set_ylabel': {'args': ['Number of GTs'],
                                            'kwargs': {}},
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
    gs = gridspec.GridSpec(2, 2)
    fig = Figure(figsize=(20, 10))
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
        axes = fig.add_subplot(gs[0, i])
        plot_barplot(numpy.arange(0, distrib_dp.shape[0]), distrib_dp,
                     mpl_params={'set_xlabel': {'args': ['Depth'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['Number of GTs'], 'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}},
                     axes=axes)
        cum_dp = numpy.sum(cum_dp, axis=0)
        cum_dp = cum_dp / cum_dp[0] * 100
        title = 'Cumulative depth distribution {}'.format(name)
        axes = fig.add_subplot(gs[1, i])
        plot_barplot(numpy.arange(0, cum_dp.shape[0]), cum_dp,
                     mpl_params={'set_xlabel': {'args': ['Depth'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['% GTs'], 'kwargs': {}},
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
    fig = Figure(figsize=(20, 10))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(2, 2)
    masks = [_is_het, _is_hom]
    names = ['Heterozygous', 'Homozygous']
    for i, (mask, name) in enumerate(zip(masks, names)):
        result = calc_gq_cumulative_distribution_per_sample(h5,
                                                            mask_function=mask,
                                                            mask_field='/calls/GT',
                                                            by_chunk=by_chunk,
                                                            max_value=max_value)
        distrib_gq, cum_gq = result
        distrib_gq = numpy.sum(distrib_gq, axis=0)
        title = 'Genotype Quality (GQ) distribution {}'.format(name)
        axes = fig.add_subplot(gs[0, i])
        plot_barplot(numpy.arange(0, distrib_gq.shape[0]), distrib_gq,
                     mpl_params={'set_xlabel': {'args': ['GQ'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['Number of GTs'], 'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}},
                     axes=axes)
        cum_gq = numpy.sum(cum_gq, axis=0)
        cum_gq = cum_gq / cum_gq[0] * 100
        title = 'Cumulative Genotype Quality (GQ) distribution {}'.format(name)
        axes = fig.add_subplot(gs[1, i])
        plot_barplot(numpy.arange(0, cum_gq.shape[0]), cum_gq,
                     mpl_params={'set_xlabel': {'args': ['GQ'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['% GTs'], 'kwargs': {}},
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
                  'set_ylabel': {'args': ['Number of GTs'], 'kwargs': {}},
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


def plot_gq_distrib_per_dp(h5, by_chunk, depths, data_dir, max_value,
                           max_depth):
        # GQ distribution per depth
    fpath = join(data_dir, 'gq_distrig_per_snp.png')
    distrib_gq, cum_gq = calc_quality_by_depth_distrib(h5, depths=depths,
                                                       max_value=max_value)

    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    fhand = open(fpath, 'w')
    axes = fig.add_subplot(111)
    title = 'Genotype Quality (GQ) distribution per depth'
    mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_ylabel': {'args': ['GQ'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}},
                  'set_xticklabels': {'args': [depths],
                                      'kwargs': {'rotation': 90}}}
    plot_boxplot(distrib_gq, axes=axes, figsize=(15, 10),
                 mpl_params=mpl_params)
    df_cum_gq = DataFrame(cum_gq)
    axes = axes.twinx()
    distrib_dp, _ = calc_depth_cumulative_distribution_per_sample(h5,
                                                            by_chunk=by_chunk,
                                                            max_depth=max_depth)
    distrib_dp_all = numpy.sum(distrib_dp, axis=0)
    plot_lines(numpy.arange(0, distrib_gq.shape[0]+1), distrib_dp_all[:distrib_gq.shape[0]+1],
               mpl_params={'set_ylabel': {'args': ['Number of GTs'],
                                          'kwargs': {}},
                           'set_title': {'args': [title], 'kwargs': {}}},
               axes=axes)

    canvas.print_figure(fhand)
    _save(fpath.strip('.png') + '.csv', df_cum_gq)


def plot_allele_obs_distrib_2D(h5, by_chunk, data_dir, max_allele_counts):
    # Allele observation distribution 2D
    if '/calls/AO' in h5.keys() and '/calls/RO' in h5.keys():
        masks = [_is_het, _is_hom_alt, _is_hom_ref]
        names = ['Heterozygous', 'Alt Homozygous', 'Ref Homozygous']
        fig = Figure(figsize=(22, 25))
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(3, 2)
        fpath = join(data_dir, 'allele_obs_distrib_per_gt.png')
        fhand = open(fpath, 'w')
        max_values = [max_allele_counts, max_allele_counts]
        for i, (mask_func, name) in enumerate(zip(masks, names)):
            axes = fig.add_subplot(gs[i, 0])
            allele_distrib_2D = calc_allele_obs_distrib_2D(h5, by_chunk=False,
                                                           mask_function=mask_func,
                                                           mask_field='/calls/GT',
                                                           max_values=max_values)
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
                                                                 mask_field='/calls/GT',
                                                                 max_values=max_values)
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
    ic = calc_inbreeding_coeficient(h5, max_num_allele=max_num_allele,
                                    by_chunk=by_chunk)
    fpath = join(data_dir, 'inbreeding_coef_distribution.png')
    fhand = open(fpath, 'w')
    title = 'Inbreeding coefficient distribution all samples'
    plot_histogram(_remove_nans(ic),
                 mpl_params={'set_xlabel': {'args': ['Inbreeding coefficient'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Number of SNPs'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}},
                             'set_xlim': {'args': [-1, 1], 'kwargs': {}}},
                 fhand=fhand)
    fpath = join(data_dir, 'ic_manhattan.png')
    fhand = open(fpath, 'w')
    title = 'Inbreeding coefficient (IC) along the genome'
    manhattan_plot(h5['/variations/chrom'], h5['/variations/pos'], ic,
                   mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['IC'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                   fhand=fhand, figsize=(15, 7.5), ylim=-1)
    bg_fhand = open(join(data_dir, 'ic.bg'), 'w')
    pos_ic = PositionalStatsCalculator(h5['/variations/chrom'],
                                       h5['/variations/pos'], ic)
    pos_ic.write(bg_fhand, ic, 'IC', 'Inbreeding coefficient',
                      track_type='bedgraph')


def plot_hwe(h5, max_num_alleles, by_chunk, data_dir,
                         ploidy=2):
    for max_num_allele in range(2, max_num_alleles):
        fpath = join(data_dir, 'hwe_chi2_qqplot_{}_alleles.png'.format(max_num_allele))
        fhand = open(fpath, 'w')
        df = len(list(combinations_with_replacement(range(max_num_allele),
                                                    ploidy))) - max_num_allele
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
        axes = axes.twinx()
        rv = chi2(df)
        x = numpy.linspace(0, max(hwe_chi2), 1000)
        axes.plot(x, rv.pdf(x), color='b', lw=2, label='Expected Chi2')
        axes.set_ylabel('Expected Chi2 density')
        canvas.print_figure(fhand)
        
        # Manhattan plot for HWE pvalue
        fpath = join(data_dir, 'hwe_pvalue_manhattan_{}_alleles.bg'.format(max_num_allele))
        fhand = open(fpath, 'w')
        hwe_pvalues = hwe_test[:, 1]
        title = 'Chi2 test for HWE df={} along the genome'.format(df)
        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        axes = fig.add_subplot(211)
        manhattan_plot(h5['/variations/chrom'], h5['/variations/pos'], hwe_test[:, 0],
                       mpl_params={'set_ylabel': {'args': ['Chi2 statistic'],
                                                'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}},
                       axes=axes, figsize=(15, 7.5), ylim=0, show_chroms=False)
        axes = fig.add_subplot(212)
        manhattan_plot(h5['/variations/chrom'], h5['/variations/pos'], hwe_pvalues,
                       mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['1 / p-value'],
                                                'kwargs': {}},
                                 'set_yscale': {'args': ['log'], 'kwargs': {}}},
                       axes=axes, figsize=(15, 7.5), ylim=1, yfunc=lambda x: 1/x,
                       yline=hwe_pvalues.shape[0]/0.05)
        canvas.print_figure(fhand)
        
        # Write bedgraph files
        bg_fhand = open(join(data_dir, 'hwe_pvalue_{}_alleles.bg'.format(max_num_allele)), 'w')
        pos_hwe_pval = PositionalStatsCalculator(h5['/variations/chrom'],
                                                 h5['/variations/pos'], hwe_pvalues)
        pos_hwe_pval.write(bg_fhand, 'hwe_pvalue',
                           'HWE test chi2 df={} test p-value'.format(df),
                           track_type='bedgraph')
        bg_fhand = open(join(data_dir, 'hwe_chi2_{}_alleles.bg'.format(max_num_allele)), 'w')
        pos_hwe_chi2 = PositionalStatsCalculator(h5['/variations/chrom'],
                                                h5['/variations/pos'],
                                                hwe_test[:, 0])
        pos_hwe_chi2.write(bg_fhand, 'hwe_chi2',
                          'HWE test chi2 df={} statistic'.format(df),
                          track_type='bedgraph')


def plot_nucleotide_diversity_measures(hdf5, max_num_alleles, window_size,
                                       data_dir, by_chunk):
    fig = Figure(figsize=(20, 20))
    canvas = FigureCanvas(fig)
    marker = 'k'
    
    # Number of variable positions per bp
    chrom, pos = hdf5['/variations/chrom'][:], hdf5['/variations/pos'][:]
    snp_density = PositionalStatsCalculator(chrom, pos,
                                            numpy.ones(hdf5['/variations/pos'].shape),
                                            window_size=window_size,
                                            step=window_size)
    snp_density = snp_density.calc_window_stat()
    bg_fhand = open(join(data_dir, 'diversity_s.bg'), 'w')
    snp_density.write(bg_fhand, 's',
                      'SNP density in windows of {} bp'.format(window_size),
                      track_type='bedgraph')
    axes = fig.add_subplot(311)
    title = 'Nucleotide diversity measures averaged in windows of {} bp'.format(window_size)
    manhattan_plot(snp_density.chrom, snp_density.pos, snp_density.stat,
                   mpl_params={'set_title': {'args': [title],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['SNPs number / bp'],
                                              'kwargs': {}},
                               'set_ylim': {'args': [0, 1.2*numpy.max(snp_density.stat)],
                                              'kwargs': {}}},
                   axes=axes, ylim=0, show_chroms=False, marker=marker)
    
    # Watterson estimator of nucleotide diversity
    
    n_seqs = hdf5['/calls/GT'].shape[1] * hdf5['/calls/GT'].shape[2]
    correction_factor = numpy.sum(1 / numpy.arange(1, n_seqs))
    watterson = snp_density
    watterson.stat = watterson.stat / correction_factor
    bg_fhand = open(join(data_dir, 'diversity_s.bg'), 'w')
    watterson.write(bg_fhand, 's',
                    'SNP density in windows of {} bp'.format(window_size),
                    track_type='bedgraph')
    axes = fig.add_subplot(312)
    manhattan_plot(watterson.chrom, watterson.pos, watterson.stat,
                   mpl_params={'set_ylabel': {'args': ['Watterson estimator'],
                                              'kwargs': {}},
                               'set_ylim': {'args': [0, 1.2*numpy.max(watterson.stat)],
                                              'kwargs': {}}},
                   axes=axes, ylim=0, show_chroms=False, marker=marker)
    
    # Expected heterozygosity
    exp_het = calc_exp_het(hdf5, max_num_allele=max_num_alleles,
                           by_chunk=by_chunk)
    pi = PositionalStatsCalculator(chrom, pos, exp_het,
                                   window_size=window_size, step=window_size)
    pi = pi.calc_window_stat()
    bg_fhand = open(join(data_dir, 'diversity_pi.bg'), 'w')
    pi.write(bg_fhand, 's',
                      'Pi in windows of {} bp'.format(window_size),
                      track_type='bedgraph')
    axes = fig.add_subplot(313)
    manhattan_plot(pi.chrom, pi.pos, pi.stat, axes=axes, ylim=0, marker=marker,
                   mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['Pi'],
                                              'kwargs': {}},
                               'set_ylim': {'args': [0, 1.2*numpy.max(pi.stat)],
                                              'kwargs': {}}})
    canvas.print_figure(open(join(data_dir, 'nucleotide_diversity.png'), 'w'))


def _save(path, dataframe):
    file = open(path, mode='w')
    dataframe.to_csv(file)

if __name__ == '__main__':
    create_plots()
