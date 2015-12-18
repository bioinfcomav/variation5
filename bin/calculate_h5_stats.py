#!/usr/bin/env python

import os
import time
import argparse
import logging
from functools import partial
from os.path import join
from itertools import combinations_with_replacement

import numpy
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib import gridspec
from scipy.stats._continuous_distns import chi2

from variation.variations.vars_matrices import VariationsH5
from variation import (SNPS_PER_CHUNK, STATS_DEPTHS, MAX_DEPTH,
                       SNP_DENSITY_WINDOW_SIZE, MIN_N_GENOTYPES,
                       MAX_N_ALLELES, MAX_ALLELE_COUNTS,
                       MANHATTAN_WINDOW_SIZE, DEF_MIN_DEPTH)
from variation.plot import (plot_barplot, plot_hist2d, manhattan_plot,
                            plot_distrib, plot_boxplot_from_distribs,
    plot_boxplot_from_distribs_series)
from variation.matrix.methods import  is_dataset
from variation.variations.index import MIN_NUM_GENOTYPES_FOR_POP_STAT
from variation.variations.stats import (calc_maf, histogram_for_chunks,
                                        histogram, PositionalStatsCalculator,
                                        calc_maf_depth_distribs_per_sample,
                                        calc_missing_gt, calc_obs_het,
                                        calc_obs_het_by_sample,
                                        calc_snp_density, DP_FIELD,
                                        calc_field_distribs_per_sample,
                                        calc_cum_distrib, call_is_hom,
                                        call_is_het, GT_FIELD,
                                        calc_gt_type_stats,
                                        calc_called_gts_distrib_per_depth,
                                        call_is_hom_ref, call_is_hom_alt,
                                        hist2d_allele_observations,
                                        hist2d_gq_allele_observations,
                                        calc_inbreeding_coef, GQ_FIELD,
                                        calc_hwe_chi2_test, calc_expected_het,
                                        CHROM_FIELD, POS_FIELD,
    calc_r2_windows)


def _setup_argparse(**kwargs):
    'It prepares the command line argument parsing.'
    parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument('input', help='Input HDF5 file')
    parser.add_argument('-o', '--output_prefix', required=True,
                        help='Output files prefix')
    parser.add_argument('-n', '--chunk_size', default=SNPS_PER_CHUNK,
                        help='Number of SNPs per chunk (None for all at once)')
    help_msg = 'Comma separated list of depths to calculate missing gts'
    help_msg += ' distributions. Use : separator for range (ex 1:50)'
    parser.add_argument('-d', '--depths', default=STATS_DEPTHS, help=help_msg)
    help_msg = 'Max depth for depth distributions ({})'.format(MAX_DEPTH)
    parser.add_argument('-mxd', '--max_depth', default=MAX_DEPTH,
                        help=help_msg, type=int)
    help_msg = 'Min depth for allele counts 2d distrib ({})'
    parser.add_argument('-mnd', '--min_depth', default=DEF_MIN_DEPTH,
                        help=help_msg.format(DEF_MIN_DEPTH), type=int)
    help_msg = 'Window size to calculate SNP density ({})'
    help_msg = help_msg.format(SNP_DENSITY_WINDOW_SIZE)
    parser.add_argument('-w', '--window_size', default=SNP_DENSITY_WINDOW_SIZE,
                        help=help_msg, type=int)
    help_msg = 'Window size to calculate stats along genome ({})'
    parser.add_argument('-wd', '--manhattan_ws', default=MANHATTAN_WINDOW_SIZE,
                        help=help_msg.format(MANHATTAN_WINDOW_SIZE), type=int)
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
    help_msg = 'Write also bedgraph files for stats calculated along genome'
    parser.add_argument('-bg', '--write_bedgraph', default=False,
                        help=help_msg, action='store_true')
    help_msg = 'Calculate genome-wise statistics and plots'
    parser.add_argument('-g', '--calc_genome_wise', default=False,
                        help=help_msg, action='store_true')
    return parser


def _parse_args(parser):
    parsed_args = parser.parse_args()
    args = {}
    args['in_fpath'] = parsed_args.input
    args['out_prefix'] = parsed_args.output_prefix
    if parsed_args.chunk_size == 'None':
        args['chunk_size'] = None
    else:
        args['chunk_size'] = parsed_args.chunk_size

    if ':' in parsed_args.depths:
        start, stop = [int(x) for x in parsed_args.depths.split(':')]
        args['depths'] = range(start, stop)
    else:
        args['depths'] = [int(x) for x in parsed_args.depths.split(',')]
    args['max_depth'] = parsed_args.max_depth
    args['min_depth'] = parsed_args.min_depth
    args['window_size'] = parsed_args.window_size
    args['manhattan_ws'] = parsed_args.manhattan_ws
    args['min_num_genotypes'] = parsed_args.min_n_gts
    args['max_num_alleles'] = parsed_args.max_num_alleles
    args['max_gq'] = parsed_args.max_gq
    args['max_allele_counts'] = parsed_args.max_allele_counts
    args['write_bedgraph'] = parsed_args.write_bedgraph
    args['calc_genome_wise'] = parsed_args.calc_genome_wise
    return args


def _log_info(logging, msg, print_time=True):
    if print_time:
        msg = '{} {}'.format(time.strftime("%Y-%m-%d %H:%M"), msg)
    logging.info(msg)
    


def create_plots():
    description = 'Calculates basic stats of a HDF5 file'
    parser = _setup_argparse(description=description)
    args = _parse_args(parser)
    
    data_dir = args['out_prefix']
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    logging.basicConfig(filename=join(data_dir, 'plots_info.log'),
                        filemode='w', level=logging.INFO)
        
    h5 = VariationsH5(args['in_fpath'], mode='r',
                      vars_in_chunk=args['chunk_size'])
    
    chunk_size = args['chunk_size']
    manhattan_ws = args['manhattan_ws']
    min_num_genotypes = args['min_num_genotypes']
    write_bg = args['write_bedgraph']
    calc_genome_wise = args['calc_genome_wise']
    
    _log_info(logging, 'Plotting MAF Distribution')
    plot_maf(h5, data_dir, chunk_size=chunk_size, window_size=manhattan_ws,
             min_num_genotypes=min_num_genotypes, write_bg=write_bg,
             calc_genome_wise=calc_genome_wise)
        
    _log_info(logging, 'Plotting Depth based MAF per Sample Distributions')
    plot_maf_depth(h5, data_dir, min_depth=args['min_depth'],
                   chunk_size=chunk_size)
          
    _log_info(logging, 'Plotting Missing Genotype rate per SNP')
    plot_missing_gt_rate_per_snp(h5, data_dir, chunk_size)
          
    _log_info(logging, 'Plotting Observed Heterozygosity')
    plot_obs_het(h5, data_dir, chunk_size=chunk_size,
                 min_num_genotypes=min_num_genotypes)
          
    _log_info(logging, 'Plotting SNP Density Distribution')
    plot_snp_dens_distrib(h5, args['window_size'], data_dir, write_bg=write_bg)
          
    _log_info(logging, 'Plotting Depth Distribution')
    plot_call_field_distribs_per_gt_type(h5, field=DP_FIELD,
                                         max_value=args['max_depth'],
                                         data_dir=data_dir,
                                         chunk_size=chunk_size)
          
    _log_info(logging, 'Plotting Genotypes Quality Distribution')
    plot_call_field_distribs_per_gt_type(h5, field=GQ_FIELD,
                                         max_value=args['max_gq'],
                                         data_dir=data_dir,
                                         chunk_size=chunk_size)
          
    _log_info(logging, 'Plotting Genotypes Statistics per Sample')
    plot_gt_stats_per_sample(h5, data_dir, chunk_size=chunk_size)
           
    _log_info(logging, 'Plotting number of Samples with higher Depth Distribution')
    plot_called_gts_distrib_per_depth(h5, args['depths'], data_dir,
                                      chunk_size=SNPS_PER_CHUNK)
           
    _log_info(logging, 'Plotting Hardy-Weinberg Equilibrium')
    plot_hwe(h5, args['max_num_alleles'], data_dir, ploidy=2,
             min_num_genotypes=min_num_genotypes, chunk_size=chunk_size)
          
         
    if calc_genome_wise:
        _log_info(logging, 'Plotting Nucleotide Diversity Measures')
        plot_nucleotide_diversity_measures(h5, args['max_num_alleles'],
                                           args['manhattan_ws'],
                                           data_dir, chunk_size=chunk_size,
                                           write_bg=write_bg,
                                           min_num_genotypes=min_num_genotypes)
        
        _log_info(logging, 'Plotting LD r2')
        plot_r2(h5, manhattan_ws, data_dir, write_bg=write_bg)
            
    _log_info(logging, 'Plotting Allele Observations Distribution 2 Dimensions')
    plot_allele_obs_distrib_2D(h5, data_dir, args['max_allele_counts'],
                               chunk_size=chunk_size)
         
    _log_info(logging, 'Plotting Inbreeding Coefficient')
    plot_inbreeding_coefficient(h5, args['max_num_alleles'],
                                data_dir, window_size=args['manhattan_ws'],
                                chunk_size=chunk_size,
                                min_num_genotypes=min_num_genotypes,
                                write_bg=write_bg,
                                calc_genome_wise=calc_genome_wise)

def _load_matrix(variations, path):
    matrix = variations[path]
    if is_dataset(matrix):
        matrix = matrix[:]
    return matrix


def plot_maf(variations, data_dir, chunk_size=SNPS_PER_CHUNK, window_size=None,
             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, write_bg=False,
             calc_genome_wise=False):
    # Calculate and plot MAF distribution
    mafs = calc_maf(variations, min_num_genotypes, chunk_size)
    maf_distrib, bins = histogram(mafs, n_bins=25, range_=(0, 1))
    
    fpath = join(data_dir, 'mafs.png')
    title = 'Maximum allele frequency (MAF) distribution'
    plot_distrib(maf_distrib, bins=bins, fhand=open(fpath, 'w'), color='c',
                   mpl_params={'set_xlabel': {'args': ['MAF'], 'kwargs': {}},
                               'set_ylabel': {'args': ['SNP number'],
                                              'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}})

    # Write bedgraph file
    if calc_genome_wise:
        chrom = _load_matrix(variations, CHROM_FIELD)
        pos = _load_matrix(variations, POS_FIELD) 
        bg_fhand = open(join(data_dir, 'maf.bg'), 'w')
        pos_maf = PositionalStatsCalculator(chrom, pos, mafs,
                                            window_size=window_size,
                                            step=window_size)
        if write_bg:
            pos_maf.write(bg_fhand, 'MAF', 'Maximum allele frequency',
                          track_type='bedgraph')
        if window_size is not None:
            pos_maf = pos_maf.calc_window_stat()
        
    
        # Manhattan plot for MAF along genome
        fpath = join(data_dir, 'maf_manhattan.png')
        fhand = open(fpath, 'w')
        title = 'Max Allele Freq (MAF) along the genome'
        chrom, pos, mafs = pos_maf.chrom, pos_maf.pos, pos_maf.stat
        mpl_params = {'set_xlabel': {'args': ['Chromosome'], 'kwargs': {}},
                      'set_ylabel': {'args': ['MAF'],'kwargs': {}},
                      'set_title': {'args': [title], 'kwargs': {}}}
        manhattan_plot(chrom, pos, mafs, mpl_params=mpl_params,
                       fhand=fhand, figsize=(15, 7.5))
    

def plot_maf_depth(variations, data_dir, min_depth=DEF_MIN_DEPTH,
                chunk_size=SNPS_PER_CHUNK):
    
    maf_dp_distribs = calc_maf_depth_distribs_per_sample(variations,
                                                         min_depth=min_depth,
                                                         n_bins=100,
                                                         chunk_size=SNPS_PER_CHUNK)
    maf_dp_distribs, bins = maf_dp_distribs

    maf_dp_dir = os.path.join(data_dir, 'maf_depth')
    if not os.path.exists(maf_dp_dir):
        os.mkdir(maf_dp_dir)
    
    samples = variations.samples
    if samples is None:
        samples = range(maf_dp_distribs.shape[0])
    
    for sample, distrib in zip(samples, maf_dp_distribs):
        fpath = join(maf_dp_dir, '{}.png'.format(sample))
        title = 'Depth based Maximum allele frequency (MAF) distribution {}'
        title = title.format(sample)
        mpl_params = {'set_xlabel': {'args': ['MAF (depth)'], 'kwargs': {}},
                      'set_ylabel': {'args': ['SNPs number'], 'kwargs': {}},
                      'set_title': {'args': [title], 'kwargs': {}},
                      'set_yscale': {'args': ['log'], 'kwargs': {}}}
        plot_distrib(distrib, bins, fhand=open(fpath, 'w'), figsize=(10, 10),
                     mpl_params=mpl_params, n_ticks=10)


def plot_missing_gt_rate_per_snp(variations, data_dir,
                                 chunk_size=SNPS_PER_CHUNK):
    _calc_missing_gt = partial(calc_missing_gt, rates=True, axis=1)
    distrib, bins = histogram_for_chunks(variations,
                                         calc_funct=_calc_missing_gt,
                                         range_=(0, 1), n_bins=20,
                                         chunk_size=chunk_size) 
    
    fpath = join(data_dir, 'missing_gt_rate.png')
    title = 'Missing Genotype rates per SNP distribution'
    plot_distrib(distrib, bins, fhand=open(fpath, 'w'), color='c',
                 mpl_params={'set_xlabel': {'args': ['Missing GT rate'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['SNP number'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}})


def plot_obs_het(variations, data_dir, chunk_size=SNPS_PER_CHUNK,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    # Calculate observed heterozygosity distribution by snp
    _calc_obs_het_by_var = partial(calc_obs_het,
                                   min_num_genotypes=min_num_genotypes)
    distrib = histogram_for_chunks(variations, calc_funct=_calc_obs_het_by_var,
                                   n_bins=25, range_=(0, 1),
                                   chunk_size=chunk_size)
    obs_het_var_distrib, bins1 = distrib
    
    # Calculate observed heterozygosity distribution by sample
    obs_het_by_sample = calc_obs_het_by_sample(variations,
                                               chunk_size=chunk_size)
    obs_het_sample_distrib, bins2 = histogram(obs_het_by_sample, n_bins=25,
                                              range_=(0, 1))
    
    # Plot distributions
    fpath = join(data_dir, 'obs_het.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    axes = fig.add_subplot(211)
    title = 'SNP observed Heterozygosity distribution'
    plot_distrib(obs_het_var_distrib, bins=bins1, fhand=open(fpath, 'w'),
                 mpl_params={'set_xlabel': {'args': ['Heterozygosity'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['SNP number'], 'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}},
                             'set_yscale': {'args': ['log'], 'kwargs': {}}},
                 axes=axes, color='c')
    axes = fig.add_subplot(212)
    title = 'Sample observed Heterozygosity distribution'
    plot_distrib(obs_het_sample_distrib, bins=bins2, fhand=open(fpath, 'w'),
                 mpl_params={'set_xlabel': {'args': ['Heterozygosity'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Sample number'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}}},
                 axes=axes, color='c')
    canvas.print_figure(fhand)


def plot_snp_dens_distrib(variations, window_size, data_dir, write_bg=False):
    # Calculate and plot variations density distribution
    density = calc_snp_density(variations, window_size)
    density_distrib, bins = histogram(density, 20)
    fpath = join(data_dir, 'snps_density.png')
    title = 'SNP density distribution per {} bp windows'.format(window_size)
    plot_distrib(density_distrib, bins, fhand=open(fpath, 'w'), color='c',
                 mpl_params={'set_xlabel': {'args': ['SNP density'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['SNP number'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}},
                             'set_yscale': {'args': ['log'], 'kwargs': {}}})

    # Manhattan plot for SNP density
    fpath = join(data_dir, 'snps_density_manhattan.png')
    fhand = open(fpath, 'w')
    title = 'SNP denisity along the genome'
    chrom = _load_matrix(variations, CHROM_FIELD)
    pos = _load_matrix(variations, POS_FIELD)
    manhattan_plot(chrom, pos, density,
                   mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                              'kwargs': {}},
                               'set_ylabel': {'args': ['SNP per {} bp'.format(window_size)],
                                              'kwargs': {}},
                               'set_title': {'args': [title], 'kwargs': {}}},
                   fhand=fhand, figsize=(15, 7.5), ylim=1)
    
    # Save in bedgraph format
    if write_bg:
        bg_fhand = open(join(data_dir, 'snp_density.bg'), 'w')
        pos_dens = PositionalStatsCalculator(chrom, pos, density)
        pos_dens.write(bg_fhand, 'snp_density',
                       'SNP number in {} bp around'.format(window_size),
                       track_type='bedgraph')


def plot_call_field_distribs_per_gt_type(variations, field, max_value,
                                         data_dir, chunk_size=SNPS_PER_CHUNK):
    # Field distribution per sample
    field_name = field.split('/')[-1]
    fpath = join(data_dir, '{}_distribution_per_sample.png'.format(field_name))
    mask_funcs = [call_is_het, call_is_hom]
    names = ['Heterozygous', 'Homozygous']
    distribs = []
    for mask_func in mask_funcs:
        dp_distribs, bins = calc_field_distribs_per_sample(variations,
                                                           field=field,
                                                           range_=(0, max_value),
                                                           n_bins=max_value,
                                                           chunk_size=chunk_size,
                                                           mask_func=mask_func,
                                                           mask_field=GT_FIELD)
        distribs.append(dp_distribs)
        
    title = '{} distribution per sample'.format(field_name)
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': [field_name], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    figsize = (variations[GT_FIELD].shape[1], 7)
    plot_boxplot_from_distribs_series(distribs, fhand=open(fpath, 'w'),
                                      mpl_params=mpl_params, figsize=figsize,
                                      colors=['pink', 'tan'],
                                      labels=names,
                                      xticklabels=variations.samples)
    
    # Overall field distributions
    fpath = join(data_dir, '{}_distribution.png'.format(field_name))
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(20, 15))
    canvas = FigureCanvas(fig)
    i = 1
    for distrib, name in zip(distribs, names):
        distrib = numpy.sum(dp_distribs, axis=0)
        distrib_cum = calc_cum_distrib(distrib)
        axes = fig.add_subplot(len(names) * 100 + 20 + i)
        i += 1
        title = '{} distribution all samples {}'.format(field_name, name)
        plot_distrib(distrib, bins, axes=axes,
                     mpl_params={'set_xlabel': {'args': [field_name],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['Number of GTs'],
                                                'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}})
        distrib_cum = distrib_cum/distrib_cum[0] * 100
        axes = fig.add_subplot(len(names) * 100 + 20 + i)
        i += 1
        title = '{} cumulative distribution all samples {}'.format(field_name,
                                                                   name)
        plot_distrib(distrib_cum, bins, axes=axes,
                     mpl_params={'set_xlabel': {'args': [field_name],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['% calls > Depth '],
                                                'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}})
    canvas.print_figure(fhand)


def plot_gt_stats_per_sample(variations, data_dir, chunk_size=SNPS_PER_CHUNK):
    gt_stats = calc_gt_type_stats(variations, chunk_size=chunk_size)
    gt_stats = gt_stats.transpose()
    figsize = (variations[GT_FIELD].shape[1], 7)
    
    # All genotypes classes per sample
    fpath = join(data_dir, 'genotype_counts_per_sample.png')
    title = 'Genotypes counts per sample'
    mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Number of GTs'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}}}
    samples = variations.samples
    if samples is not None:
        mpl_params['set_xticklabels'] = {'args': [samples], 'kwargs': {}}
    plot_barplot(gt_stats, ['Ref Homozygous', 'Heterozygous', 'Alt Homozygous',
                            'Missing GT'], mpl_params=mpl_params, 
                 color=['darkslategrey', 'c', 'paleturquoise', 'cadetblue'],
                 fpath=fpath, stacked=True, figsize=figsize)

    # Missing per sample
    fpath = join(data_dir, 'missing_per_sample.png')
    title = 'Missing genotypes counts per sample'
    mpl_params['set_ylabel'] = {'args': ['Missing Genotypes Number'], 'kwargs': {}}
    mpl_params['set_title'] = {'args': [title], 'kwargs': {}}
    plot_barplot(gt_stats[:, -1], ['Missing GT'], mpl_params=mpl_params,
                 fpath=fpath, stacked=True, figsize=figsize)

    # Heterozygous per sample
    fpath = join(data_dir, 'het_per_sample.png')
    title = 'Heterozygous counts per sample'
    mpl_params['set_ylabel'] = {'args': ['Heterozygous Number'], 'kwargs': {}}
    mpl_params['set_title'] = {'args': [title], 'kwargs': {}}
    plot_barplot(gt_stats[:, 1], ['Heterozygous'], mpl_params=mpl_params,
                 fpath=fpath, stacked=True, figsize=figsize)

    # GT percentage without missing values
    fpath = join(data_dir, 'gt_perc_per_sample.png')
    title = 'Genotypes percentage per sample'
    mpl_params['set_ylabel'] = {'args': ['% Genotypes'], 'kwargs': {}}
    mpl_params['set_title'] = {'args': [title], 'kwargs': {}}
    gt_perc = gt_stats[:, :-1] / gt_stats[:, :-1].sum(axis=1, keepdims=True)
    gt_perc *= 100
    plot_barplot(gt_perc, ['Ref Homozygous', 'Heterozygous', 'Alt Homozygous'],
                 mpl_params=mpl_params, fpath=fpath, figsize=figsize)


def plot_called_gts_distrib_per_depth(h5, depths, data_dir,
                                      chunk_size=SNPS_PER_CHUNK):
    # Distribution of the number of samples with a depth higher than
    # given values
    distribs, _ = calc_called_gts_distrib_per_depth(h5, depths=depths,
                                                    chunk_size=chunk_size)
    
    fpath = join(data_dir, 'gts_distribution_per_depth.png')
    title = 'Distribution of the number of samples with a depth higher than'
    title += ' given values'
    mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                  'set_ylabel': {'args': ['Number of samples'], 'kwargs': {}},
                  'set_title': {'args': [title], 'kwargs': {}},
                  'set_xticklabels': {'args': [depths],
                                      'kwargs': {'rotation': 90}}}
    plot_boxplot_from_distribs(distribs, fhand=open(fpath, 'w'),
                               figsize=(15, 10), mpl_params=mpl_params,
                               color='tan')


def plot_allele_obs_distrib_2D(variations, data_dir, max_allele_counts,
                               chunk_size=SNPS_PER_CHUNK):
    # Allele observation distribution 2D
    masks = [call_is_het, call_is_hom_alt, call_is_hom_ref]
    names = ['Heterozygous', 'Alt Homozygous', 'Ref Homozygous']
    
    fig = Figure(figsize=(22, 25))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(3, 2)
    fpath = join(data_dir, 'allele_obs_distrib_per_gt.png')
    fhand = open(fpath, 'w')
    
    counts_range = [[0, max_allele_counts], [0, max_allele_counts]]
    
    for i, (mask_func, name) in enumerate(zip(masks, names)):
        hist2d = hist2d_allele_observations(variations,
                                            n_bins=max_allele_counts,
                                            range_=counts_range,
                                            mask_func=mask_func,
                                            chunk_size=chunk_size)
        counts_distrib2d, xbins, ybins = hist2d
        
        axes = fig.add_subplot(gs[i, 0])
        title = 'Allele counts distribution 2D {}'.format(name)
        plot_hist2d(numpy.log10(counts_distrib2d), xbins, ybins, axes=axes,
                    mpl_params={'set_xlabel': {'args': ['Alt allele counts'],
                                               'kwargs': {}},
                                'set_ylabel': {'args': ['Ref allele counts'],
                                               'kwargs': {}},
                                'set_title': {'args': [title], 'kwargs': {}}},
                    colorbar_label='log10(counts)', fig=fig)

        hist2d = hist2d_gq_allele_observations(variations,
                                               n_bins=max_allele_counts,
                                               range_=counts_range,
                                               mask_func=mask_func,
                                               chunk_size=chunk_size,
                                               hist_counts=counts_distrib2d)
        gq_distrib2d, xbins, ybins = hist2d
        
        axes = fig.add_subplot(gs[i, 1])
        title = 'Allele counts GQ distribution 2D {}'.format(name)
        plot_hist2d(gq_distrib2d, xbins, ybins, axes=axes, fig=fig,
                    mpl_params={'set_xlabel': {'args': ['Alt allele counts'],
                                               'kwargs': {}},
                                'set_ylabel': {'args': ['Ref allele counts'],
                                               'kwargs': {}},
                                'set_title': {'args': [title], 'kwargs': {}}},
                    colorbar_label='Genotype Quality (GQ)')

    canvas.print_figure(fhand)


def plot_inbreeding_coefficient(variations, max_num_allele,  data_dir,
                                window_size, chunk_size=SNPS_PER_CHUNK,
                                min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                write_bg=False, calc_genome_wise=False):
    # Calculate Inbreeding coefficient distribution
    inbreed_coef = calc_inbreeding_coef(variations, chunk_size=chunk_size,
                                        min_num_genotypes=min_num_genotypes)
    ic_distrib, bins = histogram(inbreed_coef, 50, range_=(-1, 1))
      
    fpath = join(data_dir, 'inbreeding_coef_distribution.png')
    fhand = open(fpath, 'w')
    title = 'Inbreeding coefficient distribution all samples'
    plot_distrib(ic_distrib, bins, fhand=fhand,
                 mpl_params={'set_xlabel': {'args': ['Inbreeding coefficient'],
                                            'kwargs': {}},
                             'set_ylabel': {'args': ['Number of SNPs'],
                                            'kwargs': {}},
                             'set_title': {'args': [title], 'kwargs': {}},
                             'set_xlim': {'args': [-1, 1], 'kwargs': {}}})
    
    # Save in bedgraph file
    if calc_genome_wise:
        bg_fhand = open(join(data_dir, 'ic.bg'), 'w')
        chrom = _load_matrix(variations, CHROM_FIELD)
        pos = _load_matrix(variations, POS_FIELD)
        pos_ic = PositionalStatsCalculator(chrom, pos, inbreed_coef)
        if write_bg:
            pos_ic.write(bg_fhand, 'IC', 'Inbreeding coefficient',
                              track_type='bedgraph')
        
        # Plot Ic along genome taking sliding windows
        pos_ic = pos_ic.calc_window_stat()
        chrom, pos, ic_windows = pos_ic.chrom, pos_ic.pos, pos_ic.stat 
        fpath = join(data_dir, 'ic_manhattan.png')
        fhand = open(fpath, 'w')
        title = 'Inbreeding coefficient (IC) along the genome'
        manhattan_plot(chrom, pos, ic_windows, fhand=fhand, figsize=(15, 7.5),
                       ylim=-1,
                       mpl_params={'set_xlabel': {'args': ['Chromosome'],
                                                'kwargs': {}},
                                 'set_ylabel': {'args': ['IC'],
                                                'kwargs': {}},
                                 'set_title': {'args': [title], 'kwargs': {}}})
    

def plot_hwe(variations, max_num_alleles, data_dir, ploidy=2,
             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
             chunk_size=SNPS_PER_CHUNK):
    fpath = join(data_dir, 'hwe_chi2_distrib.png')
    fhand = open(fpath, 'w')
    fig = Figure(figsize=(10, 20))
    canvas = FigureCanvas(fig)
    
    num_alleles = range(2, max_num_alleles + 1)
    gs = gridspec.GridSpec(len(num_alleles), 1)
    for i, num_allele in enumerate(num_alleles):
        df = len(list(combinations_with_replacement(range(num_allele),
                                                    ploidy))) - num_allele
                                                    
        hwe_test =  calc_hwe_chi2_test(variations, num_allele=num_allele,
                                       min_num_genotypes=min_num_genotypes,
                                       chunk_size=chunk_size)
        hwe_chi2 = hwe_test[:, 0]
        hwe_chi2_distrib, bins = histogram(hwe_chi2, n_bins=50)
        
        # Plot observed distribution
        axes = fig.add_subplot(gs[i, 0])
        title = 'Chi2 df={} statistic values distribution'.format(df)
        mpl_params = {'set_xlabel': {'args': ['Chi2 statistic'], 'kwargs': {}},
                      'set_ylabel': {'args': ['SNP number'], 'kwargs': {}},
                      'set_title': {'args': [title], 'kwargs': {}}}
        plot_distrib(hwe_chi2_distrib, bins, axes=axes, mpl_params=mpl_params)
        
        # Plot expected chi2 distribution
        axes = axes.twinx()
        rv = chi2(df)
        x = numpy.linspace(0, max(hwe_chi2), 1000)
        axes.plot(x, rv.pdf(x), color='b', lw=2, label='Expected Chi2')
        axes.set_ylabel('Expected Chi2 density')
    canvas.print_figure(fhand)


def plot_nucleotide_diversity_measures(variations, max_num_alleles,
                                       window_size, data_dir,
                                       chunk_size=SNPS_PER_CHUNK,
                                       write_bg=False,
                                       min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    fig = Figure(figsize=(20, 20))
    canvas = FigureCanvas(fig)
    marker = 'k'
    chrom = _load_matrix(variations, CHROM_FIELD)
    pos = _load_matrix(variations, POS_FIELD)

    # Number of variable positions per bp
    snp_density = PositionalStatsCalculator(chrom, pos,
                                            numpy.ones(pos.shape),
                                            window_size=window_size,
                                            step=window_size)
    snp_density = snp_density.calc_window_stat()
    bg_fhand = open(join(data_dir, 'diversity_s.bg'), 'w')
    if write_bg:
        snp_density.write(bg_fhand, 's',
                          'SNP density in windows of {} bp'.format(window_size),
                          track_type='bedgraph')
    axes = fig.add_subplot(311)
    title = 'Nucleotide diversity measures averaged in windows of {} bp'
    title = title.format(window_size)
    mpl_params = {'set_title': {'args': [title], 'kwargs': {}},
                  'set_ylabel': {'args': ['SNPs number / bp'], 'kwargs': {}},
                  'set_ylim': {'args': [0, 1.2*numpy.max(snp_density.stat)],
                               'kwargs': {}}}
    manhattan_plot(snp_density.chrom, snp_density.pos, snp_density.stat,
                   mpl_params=mpl_params, axes=axes, ylim=0, show_chroms=False,
                   marker=marker)

    # Watterson estimator of nucleotide diversity
    n_seqs = variations[GT_FIELD].shape[1] * variations[GT_FIELD].shape[2]
    correction_factor = numpy.sum(1 / numpy.arange(1, n_seqs))
    watterson = snp_density
    watterson.stat = watterson.stat / correction_factor
    bg_fhand = open(join(data_dir, 'diversity_s.bg'), 'w')
    description = 'SNP density in windows of {} bp'.format(window_size)
    if write_bg:
        watterson.write(bg_fhand, 's', description, track_type='bedgraph')
    axes = fig.add_subplot(312)
    mpl_params={'set_ylabel': {'args': ['Watterson estimator'], 'kwargs': {}},
                'set_ylim': {'args': [0, 1.2*numpy.max(watterson.stat)],
                             'kwargs': {}}}
    manhattan_plot(watterson.chrom, watterson.pos, watterson.stat,
                   mpl_params=mpl_params, axes=axes, ylim=0, show_chroms=False,
                   marker=marker)

    # Expected heterozygosity (Pi)
    exp_het = calc_expected_het(variations, chunk_size=chunk_size,
                                min_num_genotypes=min_num_genotypes)
    pi = PositionalStatsCalculator(chrom, pos, exp_het,
                                   window_size=window_size, step=window_size)
    pi = pi.calc_window_stat()
    bg_fhand = open(join(data_dir, 'diversity_pi.bg'), 'w')
    description = 'Pi in windows of {} bp'.format(window_size)
    if write_bg:
        pi.write(bg_fhand, 's', description, track_type='bedgraph')
    axes = fig.add_subplot(313)
    mpl_params={'set_xlabel': {'args': ['Chromosome'], 'kwargs': {}},
                'set_ylabel': {'args': ['Pi'], 'kwargs': {}},
                'set_ylim': {'args': [0, 1.2*numpy.max(pi.stat)],
                             'kwargs': {}}}
    manhattan_plot(pi.chrom, pi.pos, pi.stat, axes=axes, ylim=0, marker=marker,
                   mpl_params=mpl_params)
    canvas.print_figure(open(join(data_dir, 'nucleotide_diversity.png'), 'w'))


def plot_r2(variations, window_size, data_dir, write_bg=False):
    
    # Calculate LD r2 parameter in windows
    chrom, pos, r2 = calc_r2_windows(variations, window_size=window_size)
    
    # Plot r2 distribution
    fpath = os.path.join(data_dir, 'r2_distrib.png')
    distrib, bins = histogram(r2, n_bins=50, range_=(0, 1))
    title = 'r2 distribution in windows of {} bp'.format(window_size)
    mpl_params={'set_xlabel': {'args': ['r2'], 'kwargs': {}},
                'set_ylabel': {'args': ['Number of windows'], 'kwargs': {}},
                'set_title': {'args': [title], 'kwargs': {}}}
    plot_distrib(distrib, bins, fhand=open(fpath, 'w'), figsize=(7, 7),
                 mpl_params=mpl_params)
    
    # Manhattan plot
    mask = numpy.logical_not(numpy.isnan(r2))
    chrom = chrom[mask]
    pos = pos[mask]
    r2 = r2[mask]
    fpath = os.path.join(data_dir, 'r2_manhattan.png')
    title = 'r2 along genome in windows of {} bp'.format(window_size)
    mpl_params={'set_xlabel': {'args': ['Chromosome'], 'kwargs': {}},
                'set_ylabel': {'args': ['r2'], 'kwargs': {}},
                'set_title': {'args': [title], 'kwargs': {}}}
    manhattan_plot(chrom, pos, r2, fhand=open(fpath, 'w'), figsize=(15, 7),
                   marker='k', mpl_params=mpl_params)
    
    # Write bg
    if write_bg:
        fpath = os.path.join(data_dir, 'r2_windows_{}.png'.format(window_size))
        bg_fhand = open(fpath, 'w')
        pos_r2 = PositionalStatsCalculator(chrom, pos, r2,
                                           window_size=window_size,
                                           step=window_size,
                                           take_windows=False)
        description = 'mean r2 in windows of {} bp'.format(window_size)
        pos_r2.write(bg_fhand, 'r2', description, track_type='bedgraph')


if __name__ == '__main__':
    create_plots()
