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
                                        calc_allele_obs_distrib_2D)


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
    mafs = _calc_stat(h5, _MafCalculator())
    fpath = join(data_dir, 'mafs.csv')
    df_mafs = DataFrame(mafs)
    _save(fpath, df_mafs)

    rates = _calc_stat(h5, _MissingGTCalculator())
    fpath = join(data_dir, 'missing_gt_rate.csv')
    df_rates = DataFrame(rates)
    _save(fpath, df_rates)

    counts = _calc_stat(h5, _CalledGTCalculator(rate=False))
    fpath = join(data_dir, 'called_gt_counts.csv')
    df_counts = DataFrame(counts)
    _save(fpath, df_counts)

    het = _calc_stat(h5, _ObsHetCalculatorBySnps())
    fpath = join(data_dir, 'obs_het.csv')
    df_het = DataFrame(het)
    _save(fpath, df_het)

    het_sample = _calc_stat(h5, _ObsHetCalculatorBySample())
    fpath = join(data_dir, 'obs_het_per_sample.csv')
    df_het_sample = DataFrame(het_sample)
    _save(fpath, df_het_sample)

    gt_freq = _calc_stat(h5, _CalledGTCalculator(),
                         by_chunk=args['by_chunk'])
    fpath = join(data_dir, 'gt_frequency.csv')
    df_gt_freq = DataFrame(gt_freq)
    _save(fpath, df_gt_freq)

    density = calc_snp_density(h5, args['window_size'])
    fpath = join(data_dir, 'snps_density.csv')
    df_dens = DataFrame(density)
    _save(fpath, df_dens)

    distrib_dp, cum_dp = calc_depth_cumulative_distribution_per_sample(h5,
                                                                by_chunk=True,
                                                                max_depth=args['max_depth'])
    fpath = join(data_dir, 'depth_distribution.csv')
    df_distrib_dp = DataFrame(distrib_dp)
    _save(fpath, df_distrib_dp)
    fpath = join(data_dir, 'depth_cumulative.csv')
    df_cum_dp = DataFrame(cum_dp)
    _save(fpath, df_cum_dp)

    distrib_gq, cum_gq = calc_gq_cumulative_distribution_per_sample(h5, by_chunk=True)
    fpath = join(data_dir, 'gq_distribution.csv')
    df_distrib_gq = DataFrame(distrib_gq)
    _save(fpath, df_distrib_gq)
    fpath = join(data_dir, 'gq_cumulative.csv')
    df_cum_gq = DataFrame(cum_gq)
    _save(fpath, df_cum_gq)

    max_n_alleles = args['max_num_alleles']
    allele_freq = _calc_stat(h5,
                             _AlleleFreqCalculator(max_num_allele=max_n_alleles),
                             by_chunk=True)
    fpath = join(data_dir, 'allele_frequency.csv')
    df_allele_freq = DataFrame(allele_freq)
    _save(fpath, df_allele_freq)

    exp_het = calc_expected_het(allele_freq)
    fpath = join(data_dir, 'het_expected.csv')
    df_exp_het = DataFrame(exp_het)
    _save(fpath, df_exp_het)

    inbreeding_coef = calc_inbreeding_coeficient(het, exp_het)
    fpath = join(data_dir, 'inbreeding_coeficient.csv')
    df_inbreeding_coef = DataFrame(inbreeding_coef)
    _save(fpath, df_inbreeding_coef)

    result = calc_depth_cumulative_distribution_per_sample(h5,
                                                           max_depth=args['max_depth'],
                                                           mask_function=_is_het,
                                                           mask_field='/calls/GT')
    distrib_dp_het, cum_dp_het = result
    fpath = join(data_dir, 'distribution_het_by_depth.csv')
    df_distrib_dp_het = DataFrame(distrib_dp_het)
    _save(fpath, df_distrib_dp_het)
    fpath = join(data_dir, 'cumulative_het_by_depth.csv')
    df_cum_dp_het = DataFrame(cum_dp_het)
    _save(fpath, df_cum_dp_het)

    result2 = calc_depth_cumulative_distribution_per_sample(h5,
                                                            max_depth=args['max_depth'],
                                                            mask_function=_is_hom,
                                                            mask_field='/calls/GT')
    distrib_dp_hom, cum_dp_hom = result2
    fpath = join(data_dir, 'distribution_hom_by_depth.csv')
    df_distrib_dp_hom = DataFrame(distrib_dp_hom)
    _save(fpath, df_distrib_dp_hom)
    fpath = join(data_dir, 'cumulative_hom_by_depth.csv')
    df_cum_dp_hom = DataFrame(cum_dp_hom)
    _save(fpath, df_cum_dp_hom)

    result = calc_gq_cumulative_distribution_per_sample(h5,
                                                        mask_function=_is_het,
                                                        mask_field='/calls/GT')
    distrib_gq_het, cum_gq_het = result
    fpath = join(data_dir, 'distribution_het_by_genotype_quality.csv')
    df_distrib_gq_het = DataFrame(distrib_gq_het)
    _save(fpath, df_distrib_gq_het)
    fpath = join(data_dir, 'cumulative_het_by_genotype_quality.csv')
    df_cum_gq_het = DataFrame(cum_gq_het)
    _save(fpath, df_cum_gq_het)

    result2 = calc_gq_cumulative_distribution_per_sample(h5,
                                                         mask_function=_is_hom,
                                                         mask_field='/calls/GT')
    distrib_gq_hom, cum_gq_hom = result2
    fpath = join(data_dir, 'distribution_hom_by_genotype_quality.csv')
    df_distrib_gq_hom = DataFrame(distrib_gq_hom)
    _save(fpath, df_distrib_gq_hom)
    fpath = join(data_dir, 'cumulative_hom_by_genotype_quality.csv')
    df_cum_gq_hom = DataFrame(cum_gq_hom)
    _save(fpath, df_cum_gq_hom)

    distrib_snv_density = calc_snv_density_distribution(h5, args['window_size'])
    fpath = join(data_dir, 'snv_density_distribution.csv')
    df_distrib_snv_density = DataFrame(distrib_snv_density)
    _save(fpath, df_distrib_snv_density)

    gt_stats = _calc_stat(h5, GenotypeStatsCalculator(), reduce_funct=numpy.add)
    fpath = join(data_dir, 'genotype_basic_stats.csv')
    df_gt_stats = DataFrame(gt_stats)
    _save(fpath, df_gt_stats)

    distrib, cum = calc_called_gts_distrib_per_depth(h5, depths=args['depths'])
    fpath = join(data_dir, 'gts_distribution_per_depth.csv')
    df_distrib = DataFrame(distrib)
    _save(fpath, df_distrib)
    fpath = join(data_dir, 'gts_cumulative_per_depth.csv')
    df_cum = DataFrame(cum)
    _save(fpath, df_cum)

    distrib_gq, cum_gq = calc_quality_by_depth(h5, depths=args['depths'])
    fpath = join(data_dir, 'gts_distribution_per_quality.csv')
    df_distrib_gq = DataFrame(distrib_gq)
    _save(fpath, df_distrib_gq)
    fpath = join(data_dir, 'gts_cumulative_per_quality.csv')
    df_cum_gq = DataFrame(cum_gq)
    _save(fpath, df_cum_gq)

    maf_depths_distrib = calculate_maf_depth_distribution(h5)
    fpath = join(data_dir, 'mafs_depths_distribution.csv')
    df_maf_depths_distrib = DataFrame(maf_depths_distrib)
    _save(fpath, df_maf_depths_distrib)

    allele_distrib_2D = calc_allele_obs_distrib_2D(h5, by_chunk=False)
    fpath = join(data_dir, 'allele_obs_distribution_2D.csv')
    df_allele_distrib_2D = DataFrame(allele_distrib_2D)
    _save(fpath, df_allele_distrib_2D)


#Pandas dataframe(data, index, columns, dtype, copy)
def _save(path, dataframe):
    file = open(path, mode='w')
    dataframe.to_csv(file)

if __name__ == '__main__':
    main()
