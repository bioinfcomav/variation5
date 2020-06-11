# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join
from functools import partial
from io import StringIO
import math

import numpy

from variation import AD_FIELD
from variation.variations.vars_matrices import VariationsH5, VariationsArrays
from variation.variations.stats import (calc_maf, calc_mac, histogram,
                                        histogram_for_chunks,
                                        _calc_maf_depth,
                                        calc_missing_gt, calc_obs_het,
                                        calc_obs_het_by_sample,
                                        calc_gt_type_stats, calc_called_gt,
                                        calc_snp_density, calc_allele_freq,
                                        calc_inbreeding_coef,
                                        calc_hwe_chi2_test,
                                        hist2d_allele_observations,
                                        call_is_het, calc_depth,
                                        hist2d_gq_allele_observations,
                                        calc_called_gts_distrib_per_depth,
                                        calc_field_distribs_per_sample,
                                        calc_maf_depth_distribs_per_sample,
                                        PositionalStatsCalculator, call_is_hom,
                                        calc_cum_distrib, _calc_r2,
                                        calc_r2_windows, GT_FIELD,
                                        hist2d_het_allele_freq,
                                        calc_field_distrib_for_a_sample,
                                        calc_call_dp_distrib_for_a_sample,
                                        calc_depth_mean_by_sample,
                                        calc_stats_by_sample,
                                        histograms_for_columns,
                                        write_stats_by_sample,
                                        calc_expected_het,
                                        calc_unbias_expected_het,
                                        calc_allele_observation_based_maf,
                                        _calc_a1, calc_tajima_d_and_pi)
from variation import DP_FIELD
from test.test_utils import TEST_DATA_DIR


class StatsTest(unittest.TestCase):
    def test_calc_cum_distrib(self):
        distrib = numpy.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        cum_distrib = calc_cum_distrib(distrib)
        exp = [3, 2, 2, 2, 2, 2, 1, 1, 1, 1]
        assert numpy.all(cum_distrib == exp)

        distrib = numpy.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
        cum_distrib = calc_cum_distrib(distrib)
        exp = [3, 2, 2, 2, 2, 2, 1, 1, 1, 1]
        assert numpy.all(cum_distrib == [exp, exp])

    def test_allele_observation_based_maf(self):
        allele_depths = numpy.array([])
        varis = VariationsArrays()
        varis[AD_FIELD] = allele_depths
        maf = calc_allele_observation_based_maf(varis, chunk_size=None)
        assert not list(maf)

        allele_depths_snp1 = [[10, 0, 1], # Allele Obervation in sample1
                              [4, 6, 1]] # Allele Obervation in sample2
        allele_depths_snp2 = [[10, 0, 0], # Allele Obervation in sample1
                              [0, 5, 7]] # Allele Obervation in sample2
        allele_depths_snp3 = [[-1, -1, -1], # Allele Obervation in sample1
                              [-1, -1, -1]] # Allele Obervation in sample2

        allele_depths = numpy.array([allele_depths_snp1,
                                     allele_depths_snp2,
                                     allele_depths_snp3])
        varis = VariationsArrays()
        varis[AD_FIELD] = allele_depths
        maf = calc_allele_observation_based_maf(varis, chunk_size=None)
        expected = [0.63636364, 0.45454545, numpy.nan]
        assert numpy.allclose(maf, expected, equal_nan=True)

        maf = calc_allele_observation_based_maf(varis, chunk_size=1)
        expected = [0.63636364, 0.45454545, numpy.nan]
        assert numpy.allclose(maf, expected, equal_nan=True)

    def test_maf(self):
        gts = numpy.array([])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        mafs = calc_maf(varis, chunk_size=None)
        assert mafs.shape == (0,)
        mafs = calc_maf(varis)
        assert mafs.shape == (0,)

        mafs = calc_mac(varis, chunk_size=None)
        assert mafs.shape == (0,)
        mafs = calc_mac(varis)
        assert mafs.shape == (0,)

        gts = numpy.array([[[0], [0], [0], [0]], [[0], [0], [1], [1]],
                           [[0], [0], [0], [1]], [[-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        mafs = calc_maf(varis, min_num_genotypes=1)
        assert numpy.allclose(mafs, numpy.array([1., 0.5, 0.75, numpy.NaN]),
                              equal_nan=True)

        macs = calc_mac(varis, min_num_genotypes=1)
        assert numpy.allclose(macs, numpy.array([4, 2, 3, numpy.NaN]),
                              equal_nan=True)

        varis = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        mafs = calc_maf(varis)
        assert numpy.all(mafs[numpy.logical_not(numpy.isnan(mafs))] >= 0.5)
        assert numpy.all(mafs[numpy.logical_not(numpy.isnan(mafs))] <= 1)
        assert mafs.shape == (943,)

        macs = calc_mac(varis)
        # assert macs.shape == (943,)
        min_mac = varis['/calls/GT'].shape[1] / 2
        max_mac = varis['/calls/GT'].shape[1]
        assert numpy.all(macs[numpy.logical_not(numpy.isnan(mafs))] >= min_mac)
        assert numpy.all(macs[numpy.logical_not(numpy.isnan(mafs))] <= max_mac)

    def test_calc_maf_distrib(self):
        gts = numpy.array([[[0], [0], [0], [0]], [[0], [0], [1], [1]],
                           [[0], [0], [0], [1]], [[-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis['/calls/GT'] = gts
        mafs = calc_maf(varis, min_num_genotypes=1)
        distrib, bins = histogram(mafs, n_bins=10)
        dist_expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        bins_expected = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                         1.]
        assert numpy.allclose(bins, bins_expected)
        assert numpy.allclose(distrib, dist_expected)

        varis = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        mafs = calc_maf(varis, min_num_genotypes=1)
        distrib, bins = histogram(mafs, n_bins=10)
        dist_expected = [53, 72, 77, 66, 73, 129, 74, 73, 49, 277]
        bins_expected = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                         1.]
        assert numpy.allclose(bins, bins_expected)
        assert numpy.allclose(distrib, dist_expected)

    def test_calc_maf_distrib_by_chunk(self):
        varis = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        calc_maf_for_chunk = partial(calc_maf, min_num_genotypes=1,
                                     chunk_size=None)

        distrib, bins = histogram_for_chunks(varis, calc_maf_for_chunk,
                                             n_bins=10)
        dist_expected = [53, 72, 77, 66, 73, 129, 74, 73, 49, 277]

        bins_expected = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                         1.]
        assert numpy.allclose(bins, bins_expected)
        assert numpy.allclose(distrib, dist_expected)

    def test_calculate_maf_depth(self):
        variations = {'/calls/AO': numpy.array([[[0, 0], [5, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [20, 0], [25, 0], [20, 20],
                                                 [0, 0], [20, 0]]]),
                      '/calls/RO': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10],
                                                [0]])}
        expected = numpy.array([[1, 0.5, 1, 1, 1, 1, 1, 0.5, 1 / 3, 1, 1]])
        assert numpy.allclose(_calc_maf_depth(variations, min_depth=0),
                              expected)
        expected = numpy.array([[1, 0.5, 1, 1, 1, 1, 1, 0.5, 1 / 3, 1, 1]])
        expected = [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
                    numpy.nan, 1.0, 0.5, 0.33333333333333331, numpy.nan, 1.0]
        assert numpy.allclose(_calc_maf_depth(variations), expected,
                              equal_nan=True)

    def test_calc_missing_gt_rates(self):
        gts = numpy.array([])
        varis = {'/calls/GT': gts}
        called_vars = calc_called_gt(varis, rates=False)
        assert called_vars.shape[0] == 0
        called_vars = calc_called_gt(varis, rates=True)
        assert called_vars.shape[0] == 0

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        arrays = VariationsArrays()
        arrays.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        rates = calc_missing_gt(arrays)
        rates2 = calc_missing_gt(hdf5)
        assert rates.shape == (943,)
        assert numpy.allclose(rates, rates2)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

        gts = numpy.array([[[0, 0], [0, 0]], [[0, 0], [-1, -1]],
                           [[0, 0], [-1, -1]], [[-1, -1], [-1, -1]]])
        varis = {'/calls/GT': gts}
        expected = numpy.array([2, 1, 1, 0])
        called_vars = calc_called_gt(varis, rates=False)
        assert numpy.all(called_vars == expected)

        missing_vars = calc_missing_gt(varis, rates=False)
        assert numpy.all(missing_vars == 2 - expected)

        expected = numpy.array([0, 0.5, 0.5, 1])
        rates = calc_called_gt(varis)
        assert numpy.allclose(rates, 1 - expected)

        rates = calc_missing_gt(varis)
        assert numpy.allclose(rates, expected)

    def test_calc_obs_het(self):
        gts = numpy.array([])
        dps = numpy.array([])
        varis = {'/calls/GT': gts, '/calls/DP': dps}
        het = calc_obs_het(varis, min_num_genotypes=0)
        assert het.shape[0] == 0

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = calc_obs_het(hdf5, min_num_genotypes=0)
        het_array = calc_obs_het(snps, min_num_genotypes=0)
        assert numpy.all(het_array == het_h5)

        gts = numpy.array([[[0, 0], [0, 1], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]]])

        dps = numpy.array([[5, 12, 10, 10],
                           [10, 10, 10, 10]])

        varis = {'/calls/GT': gts, '/calls/DP': dps}
        het = calc_obs_het(varis, min_num_genotypes=0)
        assert numpy.allclose(het, [0.5, 0])

        het = calc_obs_het(varis, min_num_genotypes=10)
        assert numpy.allclose(het, [numpy.NaN, numpy.NaN], equal_nan=True)

        het = calc_obs_het(varis, min_num_genotypes=0, min_call_dp=10)
        assert numpy.allclose(het, [1, 0])

        het = calc_obs_het(varis, min_num_genotypes=0, max_call_dp=11)
        assert numpy.allclose(het, [0, 0])

        het = calc_obs_het(varis, min_num_genotypes=0, min_call_dp=5)
        assert numpy.allclose(het, [0.5, 0])

    def test_calc_gt_type_stats(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = calc_gt_type_stats(hdf5)
        assert result.shape == (4, 153)
        assert numpy.all(numpy.sum(result, axis=0) == 943)

        gts = numpy.array([[[0, 0], [1, 1], [0, -1], [-1, -1]],
                           [[0, -1], [0, 0], [0, -1], [-1, -1]],
                           [[0, 1], [0, 0], [0, 0], [-1, -1]]])

        varis = {'/calls/GT': gts}
        res = calc_gt_type_stats(varis)
        expected = [[1, 2, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 2, 3]]
        assert numpy.all(res == expected)

    def test_calc_snp_density(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        density_h5 = list(calc_snp_density(hdf5, 1000))
        density_array = list(calc_snp_density(snps, 1000))
        assert density_array == density_h5
        var = {'/variations/chrom': numpy.array(['ch', 'ch', 'ch', 'ch', 'ch',
                                                 'ch', 'ch', 'ch', 'ch', 'ch',
                                                 'ch', 'ch', 'ch', 'ch']),
               '/variations/pos': numpy.array([1, 2, 3, 4, 5, 6, 7, 25, 34, 44,
                                               80, 200, 300, 302])}
        dens_var = list(calc_snp_density(var, 11))
        expected = [6, 7, 7, 7, 7, 7, 6, 1, 1, 1, 1, 1, 2, 2]
        assert dens_var == expected

        var = {'/variations/chrom': numpy.array(['ch', 'ch', 'ch', 'c2', 'c2',
                                                 'c2', 'c2', 'c2', 'c2', 'c2',
                                                 'c2', 'c2', 'c2', 'c3']),
               '/variations/pos': numpy.array([1, 2, 3, 4, 5, 6, 7, 25, 34, 44,
                                               80, 200, 300, 302])}
        dens_var = list(calc_snp_density(var, 11))
        expected = [3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1]
        assert dens_var == expected

        var = {'/variations/chrom': numpy.array(['c1', 'c4', 'c5', 'c2', 'c2',
                                                 'c2', 'c2', 'c2', 'c2', 'c2',
                                                 'c2', 'c2', 'c2', 'c3']),
               '/variations/pos': numpy.array([1, 2, 3, 4, 5, 6, 7, 25, 34, 44,
                                               80, 200, 300, 302])}
        dens_var = list(calc_snp_density(var, 11))
        expected = [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1]
        assert dens_var == expected

        var = {'/variations/chrom': numpy.array([]),
               '/variations/pos': numpy.array([])}
        dens_var = list(calc_snp_density(var, 11))
        assert dens_var == []

        var = {'/variations/chrom': numpy.array([1]),
               '/variations/pos': numpy.array([1])}
        dens_var = list(calc_snp_density(var, 11))
        assert dens_var == [1]

    def test_calc_allele_freq(self):
        gts = numpy.array([])
        varis = {'/calls/GT': gts, '/variations/alt': numpy.array([])}
        allele_freq = calc_allele_freq(varis, min_num_genotypes=0)
        assert allele_freq.shape[0] == 0

        gts = numpy.array([[[0, 0], [1, 1], [0, -1], [-1, -1]],
                           [[0, -1], [0, 0], [0, -1], [-1, -1]],
                           [[0, 1], [0, 2], [0, 0], [-1, -1]]])
        varis = {'/calls/GT': gts, '/variations/alt': numpy.zeros((3, 2))}
        allele_freq = calc_allele_freq(varis, min_num_genotypes=0)
        expected = numpy.array([[0.6, 0.4, 0], [1, 0, 0],
                                [4 / 6, 1 / 6, 1 / 6]])
        assert numpy.allclose(allele_freq, expected)

    def test_2d_allele_freq_het(self):

        gts = numpy.array([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1], [-1, -1]],
                           ])
        varis = {'/calls/GT': gts}
        res = hist2d_het_allele_freq(varis, min_num_genotypes=0, n_bins=2,
                                     allele_freq_range=(0.5, 1),
                                     het_range=(0, 1), chunk_size=None)
        hist, _, _ = res
        assert numpy.allclose(hist, numpy.array([[0., 0.], [0, 1.]]))

        gts = numpy.array([[[0, 0], [0, 0], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]],
                           [[0, 0], [-1, -1], [-1, -1], [-1, -1]],
                           [[0, 0], [1, 1], [0, 0], [1, 1]],
                           [[0, 0], [1, 1], [0, 0], [1, 1]],
                           [[0, 1], [-1, -1], [-1, -1], [-1, -1]]
                           ])
        varis = {'/calls/GT': gts, '/variations/alt': numpy.zeros((3, 2))}
        res = hist2d_het_allele_freq(varis, min_num_genotypes=0, n_bins=2,
                                     allele_freq_range=(0.5, 1),
                                     het_range=(0, 1), chunk_size=None)
        hist, xedges, yedges = res
        assert numpy.allclose(hist, numpy.array([[1., 2.], [0., 3.]]))
        assert numpy.allclose(xedges, numpy.array([0.5, 0.75, 1.]))
        assert numpy.allclose(yedges, numpy.array([0, 0.5, 1.]))

        res = hist2d_het_allele_freq(varis, min_num_genotypes=2, n_bins=2,
                                     allele_freq_range=(0.5, 1),
                                     het_range=(0, 1), chunk_size=None)
        hist, xedges, yedges = res
        assert numpy.allclose(hist, numpy.array([[0., 2.], [0., 2.]]))

        gts = numpy.array([])
        varis = {'/calls/GT': gts, '/variations/alt': numpy.array([])}
        res = hist2d_het_allele_freq(varis, chunk_size=None)
        assert res[0].shape[0] == 0
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.h5'),
                            mode='r')
        res1 = hist2d_het_allele_freq(hdf5, min_num_genotypes=2,
                                      chunk_size=None)
        res2 = hist2d_het_allele_freq(hdf5, min_num_genotypes=2)
        assert numpy.allclose(res1[0], res2[0])
        assert numpy.allclose(res1[1], res2[1])
        assert numpy.allclose(res1[2], res2[2])

        gts = numpy.array([[[0, 0], [0, 0], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]],
                           [[0, 0], [-1, -1], [-1, -1], [-1, -1]],
                           [[0, 0], [1, 1], [0, 0], [1, 1]],
                           [[0, 0], [1, 1], [0, 0], [1, 1]],
                           [[0, 1], [-1, -1], [-1, -1], [-1, -1]]
                           ])
        dps = numpy.array([[1, 5, 5, 5],
                           [5, 5, 5, 5],
                           [5, 5, 5, 5],
                           [5, 5, 5, 5],
                           [5, 5, 5, 5],
                           [5, 5, 5, 5]
                           ])
        varis = {'/calls/GT': gts, '/variations/alt': numpy.zeros((3, 2)),
                 DP_FIELD: dps}
        res = hist2d_het_allele_freq(varis, min_num_genotypes=2, n_bins=2,
                                     allele_freq_range=(0.5, 1),
                                     min_call_dp_for_het=3,
                                     het_range=(0, 1), chunk_size=None)
        hist, xedges, yedges = res
        assert numpy.allclose(hist, numpy.array([[0., 2.], [0., 1.]]))
        assert numpy.allclose(xedges, numpy.array([0.5, 0.75, 1.]))
        assert numpy.allclose(yedges, numpy.array([0, 0.5, 1.]))

    def test_expected_het(self):
        gts = [[[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 0]],
               [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]],
               [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]]]
        snps = VariationsArrays()
        snps['/calls/GT'] = numpy.array(gts)
        exp = [0.5, 0.48979592, 0.48979592]
        assert numpy.allclose(calc_expected_het(snps, min_num_genotypes=0),
                              exp)
        exp = [0.53846154, 0.52747253, 0.52747253]
        assert numpy.allclose(calc_unbias_expected_het(snps,
                                                       min_num_genotypes=0),
                              exp)

    def test_calc_inbreeding_coeficient(self):
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]]),
                      '/variations/alt': numpy.zeros((1, 1))}
        result = calc_inbreeding_coef(variations, min_num_genotypes=0,
                                      chunk_size=None)
        expected = numpy.array([1 - (0.4 / 0.42)])
        assert numpy.allclose(result, expected)

        variations = {'/calls/GT': numpy.array([]),
                      '/variations/alt': numpy.array([])}
        result = calc_inbreeding_coef(variations, min_num_genotypes=0,
                                      chunk_size=None)
        assert result.shape[0] == 0

    def test_calculate_hwe(self):
        variations = VariationsArrays()
        gts = numpy.array([])
        variations['/calls/GT'] = gts
        variations['/variations/alt'] = gts
        result = calc_hwe_chi2_test(variations, min_num_genotypes=0,
                                    chunk_size=None)
        assert result.shape[0] == 0

        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 0],
                            [0, 0], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [1, 0], [0, 1], [0, 0], [0, 1], [0, 0],
                            [0, 0], [1, 0], [1, 1], [0, 0]]])
        variations['/calls/GT'] = gts
        variations._create_matrix('/variations/alt', shape=(1, 1),
                                  dtype=numpy.int16, fillvalue=0)
        expected = numpy.array([[1.25825397e+01, 1.85240619e-03],
                                [1.25825397e+01, 1.85240619e-03]])
        result = calc_hwe_chi2_test(variations, min_num_genotypes=0,
                                    chunk_size=None)
        assert numpy.allclose(result, expected)

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        hwe_test1 = calc_hwe_chi2_test(hdf5, chunk_size=None)
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        hwe_test2 = calc_hwe_chi2_test(hdf5)
        assert numpy.allclose(hwe_test1, hwe_test2, equal_nan=True)

    def test_calc_allele_obs_distrib_2D(self):
        variations = {'/calls/AO': numpy.array([[[0, 0], [5, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [20, 0], [25, 0], [20, 20],
                                                 [0, 0]]]),
                      '/calls/RO': numpy.array([[0, 5, 15, 7, 10, 0, 0, 25,
                                                 20, 10]]),
                      '/calls/GQ': numpy.array([[40, 30, 35, 30, 0,
                                                 40, 30, 35, 30, 0]]),
                      '/calls/GT': numpy.array([[[0, 0], [1, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [1, 0], [0, 0], [0, 0],
                                                 [1, 0]]])}
        hist, _, ybins = hist2d_allele_observations(variations,
                                                    chunk_size=None)
        assert hist[0, 0] == 1
        assert hist[-1, -1] == 1
        assert ybins[0] == 0

        hist, _, _ = hist2d_allele_observations(variations,
                                                mask_func=call_is_het,
                                                chunk_size=None)
        assert hist[0, 0] == 0
        assert hist[-1, -1] == 0

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        hist, xbins, ybins = hist2d_allele_observations(hdf5,
                                                        mask_func=call_is_het,
                                                        chunk_size=None)
        hist2, xbins2, ybins2 = hist2d_allele_observations(hdf5,
                                                           mask_func=call_is_het,
                                                           chunk_size=10)
        assert numpy.allclose(xbins, xbins2)
        assert numpy.allclose(ybins, ybins2)
        assert numpy.all(hist == hist2)

    def test_calc_allele_obs_gq_distrib_2D(self):
        variations = {'/calls/AO': numpy.array([[[0, 0], [5, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [20, 0], [25, 0], [20, 20],
                                                 [0, 0]]]),
                      '/calls/RO': numpy.array([[0, 5, 15, 7, 10, 0, 0, 25,
                                                 20, 10]]),
                      '/calls/GQ': numpy.array([[40, 30, 35, 30, 0,
                                                 40, 30, 35, 30, 0]]),
                      '/calls/GT': numpy.array([[[0, 0], [1, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [1, 0], [0, 0], [0, 0],
                                                 [1, 0]]])}
        hist, _, _ = hist2d_gq_allele_observations(variations, chunk_size=None)
        assert hist[0, 0] == 40
        assert hist[-1, -1] == 35

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        hist, xbins, ybins = hist2d_gq_allele_observations(hdf5)
        hist2, xbins2, ybins2 = hist2d_gq_allele_observations(hdf5,
                                                              chunk_size=10)
        assert numpy.allclose(xbins, xbins2)
        assert numpy.allclose(ybins, ybins2)
        assert numpy.all(hist == hist2)

    def test_calc_called_gts_distribution_per_depth(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist, _ = calc_called_gts_distrib_per_depth(hdf5, depths=range(30),
                                                    chunk_size=10)
        assert dist[1, 1] == 1
        dist2, _ = calc_called_gts_distrib_per_depth(hdf5, depths=range(30),
                                                     chunk_size=None)
        assert numpy.all(dist == dist2)

        vars_ = VariationsArrays()
        vars_['/calls/GT'] = numpy.array([[[0, 0], [0, 1], [0, 1],
                                           [0, 0], [0, 1], [0, 0],
                                           [0, 0], [0, 1], [1, 1],
                                           [0, 0]]])
        vars_['/calls/DP'] = numpy.array([[10, 5, 15, 7, 10,
                                          0, 0, 25, 20, 10]])
        vars_.samples = list(range(10))
        dist, _ = calc_called_gts_distrib_per_depth(vars_, depths=[0, 5, 10,
                                                                   30])
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert numpy.all(dist == expected)

    def test_to_positional_stats(self):
        chrom = numpy.array(['chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4'])
        pos = numpy.array([10, 5, 20, 30, 40, 50])
        stat = numpy.array([1, 2, 3, 4, 5, numpy.nan])
        pos_stats = PositionalStatsCalculator(chrom, pos, stat)
        line1 = 'track type=wiggle_0 name="track1" description="description"'
        wiglines = [line1,
                    'variableStep chrom=chr1', '10 1.0',
                    'variableStep chrom=chr2', '5 2.0', '20 3.0',
                    'variableStep chrom=chr3', '30 4.0', '40 5.0']
        for line, exp in zip(pos_stats.to_wig(), wiglines[1:]):
            assert line.strip() == exp

        line1 = 'track type=bedGraph name="track1" description="description"'
        bg_lines = [line1,
                    'chr1 10 11 1.0', 'chr2 5 6 2.0', 'chr2 20 21 3.0',
                    'chr3 30 31 4.0', 'chr3 40 41 5.0']
        for line, exp in zip(pos_stats.to_bedGraph(), bg_lines[1:]):
            assert line.strip() == exp

        # Taking windows
        chrom = numpy.repeat('chr1', 5)
        pos = numpy.array([10, 20, 30, 40, 50])
        stat = numpy.array([1, 2, 3, 4, 5])
        pos_stats = PositionalStatsCalculator(chrom, pos, stat, window_size=25,
                                              step=25)
        line1 = 'track type=wiggle_0 name="track1" description="description"'
        wiglines = [line1,
                    'fixedStep chrom=chr1 start=10 span=25 step=25',
                    str(6 / 25), str(9 / 25)]
        for line, exp in zip(pos_stats.to_wig(), wiglines[1:]):
            assert line.strip() == exp

        line1 = 'track type=bedGraph name="track1" description="description"'
        bg_lines = [line1, 'chr1 10 35 {}'.format(6 / 25),
                    'chr1 35 60 {}'.format(9 / 25)]
        for line, exp in zip(pos_stats.to_bedGraph(), bg_lines[1:]):
            assert line.strip() == exp

        # Pre-calculating the windows
        pos_stats = pos_stats.calc_window_stat()
        line1 = 'track type=bedGraph name="track1" description="description"'
        bg_lines = [line1, 'chr1 10 35 {}'.format(6 / 25),
                    'chr1 35 60 {}'.format(9 / 25)]
        for line, exp in zip(pos_stats.to_bedGraph(), bg_lines[1:]):
            assert line.strip() == exp

    def test_calc_r2_windows(self):
        variations = VariationsArrays()
        chrom = numpy.array([b'chr1'] * 4)
        pos = numpy.array([1, 4, 6, 20])
        gts = numpy.array([[[0, 0], [1, 1], [0, 0]],
                           [[0, 0], [1, 1], [0, 0]],
                           [[1, 1], [0, 0], [1, 1]],
                           [[0, 0], [0, 1], [-1, -1]]])
        variations['/variations/chrom'] = chrom
        variations['/variations/pos'] = pos
        variations['/calls/GT'] = gts
        expected = [1.0, 1.0000002, 1.0, 1.0000002, 1.0, 1.0]
        assert numpy.allclose(_calc_r2(gts), expected)

        chrom, pos, r2 = calc_r2_windows(variations, 10)
        assert numpy.allclose(r2, [1.0000002384185933, numpy.nan],
                              equal_nan=True)
        assert numpy.all(chrom == b'chr1')


class SampleStatsTest(unittest.TestCase):
    def test_calc_maf_depth_distribs_per_sample(self):
        variations = VariationsArrays()
        variations['/calls/AO'] = numpy.array([])
        variations['/calls/RO'] = numpy.array([])
        distribs, bins = calc_maf_depth_distribs_per_sample(variations,
                                                            chunk_size=None)
        assert distribs is None
        assert bins is None

        variations = VariationsArrays()
        variations['/calls/AO'] = numpy.array([[[0, 0], [0, 0], [15, -1]]])
        variations['/calls/RO'] = numpy.array([[10, 5, 15]])
        variations.samples = list(range(3))
        distribs, _ = calc_maf_depth_distribs_per_sample(variations, n_bins=4,
                                                         min_depth=6,
                                                         chunk_size=None)
        expected = [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]]
        assert numpy.all(distribs == expected)

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        distribs1, _ = calc_maf_depth_distribs_per_sample(hdf5, min_depth=6,
                                                          chunk_size=None)
        distribs2, _ = calc_maf_depth_distribs_per_sample(hdf5, min_depth=6)
        assert numpy.all(distribs1 == distribs2)

    def test_calc_distrib_for_sample(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        distrib, _ = calc_field_distrib_for_a_sample(hdf5, field='/calls/DP',
                                                     sample='1_17_1_gbs',
                                                     n_bins=15)
        assert distrib.shape == (15,)

        distrib2, _ = calc_field_distrib_for_a_sample(snps, field='/calls/DP',
                                                      n_bins=15,
                                                      sample='1_17_1_gbs',
                                                      chunk_size=None)
        assert numpy.all(distrib == distrib2)

        distrib3, _ = calc_field_distrib_for_a_sample(snps, field='/calls/DP',
                                                      n_bins=15,
                                                      sample='1_17_1_gbs',
                                                      chunk_size=50)
        assert numpy.all(distrib3 == distrib2)

        vars_ = VariationsArrays()
        vars_['/calls/DP'] = numpy.array([[10, 5, 15],
                                          [0, 15, 10]])
        vars_['/calls/GT'] = numpy.array([[[0, 0], [0, 1], [1, 1]],
                                          [[0, 0], [0, 1], [1, 1]]])
        vars_.samples = list(range(3))
        distrib, _ = calc_field_distribs_per_sample(vars_, field='/calls/DP',
                                                    n_bins=16)
        expec = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        assert numpy.all(expec == distrib)
        assert numpy.all(calc_depth(vars_) == [10, 5, 15, 0, 15, 10])

        distrib, _ = calc_field_distribs_per_sample(vars_, field='/calls/DP',
                                                    n_bins=16,
                                                    mask_field='/calls/GT',
                                                    mask_func=call_is_het)
        expec = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert numpy.all(expec == distrib)
        assert numpy.all(calc_depth(vars_) == [10, 5, 15, 0, 15, 10])

        distrib, _ = calc_field_distribs_per_sample(vars_, field='/calls/DP',
                                                    n_bins=16,
                                                    mask_field='/calls/GT',
                                                    mask_func=call_is_hom)
        expec = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        assert numpy.all(expec == distrib)
        assert numpy.all(calc_depth(vars_) == [10, 5, 15, 0, 15, 10])

    def test_calc_dp_for_sample(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        cnts, _ = calc_call_dp_distrib_for_a_sample(hdf5, sample='1_17_1_gbs',
                                                    n_bins=15)
        assert cnts['hom'].shape == (15,)
        assert cnts['het'].shape == (15,)
        return
        cnts2, _ = calc_call_dp_distrib_for_a_sample(hdf5, sample='1_17_1_gbs',
                                                     n_bins=15,
                                                     chunk_size=None)
        assert numpy.all(cnts['hom'] == cnts2['hom'])
        assert numpy.all(cnts['het'] == cnts2['het'])

        cnts3, _ = calc_call_dp_distrib_for_a_sample(hdf5, sample='1_17_1_gbs',
                                                     n_bins=15, chunk_size=50)
        assert numpy.all(cnts['hom'] == cnts3['hom'])
        assert numpy.all(cnts['het'] == cnts3['het'])

    def test_calc_obs_het_sample(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = calc_obs_het_by_sample(hdf5)
        het_array = calc_obs_het_by_sample(snps)
        assert numpy.all(het_array == het_h5)

        gts = numpy.array([[[0, 0], [0, 1], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1]]])

        varis = {'/calls/GT': gts}
        het = calc_obs_het_by_sample(varis, chunk_size=None)
        assert numpy.allclose(het, [0, 1 / 3, 0, numpy.NaN], equal_nan=True)

        gts = numpy.array([])
        varis = {'/calls/GT': gts}
        het = calc_obs_het_by_sample(varis, chunk_size=None)
        assert het.shape[0] == 0

        snps = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        calc_obs_het_by_sample(snps, min_call_dp=3)
        calc_obs_het_by_sample(snps, min_call_dp=3, max_call_dp=20)
        het_0 = calc_obs_het_by_sample(snps)
        het = calc_obs_het_by_sample(snps, chunk_size=None)
        assert numpy.allclose(het_0, het)

    def test_calc_depth_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())

        distrib, _ = calc_field_distribs_per_sample(hdf5, field='/calls/DP',
                                                    n_bins=560)

        assert distrib.shape == (153, 560)

        distrib2, _ = calc_field_distribs_per_sample(snps, field='/calls/DP',
                                                     n_bins=560,
                                                     chunk_size=None)
        assert numpy.all(distrib == distrib2)

        distrib3, _ = calc_field_distribs_per_sample(snps, field='/calls/DP',
                                                     n_bins=560,
                                                     chunk_size=50)
        assert numpy.all(distrib3 == distrib2)

        distrib_het, _ = calc_field_distribs_per_sample(snps,
                                                        field='/calls/DP',
                                                        n_bins=560,
                                                        chunk_size=50,
                                                        mask_field='/calls/GT',
                                                        mask_func=call_is_het)
        assert numpy.all(distrib3 == distrib2)
        distrib_hom, _ = calc_field_distribs_per_sample(snps,
                                                        field='/calls/DP',
                                                        n_bins=560,
                                                        chunk_size=50,
                                                        mask_field='/calls/GT',
                                                        mask_func=call_is_hom)
        assert numpy.all(distrib3 == numpy.add(distrib_het, distrib_hom))

        vars_ = VariationsArrays()
        vars_['/calls/DP'] = numpy.array([[10, 5, 15],
                                          [0, 15, 10]])
        vars_['/calls/GT'] = numpy.array([[[0, 0], [0, 1], [1, 1]],
                                          [[0, 0], [0, 1], [1, 1]]])
        vars_.samples = list(range(3))
        distrib, _ = calc_field_distribs_per_sample(vars_, field='/calls/DP',
                                                    n_bins=16)
        expec = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        assert numpy.all(expec == distrib)
        assert numpy.all(calc_depth(vars_) == [10, 5, 15, 0, 15, 10])

        distrib, _ = calc_field_distribs_per_sample(vars_, field='/calls/DP',
                                                    n_bins=16,
                                                    mask_field='/calls/GT',
                                                    mask_func=call_is_het)
        expec = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert numpy.all(expec == distrib)
        assert numpy.all(calc_depth(vars_) == [10, 5, 15, 0, 15, 10])

        distrib, _ = calc_field_distribs_per_sample(vars_, field='/calls/DP',
                                                    n_bins=16,
                                                    mask_field='/calls/GT',
                                                    mask_func=call_is_hom)
        expec = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        assert numpy.all(expec == distrib)
        assert numpy.all(calc_depth(vars_) == [10, 5, 15, 0, 15, 10])

    def test_calc_dp_means(self):
        snps = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        means = calc_depth_mean_by_sample(snps)

        means2 = calc_depth_mean_by_sample(snps, chunk_size=None)
        assert means.shape[0] == 153
        assert numpy.allclose(means, means2)

    def test_stasts_per_sample(self):
        gts = numpy.array([[[0, 0], [0, 1], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]],
                           [[1, 1], [0, 0], [0, 0], [-1, -1]]])

        dps = numpy.array([[4, 5, 1, 0],
                           [4, 5, 1, 0],
                           [3, 3, 1, 0]])
        varis = {'/calls/GT': gts, DP_FIELD: dps}
        res = calc_stats_by_sample(varis, chunk_size=None, dp_n_bins=5)

        assert numpy.allclose(res['homozygosity'][:3], [1, 2 / 3, 1])
        assert numpy.isnan(res['homozygosity'][3])
        assert numpy.allclose(res['obs_het'][:3], [0, 1 / 3, 0])
        assert numpy.isnan(res['obs_het'][3])
        assert numpy.allclose(res['called_gt_rate'], [1, 1, 1 / 3, 0])

        dp_hists = res['dp_hists']
        assert numpy.allclose(dp_hists['bin_edges'], [0, 1, 2, 3, 4, 5])
        cnts = [[0, 0, 0, 3],
                [0, 0, 3, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 0, 0]]
        assert numpy.allclose(dp_hists['dp_counts'], cnts)
        cnts = [[0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 0, 0]]
        assert numpy.allclose(dp_hists['dp_no_missing_counts'], cnts)
        cnts = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0]]
        assert numpy.allclose(dp_hists['dp_het_counts'], cnts)
        cnts = [[0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 1, 0, 0]]
        assert numpy.allclose(dp_hists['dp_hom_counts'], cnts)

        varis = {'/calls/GT': gts}
        res = calc_stats_by_sample(varis, chunk_size=None, dp_n_bins=5)
        assert 'dp_hists' not in res
        expected_keys = ['homozygosity', 'obs_het', 'hom_ref_rate',
                         'called_gt_rate', 'samples']
        assert not set(res.keys()).difference(expected_keys)

        varis = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        res1 = calc_stats_by_sample(varis, chunk_size=None)
        res2 = calc_stats_by_sample(varis, chunk_size=100)
        for key in ['homozygosity', 'obs_het', 'called_gt_rate']:
            assert numpy.allclose(res1[key], res2[key])

        for key in ['bin_edges', 'dp_counts', 'dp_no_missing_counts',
                    'dp_het_counts', 'dp_hom_counts']:
            assert numpy.allclose(res1['dp_hists'][key], res2['dp_hists'][key])

        fhand = StringIO()
        write_stats_by_sample(res1, fhand)
        lines = fhand.getvalue().splitlines()
        assert 'sample\tcall_rate\theterozygosity\tmean_dp' in lines[0]

    def test_hist_for_cols(self):
        data = [[1, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 2, 2]]
        data = numpy.array(data).T
        counts, edges = histograms_for_columns(data, n_bins=2)
        expected_cnts = [[4, 2, 0],
                         [0, 2, 4]]
        expected_edges = [1, 1.5, 2]
        assert numpy.allclose(expected_cnts, counts)
        assert numpy.allclose(expected_edges, edges)


class TajimaDTest(unittest.TestCase):

    def test_calc_1_div_i(self):
        num_seqs_with_data = numpy.array([1, 2, 3, 4, 5, 0])
        res = _calc_a1(num_seqs_with_data)
        a1_per_site, a2_per_site = res
        expected = [0, 1, 1.5, 1.83333, 2.08333, numpy.nan]
        assert numpy.allclose(a1_per_site, expected, equal_nan=True)
        expected = [0, 1, 1.25, 1.36111111, 1.42361111, numpy.nan]
        assert numpy.allclose(a2_per_site, expected, equal_nan=True)

    def print_phy(self, gts):

        numbers_to_letter = {0: 'A', 1: 'T', -1: 'N'}

        seqs = numpy.array(gts)[:, :, 0].T
        fhand = open('/home/jose/soft/variscan-2.0.3/VSexamples/examples/jose.phy', 'wt')
        if False:
            import sys
            fhand = sys.stdout
        fhand.write('{} {}\n'.format(*seqs.shape))
        for seq_idx in range(seqs.shape[0]):
            seq_number_haplotype = seqs[seq_idx, :]
            seq_letters = ''.join([numbers_to_letter[gt] for gt in seq_number_haplotype])
            fhand.write('seq_{}    {}\n'.format(seq_idx, seq_letters))
        fhand.close()

    def test_calc_tajima(self):

        snp_gt = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]]
        gts = [snp_gt] * 100
        varis = {GT_FIELD: numpy.array(gts)}

        res = calc_tajima_d_and_pi(varis, min_num_genotypes=2, ploidy=2)
        assert math.isclose(res['tajima_d'], -1.266621394662836)
        assert math.isclose(res['pi'], 0.4)
        assert math.isclose(res['theta'], 0.48)

        snp_gt = [[0, 0], [0, 0], [0, 0], [-1, -1], [1, 1]]
        gts = [snp_gt] * 100
        varis = {GT_FIELD: numpy.array(gts)}

        res = calc_tajima_d_and_pi(varis, min_num_genotypes=2, ploidy=2)
        assert math.isclose(res['tajima_d'], -0.871839964837347)
        assert math.isclose(res['pi'], 0.5)
        assert math.isclose(res['theta'], 0.5454545454545454)

        snp_gt = [[0, 0], [1, 1], [0, 0], [-1, -1], [1, 1]]
        gts = [snp_gt] * 100
        varis = {GT_FIELD: numpy.array(gts)}
        res = calc_tajima_d_and_pi(varis, min_num_genotypes=2, ploidy=2)
        # print(res)
        # self.print_phy(gts)
        assert math.isclose(res['tajima_d'], 2.3249065728995997)
        assert math.isclose(res['pi'], 0.6666666666666669)
        assert math.isclose(res['theta'], 0.5454545454545454)

        snp_gt = [[0, 0], [1, 1], [0, 0], [-1, -1], [1, 1]]
        gts = [snp_gt] * 100
        varis = {GT_FIELD: numpy.array(gts)}
        try:
            res = calc_tajima_d_and_pi(varis, min_num_genotypes=10, ploidy=2)
            self.fail('ValueError expected')
        except ValueError:
            pass

        snp_gt = [[0], [0], [0], [-1], [1]]
        gts = [snp_gt] * 100
        varis = {GT_FIELD: numpy.array(gts)}

        res = calc_tajima_d_and_pi(varis, min_num_genotypes=2, ploidy=2)
        assert math.isclose(res['tajima_d'], -0.871839964837347)
        assert math.isclose(res['pi'], 0.5)
        assert math.isclose(res['theta'], 0.5454545454545454)


if __name__ == "__main__":
    # import sys; sys.argv = ['', 'TajimaDTest']
    unittest.main()
