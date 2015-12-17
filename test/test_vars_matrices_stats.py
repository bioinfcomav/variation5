# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join
from functools import partial

import numpy

from variation.variations.vars_matrices import VariationsH5, VariationsArrays
from variation.variations.stats import (calc_maf, histogram,
                                        histogram_for_chunks,
                                        _calc_maf_depth,
                                        calc_missing_gt, calc_obs_het,
                                        calc_obs_het_by_sample,
                                        calc_gt_type_stats, calc_called_gt,
                                        calc_snp_density, calc_allele_freq,
                                        calc_inbreeding_coef,
                                        calc_hwe_chi2_test,
                                        hist2d_allele_observations, GT_FIELD,
                                        call_is_het, calc_depth,
                                        hist2d_gq_allele_observations,
                                        calc_called_gts_distrib_per_depth,
                                        calc_field_distribs_per_sample,
                                        calc_maf_depth_distribs_per_sample,
                                        PositionalStatsCalculator, call_is_hom,
                                        calc_cum_distrib)
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
        
    def test_maf(self):
        gts = numpy.array([[[0], [0], [0], [0]], [[0], [0], [1], [1]],
                           [[0], [0], [0], [1]], [[-1], [-1], [-1], [-1]]])
        varis = VariationsArrays() 
        varis['/calls/GT'] = gts
        mafs = calc_maf(varis, min_num_genotypes=1)
        assert numpy.allclose(mafs, numpy.array([1., 0.5, 0.75, numpy.NaN]),
                              equal_nan=True)

        varis = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        mafs = calc_maf(varis)
        assert mafs.shape == (943,)

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
        dist_expected = [53, 75, 74, 70, 69, 129, 73, 74, 49, 277]
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
        dist_expected = [53, 75, 74, 70, 69, 129, 73, 74, 49, 277]
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

    def test_calc_maf_depth_distribs_per_sample(self):
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

    def test_calc_missing_gt_rates(self):
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
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = calc_obs_het(hdf5, min_num_genotypes=0)
        het_array = calc_obs_het(snps, min_num_genotypes=0)
        assert numpy.all(het_array == het_h5)
 
        gts = numpy.array([[[0, 0], [0, 1], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]]])

        varis = {'/calls/GT': gts}
        het = calc_obs_het(varis, min_num_genotypes=0)
        assert numpy.allclose(het, [0.5, 0])

        het = calc_obs_het(varis, min_num_genotypes=10)
        assert numpy.allclose(het, [numpy.NaN, numpy.NaN], equal_nan=True)

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
        density_h5 = calc_snp_density(hdf5, 1000)
        density_array = calc_snp_density(snps, 1000)
        assert numpy.all(density_array == density_h5)
        var = {'/variations/chrom': numpy.array(['sh', 'sh', 'sh', 'sh', 'sh',
                                                 'sh', 'sh', 'sh', 'sh', 'sh',
                                                 'sh', 'sh', 'sh', 'sh']),
               '/variations/pos': numpy.array([1, 2, 3, 4, 5, 6, 7, 25, 34, 44,
                                              80, 200, 300, 302])}
        dens_var = calc_snp_density(var, 10)
        expected = numpy.array([7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 1, 1, 2, 2])
        assert numpy.all(dens_var == expected)

    def test_calc_allele_freq(self):
        gts = numpy.array([[[0, 0], [1, 1], [0, -1], [-1, -1]],
                           [[0, -1], [0, 0], [0, -1], [-1, -1]],
                           [[0, 1], [0, 2], [0, 0], [-1, -1]]])
        varis = {'/calls/GT': gts, '/variations/alt': numpy.zeros((3, 2))}
        allele_freq = calc_allele_freq(varis, min_num_genotypes=0)
        expected = numpy.array([[0.6, 0.4, 0], [1, 0, 0],
                                [4 / 6, 1 / 6, 1 / 6]])
        assert numpy.allclose(allele_freq, expected)

        varis = {'/calls/GT': gts, '/variations/alt': numpy.zeros((3, 3))}
        allele_freq = calc_allele_freq(varis, min_num_genotypes=0)
        expected = numpy.array([[0.6, 0.4, 0, 0], [1, 0, 0, 0],
                                [4 / 6, 1 / 6, 1 / 6, 0]])
        assert numpy.allclose(allele_freq, expected)

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

    def test_calculate_hwe(self):
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
        
        distrib_het, _ = calc_field_distribs_per_sample(snps, field='/calls/DP',
                                                        n_bins=560,
                                                        chunk_size=50,
                                                        mask_field='/calls/GT',
                                                        mask_func=call_is_het)
        assert numpy.all(distrib3 == distrib2)
        distrib_hom, _ = calc_field_distribs_per_sample(snps, field='/calls/DP',
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
        print(pos_stats.chrom, pos_stats.pos, pos_stats.stat)
        line1 = 'track type=bedGraph name="track1" description="description"'
        bg_lines = [line1, 'chr1 10 35 {}'.format(6 / 25),
                    'chr1 35 60 {}'.format(9 / 25)]
        for line, exp in zip(pos_stats.to_bedGraph(), bg_lines[1:]):
            assert line.strip() == exp


if __name__ == "__main__":
    import sys;sys.argv = ['', 'StatsTest.test_to_positional_stats']
    unittest.main()
