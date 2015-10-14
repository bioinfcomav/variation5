# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import inspect
from os.path import dirname, abspath, join

import numpy

from variation.variations import VariationsH5, VariationsArrays
from variation.variations.stats import (_remove_nans,
                                        _CalledGTCalculator,
                                        _MissingGTCalculator,
                                        _MafCalculator,
                                        _ObsHetCalculatorBySnps,
                                        _ObsHetCalculatorBySample,
                                        _calc_distribution,
                                        calc_depth_cumulative_distribution_per_sample,
                                        calc_snp_density,
                                        calc_gq_cumulative_distribution_per_sample,
                                        calc_hq_cumulative_distribution_per_sample,
                                        _calc_stat, _AlleleFreqCalculator,
                                        calc_expected_het,
                                        calc_inbreeding_coeficient, _is_het,
                                        _is_hom,
                                        calc_snv_density_distribution,
                                        GenotypeStatsCalculator,
                                        calc_called_gts_distrib_per_depth,
                                        calc_quality_by_depth,
                                        _MafDepthCalculator,
                                        calculate_maf_depth_distribution,
                                        calculate_maf_distribution,
                                        calc_allele_obs_distrib_2D,
                                        calc_allele_obs_gq_distrib_2D)

from variation.matrix.methods import calc_min_max
import pysam

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


class VarMatricesStatsTest(unittest.TestCase):
    def test_calc_mafs(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        mafs = _calc_stat(snps, _MafCalculator())
        mafs2 = _calc_stat(hdf5, _MafCalculator())
        mafs = _remove_nans(mafs)
        mafs2 = _remove_nans(mafs2)
        assert numpy.allclose(mafs, mafs2)
        assert numpy.all(mafs >= 0.5)
        assert numpy.all(mafs <= 1)
        assert mafs.shape == (936,)

    def test_calc_missing_gt_rates(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        rates = _calc_stat(snps, _MissingGTCalculator())
        rates2 = _calc_stat(hdf5, _MissingGTCalculator())
        assert rates.shape == (943,)
        assert numpy.allclose(rates, rates2)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

    def test_calc_called_gt_counts(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        counts = _calc_stat(snps, _CalledGTCalculator(rate=False))
        counts2 = _calc_stat(hdf5, _CalledGTCalculator(rate=False))
        assert counts.shape == (943,)
        assert numpy.all(counts == counts2)

    def test_calc_obs_het(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = _calc_stat(hdf5, _ObsHetCalculatorBySnps())
        het_array = _calc_stat(snps, _ObsHetCalculatorBySnps())
        assert numpy.all(het_array == het_h5)

    def test_calc_obs_het_sample(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = _calc_stat(hdf5, _ObsHetCalculatorBySample())
        het_array = _calc_stat(snps, _ObsHetCalculatorBySample())
        assert numpy.all(het_array == het_h5)

    def test_calc_min_max(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        min_array, max_array = calc_min_max(snps['/calls/GT'])
        min_h5, max_h5 = calc_min_max(hdf5['/calls/GT'])
        assert min_array == min_h5
        assert max_array == max_h5

    def test_calc_gt_frequency(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        gt_freq_h5 = _calc_stat(hdf5, _CalledGTCalculator(), by_chunk=True)
        gt_freq_array = _calc_stat(snps, _CalledGTCalculator(), by_chunk=True)
        assert numpy.all(gt_freq_h5 == gt_freq_array)

    def test_calc_snp_density(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        density_h5 = calc_snp_density(hdf5, 1000)
        density_array = calc_snp_density(snps, 1000)
        assert numpy.all(density_array == density_h5)

    def test_calc_depth_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        distribution_max_dp = _calc_distribution(hdf5, fields=['/calls/DP'],
                                                 max_value=30,
                                                 fillna=0)
        assert distribution_max_dp[-2, 1] == 129
        expected = numpy.full((153,), 943)

        result = calc_depth_cumulative_distribution_per_sample(hdf5,
                                                               by_chunk=True)
        distribution, cum_dist = result
        assert distribution.shape == (153, 559)
        assert numpy.all(cum_dist[:, -1] == expected)

    def test_calc_gq_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        result = calc_gq_cumulative_distribution_per_sample(hdf5, by_chunk=True)
        distribution, cum_dist = result
        assert distribution[0, 25] == 15
        assert cum_dist[-1, -1] == 537

    def test_calc_hq_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'format_def.h5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        result = calc_hq_cumulative_distribution_per_sample(hdf5, by_chunk=True)
        distribution, cum_dist = result
        assert distribution[0, -1] == 2
        assert cum_dist[0, -1] == 2

    def test_calc_allele_freq(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        result = _calc_stat(hdf5, _AlleleFreqCalculator(max_num_allele=4),
                            by_chunk=True)
        assert result[-2, 0] == 0.95

    def test_calc_expected_het(self):
        allele_freq = numpy.array([[0.5, 0.5, 0., 0.],
                                  [0.25, 0.25, 0.5, 0.]])
        exp_het = calc_expected_het(allele_freq)
        assert numpy.all(exp_het == numpy.array([0.5, 0.625]))

    def test_calc_inbreeding_coeficient(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        obs_het = _calc_stat(hdf5, _ObsHetCalculatorBySnps())
        allele_freq = _calc_stat(hdf5,
                                 _AlleleFreqCalculator(max_num_allele=4),
                                 by_chunk=True)
        exp_het = calc_expected_het(allele_freq)
        inbreeding_coef = calc_inbreeding_coeficient(obs_het, exp_het)

    def test_calc_depth_all_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = calc_depth_cumulative_distribution_per_sample(hdf5,
                                                       max_depth=30,
                                                       mask_function=_is_het,
                                                       mask_field='/calls/GT')
        dist_dp_het, cum_dp_het = result
        assert dist_dp_het[0, 2] == 4
        assert cum_dp_het[0, -1] == 25
        result2 = calc_depth_cumulative_distribution_per_sample(hdf5,
                                                        max_depth=30,
                                                        mask_function=_is_hom,
                                                        mask_field='/calls/GT')
        dist_dp_hom, cum_dp_hom = result2
        assert dist_dp_hom[0, 2] == 72
        assert cum_dp_hom[0, -1] == 470

    def test_calc_qual_all_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = calc_gq_cumulative_distribution_per_sample(hdf5,
                                                        mask_function=_is_het,
                                                        mask_field='/calls/GT')
        dist_gq_het, cum_gq_het = result
        assert dist_gq_het[0, 0] == 7
        assert cum_gq_het[0, -1] == 25
        result2 = calc_gq_cumulative_distribution_per_sample(hdf5,
                                                        mask_function=_is_hom,
                                                        mask_field='/calls/GT')
        dist_gq_hom, cum_gq_hom = result2
        assert dist_gq_hom[0, 2] == 0
        assert cum_gq_hom[0, -1] == 510

    def test_calc_snv_density_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist_snv_density = calc_snv_density_distribution(hdf5, 100000)
        assert dist_snv_density[3] == 6
        assert dist_snv_density.shape[0] == 96

    def test_calc_genotypes_stats(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = _calc_stat(hdf5, GenotypeStatsCalculator(),
                            reduce_funct=numpy.add)
        assert result.shape == (4, 153)
        assert numpy.all(numpy.sum(result, axis=0) == 943)

    def test_calc_called_gts_distribution_per_depth(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist, cum = calc_called_gts_distrib_per_depth(hdf5,
                                                      depths=range(30))
        assert dist[1, 1] == 0
        assert cum[-1, -1] == 943

    def test_calc_gq_by_depth(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist, cum = calc_quality_by_depth(hdf5, depths=range(3))
        assert dist[0, 0] == 0
        assert cum[-1, -1] == 11680

    def test_calc_maf_depth_distrib(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.h5'),
                            mode='r')
        maf_depths_dist = calculate_maf_depth_distribution(hdf5)
        assert maf_depths_dist.shape == (hdf5['/calls/GT'].shape[1], 101)

    def test_calc_maf_distrib(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        maf_dist = calculate_maf_distribution(hdf5)
        assert maf_dist.shape == (1, 101)

    def test_calculate_called_snps_distribution_per_depth(self):
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]]),
                      '/calls/DP': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10]])}
        dist, cum = calc_called_gts_distrib_per_depth(variations,
                                                      depths=[5, 10],
                                                      by_chunk=False)
        assert dist[0, -1] == 8
        assert cum[-1, -1] == 10

    def test_calculate_gq_by_depth(self):
        variations = {'/calls/DP': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10]]),
                      '/calls/GQ': numpy.array([[40], [30], [35], [30], [0],
                                               [40], [30], [35], [30], [0]])}
        dist, cum = calc_quality_by_depth(variations, depths=[5, 10],
                                          by_chunk=False)
        assert dist[0, 30] == 1
        assert dist[1, 0] == 2
        assert cum[0, -1] == 1
        assert cum[1, -1] == 3

    def test_calculate_maf(self):
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]])}
        calc_maf = _MafCalculator(min_num_genotypes=9)
        assert calc_maf(variations) == 0.7

    def test_calculate_maf_depth(self):
        variations = {'/calls/AO': numpy.array([[[0, 0], [5, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [20, 0], [25, 0], [20, 20],
                                                 [0, 0]]]),
                      '/calls/RO': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10]])}
        calc_maf = _MafDepthCalculator()
        assert numpy.all(calc_maf(variations) == numpy.array([1, 0.5, 1, 1, 1,
                                                              1, 1, 0.5, 1/3,
                                                              1]))

    def test_calc_allele_obs_distrib_2D(self):
        variations = {'/calls/AO': numpy.array([[[0, 0], [5, 0], [-1, -1],
                                                 [0, -1], [0, 0], [0, 10],
                                                 [20, 0], [25, 0], [20, 20],
                                                 [0, 0]]]),
                      '/calls/RO': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10]]),
                      '/calls/GQ': numpy.array([[40], [30], [35], [30], [0],
                                               [40], [30], [35], [30], [0]])}
        allele_distrib_2D = calc_allele_obs_distrib_2D(variations,
                                                       by_chunk=False)
        assert allele_distrib_2D[5, 5] == 1
        assert allele_distrib_2D[20, 20] == 1
        assert allele_distrib_2D[25, 25] == 1
        gq_distrib_2D = calc_allele_obs_gq_distrib_2D(variations,
                                                      by_chunk=False)
        assert gq_distrib_2D[5, 5] == 30
        assert gq_distrib_2D[20, 20] == 30
        assert gq_distrib_2D[25, 25] == 35
        assert gq_distrib_2D[10, 0] == 40/3


if __name__ == "__main__":
#     import sys;sys.argv = ['', 'VarMatricesStatsTest.test_calc_gq_by_depth']
    unittest.main()
