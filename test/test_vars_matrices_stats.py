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
                                        CalledGTCalculator,
                                        MissingGTCalculator,
                                        MafCalculator,
                                        ObsHetCalculator,
                                        ObsHetCalculatorBySample,
                                        _calc_depth_distribution_per_sample,
                                        calc_depth_cumulative_distribution_per_sample,
                                        calc_stat_by_chunk,
                                        calc_snp_density)

from variation.matrix.methods import calc_min_max

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


class VarMatricesStatsTest(unittest.TestCase):
    def test_calc_mafs(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        mafs = calc_stat_by_chunk(snps, MafCalculator())
        mafs2 = calc_stat_by_chunk(hdf5, MafCalculator())
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
        rates = calc_stat_by_chunk(snps, MissingGTCalculator())
        rates2 = calc_stat_by_chunk(hdf5, MissingGTCalculator())
        assert rates.shape == (943,)
        assert numpy.allclose(rates, rates2)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

    def test_calc_called_gt_counts(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        counts = calc_stat_by_chunk(snps, CalledGTCalculator(rate=False))
        counts2 = calc_stat_by_chunk(hdf5, CalledGTCalculator(rate=False))
        assert counts.shape == (943,)
        assert numpy.all(counts == counts2)

    def test_calc_obs_het(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = calc_stat_by_chunk(hdf5, ObsHetCalculator())
        het_array = calc_stat_by_chunk(snps, ObsHetCalculator())
        assert numpy.all(het_array == het_h5)

    def test_calc_obs_het_sample(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        het_h5 = calc_stat_by_chunk(hdf5, ObsHetCalculatorBySample())
        het_array = calc_stat_by_chunk(snps, ObsHetCalculatorBySample())
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
        gt_freq_h5 = calc_stat_by_chunk(hdf5, CalledGTCalculator())
        gt_freq_array = calc_stat_by_chunk(snps, CalledGTCalculator())
        assert numpy.all(gt_freq_h5 == gt_freq_array)

    def test_calc_snp_density(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        density_h5 = calc_snp_density(hdf5, 1000)
        density_array = calc_snp_density(snps, 1000)
        assert numpy.all(density_array == density_h5)

    def test_num_samples_higher_equal_dp(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        distribution_max_dp = _calc_depth_distribution_per_sample(hdf5,
                                                                  max_depth=30)
        assert distribution_max_dp[-2, 1] == 129
        expected = numpy.full((153,), 943)

        distribution, cum_dist = calc_depth_cumulative_distribution_per_sample(hdf5)
        assert distribution.shape == (153, 559)
        assert numpy.all(cum_dist[:, -1] == expected)

if __name__ == "__main__":
    import sys;sys.argv = ['', 'VarMatricesStatsTest.test_num_samples_higher_equal_dp']
    unittest.main()
