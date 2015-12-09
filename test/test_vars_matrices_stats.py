# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import sys
import inspect
from os.path import dirname, abspath, join

import numpy
from subprocess import check_output

from variation.variations import VariationsH5, VariationsArrays
from variation.matrix.methods import calc_min_max
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
                                        calc_inbreeding_coeficient, _is_het,
                                        _is_hom,
                                        calc_snv_density_distribution,
                                        GenotypeStatsCalculator,
                                        calc_called_gts_distrib_per_depth,
                                        calc_quality_by_depth_distrib,
                                        _MafDepthCalculator,
                                        calc_maf_depth_distrib,
                                        calculate_maf_distribution,
                                        calc_allele_obs_distrib_2D,
                                        calc_allele_obs_gq_distrib_2D,
                                        _is_hom_ref, _is_hom_alt,
                                        calc_inbreeding_coeficient_distrib,
                                        HWECalcualtor,
                                        PositionalStatsCalculator,
                                        _ExpectedHetCalculator)
from test.test_utils import BIN_DIR
from tempfile import TemporaryDirectory

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
        var = {'/variations/chrom': numpy.array(['sh', 'sh', 'sh', 'sh', 'sh',
                                                 'sh', 'sh', 'sh', 'sh', 'sh',
                                                 'sh', 'sh', 'sh', 'sh']),
               '/variations/pos': numpy.array([1, 2, 3, 4, 5, 6, 7, 25, 34, 44,
                                              80, 200, 300, 302])}
        dens_var = calc_snp_density(var, 10)
        expected = numpy.array([7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 1, 1, 2, 2])
        assert numpy.all(dens_var == expected)

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
        assert numpy.all(cum_dist[:, 0] == expected)

    def test_calc_gq_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        result = calc_gq_cumulative_distribution_per_sample(hdf5,
                                                            by_chunk=True,
                                                            max_value=161.0)
        distribution, cum_dist = result
        assert distribution[0, 25] == 15
        assert cum_dist[-1, 0] == 537

    def test_calc_hq_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'format_def.h5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        result = calc_hq_cumulative_distribution_per_sample(hdf5,
                                                            by_chunk=True)

        distribution, cum_dist = result

        print(distribution)
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
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]])}
        calc_expected_het = _ExpectedHetCalculator(max_num_allele=2)
        exp_het = calc_expected_het(variations)
        assert exp_het[0] - 0.42 < 0.001

    def test_calc_inbreeding_coeficient(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        obs_het = _calc_stat(hdf5, _ObsHetCalculatorBySnps())
        calc_expected_het = _ExpectedHetCalculator(max_num_allele=4)
        exp_het = calc_expected_het(hdf5)
        expected = 1 - (obs_het/exp_het)
        result = calc_inbreeding_coeficient(hdf5, max_num_allele=4,
                                            by_chunk=True)
        result = _remove_nans(result)
        expected = _remove_nans(expected)
        assert numpy.all(result == expected)

        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]])}
        result = calc_inbreeding_coeficient(variations, max_num_allele=4,
                                            by_chunk=False)
        assert result[0] - 1-(0.4/0.42) < 0.0000001

    def test_calc_inbreeding_coef_distrib(self):
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]])}
        result = calc_inbreeding_coeficient_distrib(variations, by_chunk=False)
        assert result[104] == 1
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = calc_inbreeding_coeficient_distrib(hdf5, by_chunk=False)
        assert(result[-1] == 330)

    def test_calc_depth_all_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = calc_depth_cumulative_distribution_per_sample(hdf5,
                                                       max_depth=30,
                                                       mask_function=_is_het,
                                                       mask_field='/calls/GT')
        dist_dp_het, cum_dp_het = result
        assert dist_dp_het[0, 2] == 4
        assert cum_dp_het[0, 0] == 25
        result2 = calc_depth_cumulative_distribution_per_sample(hdf5,
                                                        max_depth=30,
                                                        mask_function=_is_hom,
                                                        mask_field='/calls/GT')
        dist_dp_hom, cum_dp_hom = result2
        assert dist_dp_hom[0, 2] == 72
        assert cum_dp_hom[0, 0] == 470

    def test_calc_qual_all_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result = calc_gq_cumulative_distribution_per_sample(hdf5,
                                                        mask_function=_is_het,
                                                        mask_field='/calls/GT')
        dist_gq_het, cum_gq_het = result
        assert dist_gq_het[0, 0] == 7
        assert cum_gq_het[0, 0] == 25
        result2 = calc_gq_cumulative_distribution_per_sample(hdf5,
                                                        mask_function=_is_hom,
                                                        mask_field='/calls/GT')
        dist_gq_hom, cum_gq_hom = result2
        assert dist_gq_hom[0, 2] == 0
        assert cum_gq_hom[0, 0] == 510

    def test_calc_snv_density_distribution(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist_snv_density = calc_snv_density_distribution(hdf5, 100000)
        assert dist_snv_density[3] == 12
        assert dist_snv_density.shape[0] == 95

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
        assert cum[-1, 0] == 943

    def test_calc_gq_by_depth(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist, cum = calc_quality_by_depth_distrib(hdf5, depths=range(3))
        assert dist[0, 0] == 0
        assert cum[-1, 0] == 11680

    def test_calc_maf_depth_distrib(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.h5'),
                            mode='r')
        maf_depths_dist = calc_maf_depth_distrib(hdf5, by_chunk=True)
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
        assert cum[-1, 0] == 10

    def test_calculate_gq_by_depth(self):
        variations = {'/calls/DP': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10]]),
                      '/calls/GQ': numpy.array([[40], [30], [35], [30], [0],
                                               [40], [30], [35], [30], [0]])}
        dist, cum = calc_quality_by_depth_distrib(variations, depths=[5, 10],
                                                  by_chunk=False)
        assert dist[0, 30] == 1
        assert dist[1, 0] == 2
        assert cum[0, 0] == 1
        assert cum[1, 0] == 3

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
                                                 [0, 0], [20, 0]]]),
                      '/calls/RO': numpy.array([[10], [5], [15], [7], [10],
                                                [0], [0], [25], [20], [10],
                                                [0]])}
        calc_maf = _MafDepthCalculator()
        assert numpy.all(calc_maf(variations) == numpy.array([1, 0.5, 1, 1, 1,
                                                              1, 1, 0.5, 1/3,
                                                              1, 1]))

    def test_calculate_maf_depth_dist(self):
        variations = {'/calls/AO': numpy.array([[[0, 0]], [[5, 0]], [[-1, -1]],
                                                [[0, -1]], [[0, 0]], [[0, 10]],
                                                [[20, 0]], [[25, 0]],
                                                [[20, 20]], [[0, 0]],
                                                [[20, 0]], [[-1, -1]]]),
                      '/calls/RO': numpy.array([[10, 5, 15, 7, 10,
                                                0, 0, 25, 20, 10, 0, -1]])}
        result = calc_maf_depth_distrib(variations, by_chunk=False)
        expected = numpy.zeros((1, 101))
        expected[0, 0] = 1
        expected[0, -1] = 8
        expected[0, 50] = 2
        expected[0, 33] = 1
        assert numpy.all(result == expected)

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

    def test_is_gt(self):
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]])}
        is_hom = _is_hom(variations['/calls/GT'])
        assert numpy.all(is_hom == numpy.array([True, False, False, True,
                                                False, True, True, False,
                                                True, True]))
        is_hom_ref = _is_hom_ref(variations['/calls/GT'])
        assert numpy.all(is_hom_ref == numpy.array([True, False, False, True,
                                                    False, True, True, False,
                                                    False, True]))
        is_hom_alt = _is_hom_alt(variations['/calls/GT'])
        assert numpy.all(is_hom_alt == numpy.array([False, False, False, False,
                                                    False, False, False, False,
                                                    True, False]))

    def test_calculate_hwe(self):
        variations = {'/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]],
                                                [[0, 0], [0, 1], [0, 1],
                                                 [0, 0], [0, 1], [0, 0],
                                                 [0, 0], [0, 1], [1, 1],
                                                 [0, 0]]])}
        calc_hwe_chi2 = HWECalcualtor(2, ploidy=2)
        expected_result = numpy.array([[0.0226757369615, 0.988726162928],
                                       [0.0226757369615, 0.988726162928]])
        result = calc_hwe_chi2(variations)
        for res, exp in zip(result, expected_result):
            for x, y in zip(res, exp):
                assert abs(x - y) < 0.000000001

    def test_to_positional_stats(self):
        chrom = numpy.array(['chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4'])
        pos = numpy.array([10, 5, 20, 30, 40, 50])
        stat = numpy.array([1, 2, 3, 4, 5, numpy.nan])
        pos_stats = PositionalStatsCalculator(chrom, pos, stat)
        wiglines = ['track type=wiggle_0 name="track1" description="description"',
                    'variableStep chrom=chr1', '10 1.0',
                    'variableStep chrom=chr2', '5 2.0', '20 3.0',
                    'variableStep chrom=chr3', '30 4.0', '40 5.0']
        for line, exp in zip(pos_stats.to_wig(), wiglines[1:]):
            assert line.strip() == exp

        bg_lines = ['track type=bedGraph name="track1" description="description"',
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
        wiglines = ['track type=wiggle_0 name="track1" description="description"',
                    'fixedStep chrom=chr1 start=10 span=25 step=25',
                    str(6/25), str(9/25)]
        for line, exp in zip(pos_stats.to_wig(), wiglines[1:]):
            assert line.strip() == exp

        bg_lines = ['track type=bedGraph name="track1" description="description"',
                    'chr1 10 35 {}'.format(6/25), 'chr1 35 60 {}'.format(9/25)]
        for line, exp in zip(pos_stats.to_bedGraph(), bg_lines[1:]):
            assert line.strip() == exp

        # Pre-calculating the windows
        pos_stats = pos_stats.calc_window_stat()
        bg_lines = ['track type=bedGraph name="track1" description="description"',
                    'chr1 10 35 {}'.format(6/25), 'chr1 35 60 {}'.format(9/25)]
        for line, exp in zip(pos_stats.to_bedGraph(), bg_lines[1:]):
            assert line.strip() == exp

    def test_calc_hdf5_stats_bin(self):
        bin_ = join(BIN_DIR, 'calculate_h5_stats.py')
        with TemporaryDirectory() as tmpdir:
            cmd = [sys.executable, bin_, join(TEST_DATA_DIR, 'ril.hdf5'), '-o',
                   tmpdir]
            check_output(cmd)

if __name__ == "__main__":
    import sys;sys.argv = ['', 'VarMatricesStatsTest.test_calc_snv_density_distribution']
    unittest.main()
