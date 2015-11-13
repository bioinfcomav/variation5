
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from tempfile import NamedTemporaryFile
from os.path import join

import numpy


from variation.plot import (plot_histogram, _calc_boxplot_stats, plot_boxplot,
                            qqplot, manhattan_plot, plot_barplot,
                            plot_pandas_barplot, plot_hexabinplot)
from variation.variations.vars_matrices import VariationsH5
from variation.variations.stats import (_MissingGTCalculator,
                                        _MafCalculator,
                                        calc_snp_density,
                                        _calc_stat,
                                        calc_snv_density_distribution,
                                        calc_quality_by_depth_distrib,
                                        GenotypeStatsCalculator,
                                        calc_allele_obs_distrib_2D,
                                        HWECalcualtor,
                                        _remove_nans,
                                    calc_gq_cumulative_distribution_per_sample)
from test.test_utils import TEST_DATA_DIR
import os


class PlotTest(unittest.TestCase):
    def test_histogram(self):
        numbers = numpy.random.normal(size=(10000,))
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(numbers, bins=40, fhand=fhand)
            fhand.flush()
            read_fhand = open(fhand.name, 'rb')
            assert b'\x89PNG\r\n' in read_fhand.readline()
            read_fhand.close()

    def test_calc_boxplot_stats(self):
        data = numpy.ones((1, 100))
        stats = _calc_boxplot_stats(data)[0]
        assert stats['med'] == 50
        assert stats['q1'] == 25
        assert stats['q3'] == 75
        assert stats['mean'] == 49.5

    def test_plot_boxplot(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist, _ = calc_quality_by_depth_distrib(hdf5, depths=range(1, 30))
        title = 'Depth dependent GQ distribution'
        mpl_params = {'set_xlabel': {'args': ['Depth'], 'kwargs': {}},
                      'set_ylabel': {'args': ['Genotype quality'],
                                     'kwargs': {}},
                      'set_title': {'args': [title],
                                    'kwargs': {}}}
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_boxplot(dist, fhand=fhand, mpl_params=mpl_params)

    def test_plot_histogram(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        mafs = _calc_stat(hdf5, _MafCalculator())
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(mafs, range_=(0, 1), color='c', label='Mafs',
                           fhand=fhand)

        rates = _calc_stat(hdf5, _MissingGTCalculator())
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(rates, color='c', label='Missing GT',
                           fhand=fhand)

        density_h5 = calc_snp_density(hdf5, 100000)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(density_h5, color='c', label='Density',
                           fhand=fhand)

    def test_plot_barplot(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist_gq, _ = calc_gq_cumulative_distribution_per_sample(hdf5,
                                                                by_chunk=True)
        height_ = numpy.sum(dist_gq, axis=0)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_barplot(numpy.arange(0, 161), height_, fhand=fhand)

        dist_snv_density = calc_snv_density_distribution(hdf5, 100000)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_barplot(numpy.arange(0, dist_snv_density.shape[0]),
                         dist_snv_density, fhand=fhand)

    def test_plot_pandas_barplot(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        gt_stats = _calc_stat(hdf5, GenotypeStatsCalculator(),
                              reduce_funct=numpy.add)
        gt_stats = gt_stats.transpose()
        with NamedTemporaryFile(suffix='.png') as fhand:
            fpath = fhand.name
            fhand.close()
            plot_pandas_barplot(gt_stats, ['ref_hom', 'het', 'alt_hom',
                                           'missing'], fpath, stacked=True)
            os.remove(fpath)

    def test_plot_hexbin(self):
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

        with NamedTemporaryFile(suffix='.png') as fhand:
            fpath = fhand.name
            fhand.close()
            plot_hexabinplot(allele_distrib_2D, ['x', 'y', 'z'], fpath)
            os.remove(fpath)

    def test_qqplot(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        hwe_test = _calc_stat(hdf5, HWECalcualtor(2, 2))
        hwe_chi2 = _remove_nans(hwe_test[:, 0])
        with NamedTemporaryFile(suffix='.png') as fhand:
            qqplot(hwe_chi2, distrib='chi2', distrib_params=(3,), fhand=fhand)

    def test_manhattan_plot(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.h5'),
                            mode='r')
        hwe_test = _calc_stat(hdf5, HWECalcualtor(2, 2))
        hwe_pvalues = hwe_test[:, 1]
        with NamedTemporaryFile(suffix='.png') as fhand:
            manhattan_plot(hdf5['/variations/chrom'], hdf5['/variations/pos'],
                           hwe_pvalues, fhand=fhand,
                           yfunc=lambda x: -numpy.log10(x),
                           figsize=(20, 10), ylim=0)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PlotTest.test_plot_pandas_barplot']
    unittest.main()
