
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import os
import unittest
from tempfile import NamedTemporaryFile
from io import StringIO

import numpy

from variation.plot import (_calc_boxplot_stats, plot_boxplot_from_distribs,
                            qqplot, manhattan_plot, plot_barplot, plot_distrib,
                            plot_hist2d, plot_boxplot_from_distribs_series,
                            write_curlywhirly, _look_for_first_different)
from variation.variations.plot import pairwise_ld
from variation.variations import VariationsH5
from test.test_utils import TEST_DATA_DIR


class PlotTest(unittest.TestCase):

    def test_calc_boxplot_stats(self):
        data = numpy.ones((1, 100))
        stats = _calc_boxplot_stats(data)[0]
        assert stats['med'] == 50
        assert stats['q1'] == 25
        assert stats['q3'] == 75
        assert stats['mean'] == 49.5

    def test_plot_boxplot(self):
        distribs = numpy.array([[0, 0, 0, 0, 0, 1, 3, 5, 3, 1],
                                [0, 0, 0, 0, 1, 3, 5, 3, 1, 0],
                                [0, 1, 3, 5, 3, 1, 0, 0, 0, 0],
                                [1, 3, 5, 3, 1, 0, 0, 0, 0, 0]])
        mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                      'set_ylabel': {'args': ['Depth'],
                                     'kwargs': {}}}
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_boxplot_from_distribs(distribs, fhand=fhand,
                                       mpl_params=mpl_params)

    def test_plot_boxplot_series(self):
        distribs = numpy.array([[0, 0, 0, 0, 0, 1, 3, 5, 3, 1],
                                [0, 0, 0, 0, 1, 3, 5, 3, 1, 0],
                                [0, 1, 3, 5, 3, 1, 0, 0, 0, 0],
                                [1, 3, 5, 3, 1, 0, 0, 0, 0, 0]])
        mpl_params = {'set_xlabel': {'args': ['Samples'], 'kwargs': {}},
                      'set_ylabel': {'args': ['Depth'],
                                     'kwargs': {}}}
        distribs_series = [distribs[[0, 2], :], distribs[[1, 3], :]]
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_boxplot_from_distribs_series(distribs_series, fhand=fhand,
                                              mpl_params=mpl_params)

    def test_plot_distrib(self):
        x = numpy.array(list(range(1, 11)))
        distrib, bins = numpy.histogram(x)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_distrib(distrib, bins, fhand=fhand)

        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_distrib(distrib, bins, fhand=fhand, vlines=[3, 7])

    def test_plot_barplot(self):
        matrix = numpy.array([[1, 2, 3, 4],
                              [2, 1, 4, 3],
                              [4, 3, 2, 1]])
        with NamedTemporaryFile(suffix='.png') as fhand:
            fpath = fhand.name
            fhand.close()
            plot_barplot(matrix, ['ref_hom', 'het', 'alt_hom', 'missing'],
                         fpath, stacked=True, figsize=(10, 10))
            os.remove(fpath)

    def test_qqplot(self):
        chi2_statistic = numpy.random.rand(100)
        with NamedTemporaryFile(suffix='.png') as fhand:
            qqplot(chi2_statistic, distrib='chi2', distrib_params=(3,),
                   fhand=fhand)

    def test_manhattan_plot(self):

        assert _look_for_first_different([1, 1, 3], 0) == 2
        assert _look_for_first_different([1, 1, 3], 1) == 2
        assert _look_for_first_different([1, 1, 3], 2) == 3
        assert _look_for_first_different([1, 1, 1], 0) == 3

        chrom = numpy.array([b'chr1'] * 3 + [b'chr2'] * 3 + [b'chr3'] * 3)
        pos = numpy.array([1, 2, 3, 2, 5, 10, 1, 2, 3])
        statistic = numpy.array([2, 3, 2, 5, 3, 1, 3, 4, 2])
        with NamedTemporaryFile(suffix='.png') as fhand:
            manhattan_plot(chrom, pos, statistic, fhand=fhand,
                           figsize=(10, 10))

        with NamedTemporaryFile(suffix='.png') as fhand:
            manhattan_plot(chrom, pos, statistic, fhand=fhand,
                           split_by_chrom=True)

    def test_plot_hist2d(self):
        x = numpy.random.normal(loc=1, size=1000)
        y = numpy.random.normal(size=1000)
        distrib, xbins, ybins = numpy.histogram2d(x, y)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_hist2d(distrib, xbins, ybins, fhand=fhand,
                        colorbar_label='Counts', hist1d=False)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_hist2d(distrib, xbins, ybins, fhand=fhand,
                        colorbar_label='Counts')
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_hist2d(distrib, xbins, ybins, fhand=fhand,
                        log_normed=True,
                        mpl_params={'set_xlabel': 'x',
                                    'set_ylabel': 'y'})

    def test_curlywhirly(self):
        coords = numpy.random.normal(size=(3, 4))
        labels = ['acc1', 'acc2', 'acc3']
        fhand = StringIO()
        write_curlywhirly(fhand, labels, coords)
        expected = 'categories:all\tlabel\tdim_0\t'
        assert fhand.getvalue().splitlines()[0].startswith(expected)
        classes = {'class': ['class1', 'class2', 'class2']}
        fhand = StringIO()
        write_curlywhirly(fhand, labels, coords, classifications=classes)
        assert fhand.getvalue().splitlines()[1].startswith('class1\tacc1\t')

    def test_pairwise_ld(self):
        hdf5 = VariationsH5(os.path.join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        with NamedTemporaryFile(suffix='.png') as fhand:
            pairwise_ld(hdf5, fhand=fhand)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PlotTest.test_plot_hist2d']
    unittest.main()
