
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import os
import unittest
from tempfile import NamedTemporaryFile


import numpy

from variation.plot import (_calc_boxplot_stats, plot_boxplot,
                            qqplot, manhattan_plot, plot_barplot, plot_distrib,
                            plot_hist2d)


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
            plot_boxplot(distribs, fhand=fhand, mpl_params=mpl_params)

    def test_plot_histogram(self):
        x = numpy.array(list(range(1, 11)))
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(x, fhand=fhand)

    def test_plot_distrib(self):
        x = numpy.array(list(range(1, 11)))
        distrib, bins = numpy.histogram(x)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_distrib(distrib, bins, fhand=fhand)

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
        chrom = numpy.array([b'chr1'] * 3 + [b'chr2'] * 3 + [b'chr3'] * 3)
        pos = numpy.array([1, 2, 3, 2, 5, 10, 1, 2, 3])
        statistic = numpy.array([2, 3, 2, 5, 3, 1, 3, 4, 2])
        with NamedTemporaryFile(suffix='.png') as fhand:
            manhattan_plot(chrom, pos, statistic, fhand=fhand,
                           figsize=(10, 10), ylim=0, marker='k')

    def test_plot_hist2d(self):
        x = numpy.random.rand(1000)
        y = numpy.random.rand(1000)
        distrib, xbins, ybins = numpy.histogram2d(x, y)
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_hist2d(distrib, xbins, ybins, fhand=fhand,
                        colorbar_label='Counts')


if __name__ == "__main__":
#     import sys;sys.argv = ['', 'PlotTest.test_manhattan_plot']
    unittest.main()
