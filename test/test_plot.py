
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
                            qqplot, manhattan_plot, plot_barplot,
                            plot_histogram, plot_hist2d,
                            plot_boxplot_from_distribs_series,
                            write_curlywhirly, _look_for_first_different,
                            _estimate_percentiles_from_distrib, plot_hists,
                            plot_stacked_histograms, plot_sample_dp_hits,
                            plot_sample_missing_het_stats)
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
            plot_histogram(distrib, bins, fhand=fhand, bin_labels=list(x))

        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(distrib, bins, fhand=fhand, vlines=[3, 7])

    def test_plot_stacked_hists(self):
        vals1 = numpy.random.normal(loc=1, scale=0.5, size=500)
        vals2 = numpy.random.normal(loc=2, scale=0.5, size=500)
        counts, edges = numpy.histogram(vals1, range=(0, 3), bins=20)
        counts = {'vals1': counts}
        counts['vals2'], _ = numpy.histogram(vals2, range=(0, 3), bins=20)

        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_stacked_histograms(counts, edges, fhand=fhand)

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


class BoxplotsTests(unittest.TestCase):
    def test_quartile_estimation(self):
        counts = numpy.array([1, 3, 6, 10, 7, 2, 1])
        edges = numpy.array(range(len(counts) + 1))
        quartiles = [0, 0.25, 0.5, 0.75, 1]
        est = _estimate_percentiles_from_distrib(counts, edges,
                                                 percentiles=quartiles)
        exp = [0.5, 2.08333333, 3., 3.85714286, 6.5]
        assert numpy.allclose(est, exp)

        counts = numpy.array([counts,
                              [10, 5, 4, 3, 1, 1, 1],
                              [1, 1, 1, 2, 3, 5, 10]])
        # edges = numpy.arange(counts.shape[1])
        est = _estimate_percentiles_from_distrib(counts, edges,
                                                 percentiles=quartiles)
        exp = [exp,
               [0.5, 0.5, 1., 2.4375, 6.5],
               [0.5, 3.75, 5.2, 5.925, 6.5]]
        assert numpy.allclose(exp, est)

        counts = counts.T
        est = _estimate_percentiles_from_distrib(counts, edges,
                                                 percentiles=quartiles,
                                                 samples_in_rows=False)
        assert numpy.allclose(est, numpy.array(exp).T)

    def test_plot_boxplot(self):

        counts = numpy.array([[2, 3, 6, 10, 7, 2, 0],
                              [10, 5, 4, 3, 2, 2, 0],
                              [0, 2, 3, 2, 3, 5, 10]])
        counts = counts.T
        edges = numpy.arange(counts.shape[0] + 1)

        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_hists(counts, edges, fhand=fhand, log_hist_axes=True,
                       xlabels=['sample1', 'sample2', 'sample3'],
                       mpl_params={'set_ybound': {'kwargs': {'upper': 70}}})
            plot_hists(counts, edges, fhand=fhand, log_hist_axes=False,
                       xlabels=['sample1', 'sample2', 'sample3'],
                       plot_quartiles=True)

        # with two counts
        counts2 = numpy.array([[0, 2, 5, 10, 8, 3, 2],
                              [0, 9, 3, 2, 3, 5, 3],
                              [3, 4, 2, 2, 3, 9, 0]])
        counts2 = counts2.T
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_hists(counts, edges, counts2=counts2, fhand=fhand,
                       log_hist_axes=True,
                       xlabels=['sample1', 'sample2', 'sample3'],
                       mpl_params={'set_ybound': {'kwargs': {'upper': 70}}})
            plot_hists(counts, edges, fhand=fhand, counts2=counts2,
                       log_hist_axes=False,
                       xlabels=['sample1', 'sample2', 'sample3'],
                       plot_quartiles=True)

    def test_plot_sample_stats(self):
        call_gt = numpy.array([0.61505832, 0.55779427, 0.60233298, 0.65641569])
        obs_het = numpy.array([0.04827586, 0.02281369, 0.0334507, 0.04846527])
        bin_edges = [0, 90, 180, 270, 360, 450]
        dp_counts = numpy.array([[569, 514, 548, 591],
                                 [6, 9, 18, 15],
                                 [5, 2, 0, 10],
                                 [0, 1, 2, 1],
                                 [0, 0, 0, 2]])
        dp_no_missing = numpy.array([[569, 514, 548, 591],
                                     [6, 9, 18, 15],
                                     [5, 2, 0, 10],
                                     [0, 1, 2, 1],
                                     [0, 0, 0, 2]])
        dp_het_counts = numpy.array([[26, 11, 18, 30],
                                     [0, 0, 0, 0],
                                     [2, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 0]])
        dp_hom_counts = numpy.array([[543, 503, 530, 561],
                                     [6, 9, 18, 15],
                                     [3, 1, 0, 10],
                                     [0, 1, 1, 1],
                                     [0, 0, 0, 2]])

        sample_stats = {'called_gt_rate': call_gt,
                        'obs_het': obs_het,
                        'dp_hists': {'bin_edges': numpy.array(bin_edges),
                                     'dp_counts': numpy.array(dp_counts),
                                     'dp_no_missing_counts': dp_no_missing,
                                     'dp_het_counts': dp_het_counts,
                                     'dp_hom_counts': dp_hom_counts}}

        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_sample_missing_het_stats(sample_stats, fhand=fhand,
                                          samples=['s1', 's2', 's3', 's4'])
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_sample_dp_hits(sample_stats['dp_hists'], fhand=fhand,
                                samples=['s1', 's2', 's3', 's4'],
                                log_axis_for_hists=True)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'BoxplotsTests.test_plot_sample_stats']
    unittest.main()
