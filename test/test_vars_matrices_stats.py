# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join
from tempfile import NamedTemporaryFile
import os
import numpy

from test_utils import TEST_DATA_DIR
from variation.vars_matrices import VariationsH5, VariationsArrays
from variation.vars_matrices.stats import (calc_mafs,
                                           missing_gt_rate,
                                           called_gt_counts,
                                           plot_hist_mafs,
                                           plot_hist_missing_rate,
                                           _remove_nans)
from variation.iterutils import first
from variation.matrix.stats import counts_by_row

class VarMatricesStatsTest(unittest.TestCase):
    def test_calc_mafs(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        mafs = calc_mafs(snps)
        mafs2 = calc_mafs(hdf5)
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
        rates = missing_gt_rate(snps)
        rates2 = missing_gt_rate(hdf5)
        assert rates.shape == (943,)
        assert numpy.allclose(rates, rates2)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

    def test_calc_called_gt_counts(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        counts = called_gt_counts(snps)
        counts2 = called_gt_counts(hdf5)
        assert counts.shape == (943,)
        assert numpy.all(counts == counts2)

    def test_plot_histogram_mafs(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result1 = plot_hist_mafs(hdf5, no_interactive_win=True)
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        result2 = plot_hist_mafs(snps, no_interactive_win=True)
        assert numpy.all(numpy.allclose(result1[1], result2[1]))

        fhand = NamedTemporaryFile(suffix='.png')
        plot_hist_mafs(snps, fhand=fhand, no_interactive_win=True)

    def test_plot_histogram_missing(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        result1 = plot_hist_missing_rate(hdf5, no_interactive_win=True)
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        result2 = plot_hist_missing_rate(snps, no_interactive_win=True)
        assert numpy.all(numpy.allclose(result1[1], result2[1]))

        fhand = NamedTemporaryFile(suffix='.png')
        plot_hist_missing_rate(snps, fhand=fhand, no_interactive_win=True)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'VarMatricesStatsTest.test_calc_mafs']
    unittest.main()
