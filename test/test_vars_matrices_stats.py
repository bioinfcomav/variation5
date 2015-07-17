# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

import numpy

from test_utils import TEST_DATA_DIR
from variation.vars_matrices import VariationsH5, VariationsArrays
from variation.vars_matrices.stats import (calc_mafs,
                                           missing_gt_rate,
                                           called_gt_counts)
from variation.iterutils import first

class varMatricesStatsTest(unittest.TestCase):
    def test_calc_mafs_varh5(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        mafs = calc_mafs(chunk)
        assert numpy.all(mafs >= 0.5)
        assert numpy.all(mafs <= 1)
        assert mafs.shape == (200,)

    def test_calc_missing_gt_rates_varh5(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        rates = missing_gt_rate(chunk)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

    def test_calc_called_gt_counts_varh5(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        counts = called_gt_counts(chunk)
        assert counts.shape == (200,)
        assert numpy.max(counts) == 306

    def test_calc_mafs_varArray(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps = first(hdf5.iterate_chunks())
        mafs = calc_mafs(snps)
        assert numpy.all(mafs >= 0.5)
        assert numpy.all(mafs <= 1)
        assert mafs.shape == (200,)

    def test_calc_missing_gt_rates_varArray(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps = first(hdf5.iterate_chunks())
        rates = missing_gt_rate(snps)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

    def test_calc_called_gt_counts_varArray(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps = first(hdf5.iterate_chunks())
        counts = called_gt_counts(snps)
        assert counts.shape == (200,)
        assert numpy.max(counts) == 306

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'VcfH5ChunkStatsTest.test_calc_mafs']
    unittest.main()