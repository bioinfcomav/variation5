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
from variation.vcfh5 import VcfH5
from variation.vcfh5.chunk_stats import (calc_mafs,
                                         missing_gt_rate,
                                         called_gt_counts)
from variation.iterutils import first

class VcfH5ChunkStatsTest(unittest.TestCase):
    def test_calc_mafs(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        mafs = calc_mafs(chunk)
        assert numpy.all(mafs >= 0.5)
        assert numpy.all(mafs <= 1)
        assert mafs.shape == (200,)


    def test_calc_missing_gt_rates(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        rates = missing_gt_rate(chunk)
        assert numpy.min(rates) == 0
        assert numpy.all(rates <= 1)

    def test_calc_called_gt_counts(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        counts = called_gt_counts(chunk)
        assert counts.shape == (200,)
        assert numpy.max(counts) == 306

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'VcfH5ChunkStatsTest.test_calc_mafs']
    unittest.main()
