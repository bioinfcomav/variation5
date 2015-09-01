# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import inspect
from os.path import dirname, abspath, join
from tempfile import NamedTemporaryFile

import numpy

from variation.vars_matrices import VariationsH5, VariationsArrays
from variation.vars_matrices.stats import (calc_mafs,
                                           missing_gt_rate,
                                           called_gt_counts,
                                           _remove_nans,
                                           calc_obs_het)
from variation.matrix.methods import calc_min_max

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


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

    def test_calc_obs_het(self):
        mat = numpy.array([[[0, 0], [1, -1], [0, 1], [-1, -1]]])
        is_het = calc_obs_het(mat)
        assert is_het == 0.25

    def test_calc_min_max(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        min_array, max_array = calc_min_max(snps['/calls/GT'])
        min_h5, max_h5 = calc_min_max(hdf5['/calls/GT'])
        assert min_array == min_h5
        assert max_array == max_h5

if __name__ == "__main__":
    import sys;sys.argv = ['', 'VarMatricesStatsTest.test_calc_min_max']
    unittest.main()
