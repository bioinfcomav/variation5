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
from variation.variations.allele_stats import (count_variant,
                                                  count_non_variant, is_variant,
                                                  is_non_variant, allele_number,
    is_singleton, count_doubleton, is_doubleton, count_singleton, allele_count,
    allele_frequency)

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))

class VarMatricesAlleleStatsTest(unittest.TestCase):
    def test_count_variants(self):
        #mat = numpy.array([[[0, 0], [1, -1]], [[0, 1], [-1, -1]], [[0, 0], [0, 0]]])
        #counts_mat == 2 & counts_non_mat == 1
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        counts_var_h5 = count_variant(hdf5)
        counts_var_array = count_variant(snps)
        assert counts_var_h5 == 943
        assert counts_var_h5 == counts_var_array

        counts_non_var_h5 = count_non_variant(hdf5)
        counts_non_var_array = count_non_variant(snps)
        assert counts_non_var_h5 == 0
        assert counts_non_var_array == counts_non_var_h5

        assert hdf5['/calls/GT'].shape[0] == (counts_var_h5 + counts_non_var_h5)

    def test_is_variants(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        is_var = is_variant(snps)
        is_non_var = is_non_variant(snps)
        assert numpy.all(numpy.logical_xor(is_var,is_non_var) == True)

    def test_count_singleton(self):
        #mat = numpy.array([[[0, 0], [1, -1]], [[0, 1], [-1, -1]], [[0, 0], [0, 0]]])
        #counts_mat == 2 & counts_non_mat == 1
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        counts_sinlge_h5 = count_singleton(hdf5, allele=0)
        counts_sinlge_array = count_singleton(snps, allele=0)
        assert counts_sinlge_h5 == counts_sinlge_array
        assert counts_sinlge_h5 == 11

    def test_is_singleton(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        is_single_array = is_singleton(snps, allele=0)
        is_single_h5 = is_singleton(hdf5, allele = 0)
        assert numpy.all(is_single_array == is_single_h5)

    def test_count_doubleton(self):
        #mat = numpy.array([[[0, 0], [1, -1]], [[0, 1], [-1, -1]], [[0, 0], [0, 0]]])
        #counts_mat == 2 & counts_non_mat == 1
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        counts_double_h5 = count_doubleton(hdf5, allele=1)
        counts_double_array = count_doubleton(snps, allele=1)
        assert counts_double_h5 == counts_double_array
        assert counts_double_h5 == 32

    def test_is_doubleton(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        is_double_array = is_doubleton(snps, allele=1)
        is_double_h5 = is_doubleton(hdf5, allele=1)
        assert numpy.all(is_double_array == is_double_h5)

    def test_allele_number(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        allele_h5 = allele_number(hdf5)
        allele_array = allele_number(snps)
        assert numpy.all(allele_array == allele_h5)
        assert numpy.max(allele_array) == 306

    def test_allele_count(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        allele_count_h5 = allele_count(hdf5, allele=0)
        allele_count_array = allele_count(snps, allele=0)
        assert numpy.all(allele_count_array == allele_count_h5)
        assert allele_count_h5[30] == 0

    def test_allele_frecuency(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
        allele_freq0 = allele_frequency(hdf5, allele=0)
        allele_freq1 = allele_frequency(snps, allele=1)
        allele_freq2 = allele_frequency(hdf5, allele=2)
        allele_freq3 = allele_frequency(hdf5, allele=3)
        allele_all_freqs = allele_freq0 + allele_freq1 + allele_freq2 + allele_freq3
        assert numpy.all(allele_all_freqs)== 1

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'VarMatricesAlleleStatsTest.test_count_doubleton']
    unittest.main()
