# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

import numpy

from variation.variations.vars_matrices import (VariationsH5,
                                                select_dset_from_chunks)
from variation.matrix.methods import extend_matrix
from variation.matrix.stats import (row_value_counter_fact,
                                    counts_by_row)
from variation.iterutils import first
from test.test_utils import TEST_DATA_DIR


class RowValueCounterTest(unittest.TestCase):
    def test_count_value_per_row(self):
        mat = numpy.array([[0, 0], [1, -1], [2, -1], [-1, -1]])
        missing_counter = row_value_counter_fact(value=-1)
        assert numpy.all(missing_counter(mat) == [0, 1, 1, 2])

        missing_counter = row_value_counter_fact(value=-1, ratio=True)
        assert numpy.allclose(missing_counter(mat), [0., 0.5, 0.5, 1.])

        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        chunks = list(hdf5.iterate_chunks())
        gt_chunk = first(select_dset_from_chunks(chunks, '/calls/GT'))

        homo_counter = row_value_counter_fact(value=2)
        assert numpy.all(homo_counter(gt_chunk) == [0, 0, 4, 0, 1])

        missing_counter = row_value_counter_fact(value=2, ratio=True)
        expected = [0., 0, 0.66666, 0., 0.166666]
        assert numpy.allclose(missing_counter(gt_chunk), expected)
        hdf5.close()

    def test_count_alleles(self):

        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        genotypes = chunk['/calls/GT']
        expected = [[3, 3, 0],
                    [5, 1, 0],
                    [0, 2, 4],
                    [6, 0, 0],
                    [2, 3, 1]]
        counts = counts_by_row(genotypes, missing_value=-1)
        assert numpy.all(expected == counts)
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunks = hdf5.iterate_chunks(kept_fields=['/calls/GT'])
        chunks = (chunk['/calls/GT'] for chunk in chunks)
        matrix = first(chunks)
        for _ in range(20):
            extend_matrix(matrix, chunks)

        counts = counts_by_row(matrix, missing_value=-1)

        gts = [[[-1, -1], [-1, -1], [-1, -1], [0, 0], [0, 0], [0, 0]]]
        gts = numpy.array(gts)
        counts = counts_by_row(gts, missing_value=-1)
        assert numpy.all(counts == [[6]])

        gts = [[[0, 0], [0, 0], [0, 0]],
               [[0, 0], [0, 0], [0, 0]]]
        gts = numpy.array(gts)
        counts = counts_by_row(gts, missing_value=-1)
        assert numpy.all(counts == [[6, 6]])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'RowValueCounterTest.test_count_alleles']
    unittest.main()
