# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import shutil
from tempfile import NamedTemporaryFile
from posixpath import join

import numpy

from variation.matrix.methods import (append_matrix, extend_matrix,
                                      append_different_size)
from variation.variations.vars_matrices import VariationsH5
from test.test_utils import TEST_DATA_DIR


class ArrayTest(unittest.TestCase):

    def test_extend(self):
        mats = [numpy.array([[1, 1], [2, 2]]), numpy.array([[3, 3], [4, 4]]),
                numpy.array([[5, 5]])]
        expected = ([[0, 0],
                     [1, 1],
                     [2, 2],
                     [3, 3],
                     [4, 4],
                     [5, 5]])
        array = numpy.zeros([1, 2], dtype=numpy.int)
        extend_matrix(array, mats)
        assert numpy.all(array == expected)

    def test_append_matrix(self):
        in_fpath = join(TEST_DATA_DIR, '1000snps.hdf5')
        array = numpy.array([[1, 1, 1], [2, 2, 2]])
        expected = ([[1, 8, 5],
                     [3, 5, 3],
                     [6, 0, 4],
                     [7, 4, 2],
                     [4, 2, 3],
                     [1, 1, 1],
                     [2, 2, 2]])
        expected2 = [[1, 1, 1],
                     [2, 2, 2],
                     [1, 8, 5],
                     [3, 5, 3],
                     [6, 0, 4],
                     [7, 4, 2],
                     [4, 2, 3],
                     [1, 1, 1],
                     [2, 2, 2],
                     [1, 8, 5],
                     [3, 5, 3],
                     [6, 0, 4],
                     [7, 4, 2],
                     [4, 2, 3],
                     [1, 1, 1],
                     [2, 2, 2]]
        with NamedTemporaryFile(suffix='.h5') as fhand_out:
            shutil.copy(in_fpath, fhand_out.name)
            hdf5 = VariationsH5(fhand_out.name, mode='r+')
            dset = hdf5['/calls/DP']
            orig_array = dset.value
            append_matrix(dset, array)
            assert numpy.all(dset.value == expected)

            append_matrix(dset, dset)

            array2 = numpy.array([[1, 1, 1], [2, 2, 2]])
            append_matrix(array2, dset.value)
            assert numpy.all(expected2 == array2)

        append_matrix(orig_array, array)
        assert numpy.all(orig_array == expected)

    def test_append_different_size(self):
        matrix1 = numpy.array([[1, 1, 1], [2, 2, 2]])
        matrix2 = numpy.array([[1, 1, 1, 1]])
        matrix = append_different_size(matrix1, matrix2)
        assert numpy.all(matrix == numpy.array([[1, 1, 1, -1],
                                                [2, 2, 2, -1],
                                                [1, 1, 1, 1]]))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'ArrayTest.test_append_different_size']
    unittest.main()
