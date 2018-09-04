# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import shutil
from tempfile import NamedTemporaryFile
import tempfile
from posixpath import join

import numpy
import h5py

from variation.matrix.methods import (append_matrix, extend_matrix,
                                      append_different_size,
                                      iterate_matrix_chunks,
                                      calc_min_max, resize_array,
                                      concat_vector, concat_matrices,
                                      vstack, _set_matrix_by_chunks)
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

    def test_itereate_chunks(self):
        mat = numpy.array([[1, 2, 3], [4, 5, 6]])
        exp = [[[1, 2, 3]], [[4, 5, 6]]]
        res = list(iterate_matrix_chunks(mat, chunk_size=1))
        assert numpy.all(res[0] == exp[0])
        assert numpy.all(res[1] == exp[1])

        exp = [[2], [5]]
        res = list(iterate_matrix_chunks(mat, chunk_size=1, sample_idx=1))
        assert numpy.all(res[0] == exp[0])
        assert numpy.all(res[1] == exp[1])

    def test_min_max(self):
        mat = numpy.array([[1, 2, 3], [4, 5, 6]])
        assert calc_min_max(mat) == (1, 6)
        assert calc_min_max(mat, chunk_size=1) == (1, 6)
        assert calc_min_max(mat, chunk_size=None) == (1, 6)
        assert calc_min_max(mat, sample_idx=1) == (2, 5)
        assert calc_min_max(mat, sample_idx=1, chunk_size=None) == (2, 5)


class VStackTest(unittest.TestCase):

    def test_vector_concat(self):
        mat1 = numpy.array([0, 1])
        mat2 = numpy.array([2, 3])
        mat = concat_vector([mat1, mat2], -1)
        assert numpy.all(mat == numpy.array([0, 1, 2, 3]))

        mat = concat_matrices([mat1, mat2], -1)
        assert numpy.all(mat == numpy.array([0, 1, 2, 3]))

    def test_no_vectors(self):
        try:
            concat_vector([], -1)
            self.fail('ValueError expected')
        except ValueError:
            pass

    def test_2d_stacking(self):
        mat1 = numpy.array([[0, 1], [2, 3]])
        mat2 = numpy.array([[3, 4], [5, 6]])
        mat = vstack([mat1, mat2], -1)
        assert numpy.all(mat == [[0, 1], [2, 3], [3, 4], [5, 6]])

    def test_no_matrices(self):
        try:
            vstack([], -1)
            self.fail('ValueError expected')
        except ValueError:
            pass

    def test_2d_stacking_different_shapes(self):
        mat1 = numpy.array([[0, 1], [2, 3]])
        mat2 = numpy.array([[3], [5], [6]])
        mat = vstack([mat1, mat2], -1)
        assert numpy.all(mat == [[0, 1], [2, 3], [3, -1], [5, -1], [6, -1]])

        mat = vstack([mat2, mat1], -1)
        assert numpy.all(mat == [[3, -1], [5, -1], [6, -1], [0, 1], [2, 3]])

        mat1 = numpy.array([[[10, 11, 12, 13],
                             [14, 15, 16, 17],
                             [18, 19, 20, 21]],
                            [[22, 23, 24, 25],
                             [26, 27, 28, 29],
                             [30, 31, 32, 33]]])
        mat2 = numpy.array([[[40, 41, 42, 43],
                             [44, 45, 46, 47]]])
        mat3 = numpy.array([[[10, 11, 12, 13],
                             [14, 15, 16, 17],
                             [18, 19, 20, 21]],
                            [[22, 23, 24, 25],
                             [26, 27, 28, 29],
                             [30, 31, 32, 33]],
                            [[40, 41, 42, 43],
                             [44, 45, 46, 47],
                             [-1, -1, -1, -1]]])
        mat = vstack([mat1, mat2], -1)
        assert numpy.all(mat == mat3)

        mat = concat_matrices([mat1, mat2], -1)
        assert numpy.all(mat == mat3)

    def test_str_vector_concat(self):
        mat1 = numpy.array([b'hola'])
        mat2 = numpy.array([b'caracola'])
        mat = concat_vector([mat1, mat2], b'x')
        assert numpy.all(mat == numpy.array([b'hola', b'caracola']))


class VStackH5Test(unittest.TestCase):

    def create_dset(self, mat, fillvalue, maxshape=None, path='/dset'):
        h5_fhand = tempfile.NamedTemporaryFile(suffix='.h5')
        h5 = h5py.File(h5_fhand.name, "w")
        dset = h5.create_dataset(path, data=mat,
                                 maxshape=maxshape, fillvalue=fillvalue)
        h5_fhand.flush()
        return h5_fhand, dset

    def test_vector_concat(self):
        mat1 = numpy.array([0, 1])
        _, dset1 = self.create_dset(mat1, -1, maxshape=(None,))

        mat2 = numpy.array([2, 3])
        mat = concat_vector([dset1, mat2], -1)
        assert numpy.all(mat[:] == numpy.array([0, 1, 2, 3]))

        mat1 = numpy.array([0, 1])
        _, dset1 = self.create_dset(mat1, -1)

        mat2 = numpy.array([2, 3])
        mat = concat_vector([dset1, mat2], -1)
        assert numpy.all(mat[:] == numpy.array([0, 1, 2, 3]))

        mat1 = numpy.array([b'a', b'b'])
        _, dset1 = self.create_dset(mat1, b'x')

        mat2 = numpy.array([b'c', b'd'])
        mat = concat_vector([dset1, mat2], -1)
        assert numpy.all(mat[:] == numpy.array([b'a', b'b', b'c', b'd']))

    def test_str_vector_different_size(self):

        mat1 = numpy.array([b'a', b'b'])
        _, dset1 = self.create_dset(mat1, b'x')

        mat2 = numpy.array([b'cf', b'd'])
        mat = concat_vector([dset1, mat2], -1)
        assert numpy.all(mat[:] == numpy.array([b'a', b'b', b'cf', b'd']))

    def test_dset_replacement(self):
        mat1 = numpy.array([0, 1])
        _, dset1 = self.create_dset(mat1, -1,
                                           path='/hola')

        mat2 = numpy.array([2, 3])
        mat = concat_vector([dset1, mat2], -1)
        assert mat.name == None

        _, dset1 = self.create_dset(mat1, -1,
                                           path='/hola')
        mat = concat_vector([dset1, mat2], -1,
                            if_first_matrix_is_dataset_replace_it=True)
        assert mat.name == '/hola'
        assert mat is not dset1

        _, dset1 = self.create_dset(mat1, -1, maxshape=(None,),
                                           path='/hola')
        mat = concat_vector([dset1, mat2], -1,
                            if_first_matrix_is_dataset_replace_it=True)
        assert mat.name == '/hola'
        assert mat is dset1

    def test_2d_stacking(self):
        mat1 = numpy.array([[0, 1], [2, 3]])
        _, dset1 = self.create_dset(mat1, -1)
        mat2 = numpy.array([[3, 4], [5, 6]])
        mat = vstack([dset1, mat2], -1)
        assert numpy.all(mat[:] == [[0, 1], [2, 3], [3, 4], [5, 6]])

        _, dset1 = self.create_dset(mat1, -1, maxshape=(None, 2))
        mat = vstack([dset1, mat2], -1)
        assert numpy.all(mat[:] == [[0, 1], [2, 3], [3, 4], [5, 6]])

    def test_2d_stacking_different_shapes(self):
        mat1 = numpy.array([[0, 1], [2, 3]])
        _, dset1 = self.create_dset(mat1, -1)
        mat2 = numpy.array([[3], [5], [6]])
        mat = vstack([dset1, mat2], -1)
        assert numpy.all(mat[:] == [[0, 1], [2, 3], [3, -1], [5, -1], [6, -1]])

        h5_fhand2, dset2 = self.create_dset(mat2, -1)
        mat = vstack([dset2, mat1], -1)
        assert numpy.all(mat[:] == [[3, -1], [5, -1], [6, -1], [0, 1], [2, 3]])

        mat1 = numpy.array([[[10, 11, 12, 13],
                             [14, 15, 16, 17],
                             [18, 19, 20, 21]],
                            [[22, 23, 24, 25],
                             [26, 27, 28, 29],
                             [30, 31, 32, 33]]])
        mat2 = numpy.array([[[40, 41, 42, 43],
                             [44, 45, 46, 47]]])
        mat3 = numpy.array([[[10, 11, 12, 13],
                             [14, 15, 16, 17],
                             [18, 19, 20, 21]],
                            [[22, 23, 24, 25],
                             [26, 27, 28, 29],
                             [30, 31, 32, 33]],
                            [[40, 41, 42, 43],
                             [44, 45, 46, 47],
                             [-1, -1, -1, -1]]])
        _, dset1 = self.create_dset(mat1, -1)
        mat = vstack([dset1, mat2], -1)
        assert numpy.all(mat[:] == mat3)

    def test_set_chunk_by_chunk(self):
        mat = numpy.zeros(10, dtype=int)
        _set_matrix_by_chunks(mat, slice(5, 9), numpy.array([1, 1, 1, 1]),
                              chunk_size=2)
        assert numpy.all(mat == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0])

        _set_matrix_by_chunks(mat, slice(4, 8), numpy.array([2, 2, 2, 2]),
                              chunk_size=3)
        assert numpy.all(mat == [0, 0, 0, 0, 2, 2, 2, 2, 1, 0])

        mat = numpy.array([[0, 1], [2, 3], [4, 5]])
        _set_matrix_by_chunks(mat, slice(0, 2), numpy.array([[6, 7], [8, 9]]),
                              chunk_size=1)
        assert numpy.all(mat == [[6, 7], [8, 9], [4, 5]])

        mat = numpy.zeros(10, dtype=int)
        _set_matrix_by_chunks(mat, slice(5, 9), numpy.array([1, 1, 1, 1]),
                              chunk_size=20)
        assert numpy.all(mat == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0])

        mat = numpy.zeros(10, dtype=int)
        _set_matrix_by_chunks(mat, slice(-5, -1), numpy.array([1, 1, 1, 1]),
                              chunk_size=20)
        assert numpy.all(mat == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0])

    def test_3d_stacking_different_shapes(self):
        mat1 = numpy.array([[[10, 11, 12],
                             [14, 15, 16],
                             [18, 19, 20]],
                            [[22, 23, 24],
                             [26, 27, 28],
                             [30, 31, 32]]])
        mat2 = numpy.array([[[40, 41, 42, 43],
                             [44, 45, 46, 47]]])
        mat3 = numpy.array([[[10, 11, 12, -1],
                             [14, 15, 16, -1],
                             [18, 19, 20, -1]],
                            [[22, 23, 24, -1],
                             [26, 27, 28, -1],
                             [30, 31, 32, -1]],
                            [[40, 41, 42, 43],
                             [44, 45, 46, 47],
                             [-1, -1, -1, -1]]])
        _, dset1 = self.create_dset(mat1, -1)
        mat = vstack([dset1, mat2], -1)
        assert numpy.all(mat[:] == mat3)

        mat1 = numpy.array([[[10, 11, 12, 13],
                             [14, 15, 16, 17],
                             [18, 19, 20, 21]],
                            [[22, 23, 24, 25],
                             [26, 27, 28, 29],
                             [30, 31, 32, 33]]])
        mat2 = numpy.array([[[40, 41, 42],
                             [44, 45, 46]]])
        mat3 = numpy.array([[[10, 11, 12, 13],
                             [14, 15, 16, 17],
                             [18, 19, 20, 21]],
                            [[22, 23, 24, 25],
                             [26, 27, 28, 29],
                             [30, 31, 32, 33]],
                            [[40, 41, 42, -1],
                             [44, 45, 46, -1],
                             [-1, -1, -1, -1]]])
        _, dset1 = self.create_dset(mat1, -1)
        mat = vstack([dset1, mat2], -1)
        assert numpy.all(mat[:] == mat3)


class ResizeTest(unittest.TestCase):

    def test_resize(self):
        mat = numpy.array([[1, 1]])
        mat = resize_array(mat, shape=[2, 3], missing_value=-1)
        expected = [[1, 1, -1],
                    [-1, -1, -1]]
        assert numpy.all(mat == expected)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'VStackH5Test.test_3d_stacking_different_shapes']
    unittest.main()
