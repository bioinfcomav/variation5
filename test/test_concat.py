
import unittest
import numpy
import h5py
from tempfile import NamedTemporaryFile

from variation.utils.concat import (concat_chunks_into_array,
                                    concat_chunks_into_dset)

# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111


class ConcatTest(unittest.TestCase):
    def test_concatenate(self):
        mats = [numpy.array([[1, 1], [2, 2]]), numpy.array([[3, 3]])]
        expected = [[1, 1], [2, 2], [3, 3]]

        assert numpy.all(concat_chunks_into_array(mats) == expected)
        assert numpy.all(concat_chunks_into_array(iter(mats)) == expected)
        assert numpy.all(concat_chunks_into_array(iter(mats),
                         concat_in_memory=True) == expected)

        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            hdf5 = h5py.File(fhand.name, 'w')
            grp = hdf5.create_group('concat')

            dset = concat_chunks_into_dset(mats, grp, 'concat',
                                           rows_in_chunk=2)
            assert numpy.all(dset[:] == [[1, 1], [2, 2], [3, 3]])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_vcf_parsing']
    unittest.main()
