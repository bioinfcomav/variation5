
import unittest
import inspect
from os.path import dirname, abspath, join
from tempfile import NamedTemporaryFile

import numpy
import h5py

from variation.stats import RowValueCounter
from variation.inout import VCFParser, vcf_to_hdf5, dsets_chunks_iter
from variation.filters import (select_dset_chunks_for_field,
                               keep_only_data_from_dset_chunks)

# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


TEST_VCF2 = join(TEST_DATA_DIR, 'format_def.vcf')


class RowValueCounterTest(unittest.TestCase):
    def test_count_value_per_row(self):
        mat = numpy.array([[0, 0], [1, -1], [2, -1], [-1, -1]])
        missing_counter = RowValueCounter(value=-1)
        assert numpy.all(missing_counter(mat) == [0, 1, 1, 2])

        missing_counter = RowValueCounter(value=-1, ratio=True)
        assert numpy.allclose(missing_counter(mat), [0., 0.5, 0.5, 1.])


        vcf_fhand = open(TEST_VCF2, 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)

        with NamedTemporaryFile(suffix='.hdf5') as hdf5_fhand:

            vcf_to_hdf5(vcf, hdf5_fhand.name)
            hdf5_fhand.flush()
            hdf5 = h5py.File(hdf5_fhand.name, 'r')
            chunks = dsets_chunks_iter(hdf5)
            gt_chunks = select_dset_chunks_for_field(chunks, 'GT')
            gt_chunks = list(keep_only_data_from_dset_chunks(gt_chunks))
            vcf_fhand.close()
            homo_counter = RowValueCounter(value=2)
            assert numpy.all(homo_counter(gt_chunks[0]) == [0, 0, 4, 0, 1])

            missing_counter = RowValueCounter(value=2, ratio=True)
            expected = [0., 0, 0.66666, 0., 0.166666]
            assert numpy.allclose(missing_counter(gt_chunks[0]), expected)
            hdf5_fhand.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_vcf_parsing']
    unittest.main()
