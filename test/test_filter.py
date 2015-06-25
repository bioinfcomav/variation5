
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

import h5py
import numpy

from variation.inout import dsets_chunks_iter, write_hdf5_from_chunks
from variation.filters import (filter_dsets_chunks,
                               _filter_all_gts_in_dsets_chunk,
                               _filter_no_gts_in_dsets_chunk,
                               select_dset_chunks_for_field,
                               keep_only_data_from_dset_chunks)
from variation.stats import RowValueCounter
from variation.utils.concat import concat_chunks_into_array


TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


class FilterTest(unittest.TestCase):
    def test_filter(self):

        fpath = join(TEST_DATA_DIR, '1000snps.hdf5')
        in_hdf5 = h5py.File(fpath, 'r')

        chunks = list(dsets_chunks_iter(in_hdf5))

        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            out_hdf5 = h5py.File(fhand.name, 'w')
            chunks_all = filter_dsets_chunks(_filter_all_gts_in_dsets_chunk,
                                         chunks)
            write_hdf5_from_chunks(out_hdf5, chunks_all)
            fhand.flush()

            hdf5_2 = h5py.File(fhand.name)
            assert hdf5_2['/calls/GT'].shape[0] == 0
            fhand.close()

        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            out_hdf5 = h5py.File(fhand.name, 'w')
            chunks_all = filter_dsets_chunks(_filter_all_gts_in_dsets_chunk,
                                         chunks)
            chunks_none = filter_dsets_chunks(_filter_no_gts_in_dsets_chunk,
                                         chunks)
            write_hdf5_from_chunks(out_hdf5, chunks)
            fhand.flush()

            hdf5_2 = h5py.File(fhand.name)
            assert numpy.all(in_hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
            fhand.close()

    def test_filter_missing(self):
        fpath = join(TEST_DATA_DIR, '1000snps.hdf5')
        hdf5 = h5py.File(fpath, 'r')
        chunks = dsets_chunks_iter(hdf5, kept_fields=['GT'])
        gt_chunks = select_dset_chunks_for_field(chunks, 'GT')
        gt_chunks = keep_only_data_from_dset_chunks(gt_chunks)
        calc_missing_data = RowValueCounter(value=-1, ratio=True)
        missing_chunks = map(calc_missing_data, gt_chunks)
        missing = concat_chunks_into_array(missing_chunks)
        assert abs(missing.max() - 0) < 0.01

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_vcf_parsing']
    unittest.main()
