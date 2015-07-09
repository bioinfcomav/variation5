
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import inspect
import os
from os.path import dirname, abspath, join
from tempfile import NamedTemporaryFile

import numpy

from variation.vcfh5 import VcfH5
from variation.vcfh5.filters import (_filter_all,
                                     _filter_none,
                                     mafs_filter_fact,
                                     missing_rate_filter_fact,
                                     min_called_gts_filter_fact)
from variation.iterutils import first
#from variation.utils.concat import concat_chunks_into_array



TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


class FilterTest(unittest.TestCase):
    def test_filter(self):

        #fpath = join(TEST_DATA_DIR, '1000snps.hdf5')
        #in_hdf5 = h5py.File(fpath, 'r')
        in_hdf5 = VcfH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        chunks = list(in_hdf5.iterate_chunks())
        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            #out_hdf5 = h5py.File(fhand.name, 'w')
            os.remove(fhand.name)
            out_hdf5 = VcfH5(fhand.name, mode='w')
            flt_chunks = map(_filter_all, chunks)
            out_hdf5.write_chunks(flt_chunks)

            hdf5_2 = VcfH5(fhand.name, mode='r')
            assert hdf5_2['/calls/GT'].shape[0] == 0
            hdf5_2.close()

        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            #out_hdf5 = h5py.File(fhand.name, 'w')
            os.remove(fhand.name)
            out_hdf5 = VcfH5(fhand.name, mode='w')
            flt_chunks = map(_filter_none, chunks)
            out_hdf5.write_chunks(flt_chunks)

            hdf5_2 = VcfH5(fhand.name, mode='r')
            assert numpy.all(in_hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
            hdf5_2.close()

    def test_filter_missing(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())

        flt_chunk = min_called_gts_filter_fact(min_=250)(chunk)
        assert first(flt_chunk.values()).shape[0] == 62

        flt_chunk = min_called_gts_filter_fact()(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = min_called_gts_filter_fact(min_= 307)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0

        flt_chunk = missing_rate_filter_fact(min_=0.6)(chunk)
        assert first(flt_chunk.values()).shape[0] == 64

        flt_chunk = missing_rate_filter_fact()(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = missing_rate_filter_fact(min_= 1.1)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0

    #TODO test for maf filter
    def test_filter_mafs(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        filter_mafs = mafs_filter_fact(min_=0.6)
        flt_chunk = filter_mafs(chunk)
        assert first(flt_chunk.values()).shape[0] == 182

        flt_chunk = mafs_filter_fact()(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = mafs_filter_fact(max_=0.6)(chunk)
        assert first(flt_chunk.values()).shape[0] == 18

        flt_chunk = mafs_filter_fact(min_=0.6, max_=0.9)(chunk)
        assert first(flt_chunk.values()).shape[0] == 125

        flt_chunk = mafs_filter_fact(min_=1.1)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0

        flt_chunk = mafs_filter_fact(max_=0)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_vcf_parsing']
    unittest.main()
