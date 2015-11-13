
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import os
from os.path import join
from tempfile import NamedTemporaryFile

import numpy
from test.test_utils import TEST_DATA_DIR
from variation.variations import VariationsArrays, VariationsH5
from variation.variations.filters import (_filter_all,
                                          _filter_none,
                                          mafs_filter_fact,
                                          missing_rate_filter_fact,
                                          min_called_gts_filter_fact,
                                          quality_filter_genotypes_fact,
                                          quality_filter_snps_fact,
                                          filter_biallelic,
                                          filter_monomorphic_snps_fact,
                                          filter_biallelic_and_polymorphic,
                                          filter_gts_by_dp_fact,
                                          filter_gt_no_data)
from variation.iterutils import first


class FilterTest(unittest.TestCase):
    def test_filter_varh5(self):
        in_hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        chunks = list(in_hdf5.iterate_chunks())
        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            # out_hdf5 = h5py.File(fhand.name, 'w')
            os.remove(fhand.name)
            out_hdf5 = VariationsH5(fhand.name, mode='w')
            flt_chunks = map(_filter_all, chunks)
            out_hdf5.put_chunks(flt_chunks)

            hdf5_2 = VariationsH5(fhand.name, mode='r')
            assert hdf5_2['/calls/GT'].shape[0] == 0
            hdf5_2.close()

        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            # out_hdf5 = h5py.File(fhand.name, 'w')
            os.remove(fhand.name)
            out_hdf5 = VariationsH5(fhand.name, mode='w')
            flt_chunks = map(_filter_none, chunks)
            out_hdf5.put_chunks(flt_chunks)

            hdf5_2 = VariationsH5(fhand.name, mode='r')
            assert numpy.all(in_hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
            hdf5_2.close()

    def test_filter_missing_varh5(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        flt_chunk = min_called_gts_filter_fact(min_=100)(chunk)
        assert first(flt_chunk.values()).shape[0] == 94

        flt_chunk = min_called_gts_filter_fact()(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 200

        flt_chunk = min_called_gts_filter_fact(min_=307)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

        flt_chunk = missing_rate_filter_fact(min_=0.6)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 64

        flt_chunk = missing_rate_filter_fact()(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 200

        flt_chunk = missing_rate_filter_fact(min_=1.1)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

    def test_filter_mafs_varh5(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        filter_mafs = mafs_filter_fact(min_=0.6)
        flt_chunk = filter_mafs(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 182

        flt_chunk = mafs_filter_fact()(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 200

        flt_chunk = mafs_filter_fact(max_=0.6)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 18

        flt_chunk = mafs_filter_fact(min_=0.6, max_=0.9)(chunk)
        assert flt_chunk[path].shape[0] == 125

        flt_chunk = mafs_filter_fact(min_=1.1)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

        flt_chunk = mafs_filter_fact(max_=0)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

    def test_filter_varArray(self):
        var_h5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        chunks = var_h5.iterate_chunks()
        var_array = VariationsArrays()
        var_array.put_chunks(chunks)

        flt_var_array = VariationsArrays()
        flt_chunks = map(_filter_all, var_array.iterate_chunks())
        flt_var_array.put_chunks(flt_chunks)
        assert flt_var_array['/calls/GT'].shape[0] == 0

        flt_chunks = map(_filter_none, var_h5.iterate_chunks())
        out_snps = VariationsArrays()
        out_snps.put_chunks(flt_chunks)
        assert numpy.all(var_h5['/calls/GT'][:] == out_snps['/calls/GT'][:])

    def test_filter_missing_varArray(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        # Fichero de entrada esta mal, cogeremos solo algunos
        kept_fields = ['/calls/GT', '/variations/alt', '/variations/info/AF']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = min_called_gts_filter_fact(min_=100)(chunk)
        assert first(flt_chunk.values()).shape[0] == 94

        flt_chunk = min_called_gts_filter_fact()(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = min_called_gts_filter_fact(min_=307)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0

        flt_chunk = missing_rate_filter_fact(min_=0.6)(chunk)
        assert first(flt_chunk.values()).shape[0] == 64

        flt_chunk = missing_rate_filter_fact()(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = missing_rate_filter_fact(min_=1.1)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0

    def test_filter_mafs_varArray(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        # Fichero de entrada esta mal, cogeremos solo algunos
        kept_fields = ['/calls/GT', '/variations/alt', '/variations/info/AF']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
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

    def test_filter_quality_genotype(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/GQ', '/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = quality_filter_genotypes_fact(min_=0.6)(chunk)
        assert numpy.all(flt_chunk[0][147] == [-1, -1])

        flt_chunk = quality_filter_genotypes_fact()(chunk)
        assert numpy.all(flt_chunk.shape[0] == 200)

    def test_filter_quality_snps(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/variations/qual']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = quality_filter_snps_fact(min_=530)(chunk)
        assert first(flt_chunk.values()).shape[0] == 126

        flt_chunk = quality_filter_snps_fact()(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = quality_filter_snps_fact(max_=1000)(chunk)
        assert first(flt_chunk.values()).shape[0] == 92

        flt_chunk = quality_filter_snps_fact(min_=530, max_=1000)(chunk)
        assert first(flt_chunk.values()).shape[0] == 18

        flt_chunk = quality_filter_snps_fact(min_=586325202)(chunk)
        assert first(flt_chunk.values()).shape[0] == 13

        flt_chunk = quality_filter_snps_fact(max_=-1)(chunk)
        assert first(flt_chunk.values()).shape[0] == 0

    def test_filter_quality_dp(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/DP', '/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        filter_gts_by_dp = filter_gts_by_dp_fact(min_=300)
        flt_chunk = filter_gts_by_dp(chunk)
        assert numpy.all(flt_chunk[0][147] == [-1, -1])

        filter_gts_by_dp = filter_gts_by_dp_fact()
        flt_chunk = filter_gts_by_dp(chunk)
        assert numpy.all(flt_chunk.shape[0] == 200)

    def test_filter_monomorfic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        filter_monomorphic_snps = filter_monomorphic_snps_fact(0.9)
        flt_chunk = filter_monomorphic_snps(chunk)
        assert flt_chunk.num_variations == 59

    def test_filter_biallelic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = filter_biallelic(chunk)
        assert flt_chunk['/calls/GT'].shape == (200, 153, 2)
        flt_chunk = filter_biallelic_and_polymorphic(chunk)
        assert flt_chunk['/calls/GT'].shape == (174, 153, 2)

    def test_filter_no_data(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        h5 = VariationsH5(fpath, "r")
        result = filter_gt_no_data(h5)
        assert result['/calls/GT'].shape[0] == 2

if __name__ == "__main__":
#     import sys;sys.argv = ['', 'FilterTest.test_filter_missing_varArray']
    unittest.main()
