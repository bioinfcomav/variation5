
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
                                          min_called_gts_filter_fact,
                                          quality_filter_genotypes_fact,
                                          quality_filter_snps_fact,
                                          filter_biallelic,
                                          filter_monomorphic_snps_fact,
                                          filter_biallelic_and_polymorphic,
                                          filter_gts_by_dp_fact,
                                          obs_het_filter_fact)
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

    def test_filter_mafs_varh5(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        filter_mafs = mafs_filter_fact(min_=0.6, min_num_genotypes=0)
        flt_chunk = filter_mafs(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 182

        flt_chunk = mafs_filter_fact()(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 200

        flt_chunk = mafs_filter_fact(max_=0.6)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 18

        flt_chunk = mafs_filter_fact(min_=0.6, max_=0.9,
                                     min_num_genotypes=0)(chunk)
        assert flt_chunk[path].shape[0] == 125

        flt_chunk = mafs_filter_fact(min_=1.1,
                                     min_num_genotypes=0)(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

        flt_chunk = mafs_filter_fact(max_=0,
                                     min_num_genotypes=0)(chunk)
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

    def test_filter_called_gt(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]],
                           [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        variations['/calls/GT'] = gts
        
        filter_by_called = min_called_gts_filter_fact(min_=5, rates=False)
        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
        filtered = filter_by_called(variations)
        assert numpy.all(filtered['/calls/GT'] == expected)
        
        filter_by_called = min_called_gts_filter_fact(min_=2, rates=False)
        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filtered = filter_by_called(variations)
        assert numpy.all(filtered['/calls/GT'] == expected)
        
        filter_by_called = min_called_gts_filter_fact(min_=0.4, rates=True)
        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filtered = filter_by_called(variations)
        assert numpy.all(filtered['/calls/GT'] == expected)
        
        filter_by_called = min_called_gts_filter_fact(min_=0, rates=True)
        filtered = filter_by_called(variations)
        assert numpy.all(filtered['/calls/GT'] == variations['/calls/GT'])
        
        # With hdf5 file
        filter_by_called = min_called_gts_filter_fact(min_=0.4, rates=True)
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_hdf5 = filter_by_called(hdf5)
        
        chunks = hdf5.iterate_chunks(kept_fields=['/calls/GT'])
        filtered_vars = VariationsArrays()
        filtered_chunks = map(filter_by_called, chunks)
        filtered_vars.put_chunks(filtered_chunks)
        
        res = filtered_hdf5['/calls/GT'] == filtered_vars['/calls/GT']
        assert numpy.all(res)

    def test_filter_mafs(self):
        # with some missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        variations['/calls/GT'] = gts
        filter_mafs = mafs_filter_fact(min_num_genotypes=5)
        filtered = filter_mafs(variations)
        assert numpy.all(filtered['/calls/GT'] == gts)
        
        filter_mafs = mafs_filter_fact(min_=0, min_num_genotypes=5)
        filtered = filter_mafs(variations)
        assert numpy.all(filtered['/calls/GT'] == gts[[0, 1, 2]])

        # without missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations['/calls/GT'] = gts
        
        filter_maf = mafs_filter_fact(max_=0.8, min_num_genotypes=0)
        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_maf(variations)
        assert numpy.all(filtered['/calls/GT'] == expected)
        
        filter_maf = mafs_filter_fact(min_=0.6, max_=0.8, min_num_genotypes=0)
        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_maf(variations)
        assert numpy.all(filtered['/calls/GT'] == expected)
        
        filter_maf = mafs_filter_fact(max_=0.5, min_num_genotypes=0)
        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        filtered = filter_maf(variations)
        assert numpy.all(filtered['/calls/GT'] == expected)
        
        filter_maf = mafs_filter_fact(min_=0.5, max_=1, min_num_genotypes=0)
        filtered = filter_maf(variations)
        assert numpy.all(filtered['/calls/GT'] == variations['/calls/GT'])
        
        # With hdf5 files        
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunks = hdf5.iterate_chunks(kept_fields=['/calls/GT'])
        filter_mafs = mafs_filter_fact(min_=0.6, max_=0.9)
        
        filtered_hdf5 = filter_mafs(hdf5)
        filtered_vars = VariationsArrays()
        filtered_chunks = map(filter_mafs, chunks)
        filtered_vars.put_chunks(filtered_chunks)
        res = filtered_hdf5['/calls/GT'] == filtered_vars['/calls/GT']
        assert numpy.all(res)
        
    def test_filter_obs_het(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations['/calls/GT'] = gts
        
        filter_obs_het = obs_het_filter_fact(min_num_genotypes=0)
        filtered = filter_obs_het(variations)
        assert numpy.all(filtered['/calls/GT'] == gts)
        
        filter_obs_het = obs_het_filter_fact(min_=0.2, min_num_genotypes=0)
        filtered = filter_obs_het(variations)
        assert numpy.all(filtered['/calls/GT'] == gts[[0, 2, 3]])
        
        filter_obs_het = obs_het_filter_fact(max_=0.1, min_num_genotypes=0)
        filtered = filter_obs_het(variations)
        assert numpy.all(filtered['/calls/GT'] == gts[[1]])
        
        filter_obs_het = obs_het_filter_fact(max_=0.3, min_=0.2,
                                             min_num_genotypes=0)
        filtered = filter_obs_het(variations)
        assert numpy.all(filtered['/calls/GT'] == gts[[0, 2, 3]])
        
        # With hdf5 files        
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunks = hdf5.iterate_chunks(kept_fields=['/calls/GT'])
        filter_obs_het = obs_het_filter_fact(min_=0.6, max_=0.9)
        
        filtered_hdf5 = filter_obs_het(hdf5)
        filtered_vars = VariationsArrays()
        filtered_chunks = map(filter_obs_het, chunks)
        filtered_vars.put_chunks(filtered_chunks)
        res = filtered_hdf5['/calls/GT'] == filtered_vars['/calls/GT']
        assert numpy.all(res)

    def test_filter_quality_genotype(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations['/calls/GT'] = gts
        variations['/calls/GQ'] = gqs
        
        filter_by_gq = quality_filter_genotypes_fact()
        filtered = filter_by_gq(variations)
        assert numpy.all(filtered == variations['/calls/GT'])
        
        filter_by_gq = quality_filter_genotypes_fact(min_=10)
        expected = numpy.array([[[0, 0], [1, 1], [-1, -1], [1, 1], [0, 0]],
                                [[0, 0], [-1, -1], [-1, -1], [0, 0],
                                 [-1, -1]]])
        filtered = filter_by_gq(variations)
        assert numpy.all(filtered == expected)
        
        filter_by_gq = quality_filter_genotypes_fact(min_=100)
        filtered = filter_by_gq(variations)
        assert numpy.all(filtered == -1)
        
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/GQ', '/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = quality_filter_genotypes_fact(min_=0.6)(chunk)
        assert numpy.all(flt_chunk[0][147] == [-1, -1])

        flt_chunk = quality_filter_genotypes_fact()(chunk)
        assert numpy.all(flt_chunk.shape[0] == 200)

    def test_filter_quality_snps(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1]], [[0, 1], [1, 1]],
                            [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                            [[0, 1], [0, 0]]])
        snp_quals = numpy.array([5, 10, 15, 5, 20])
        variations['/calls/GT'] = gts
        variations['/variations/qual'] = snp_quals
        
        filter_by_qual = quality_filter_snps_fact()
        filtered = filter_by_qual(variations)
        filtered_qual = filtered['/variations/qual']
        filtered_gts = filtered['/calls/GT']
        assert numpy.all(variations['/variations/qual'] == filtered_qual)
        assert numpy.all(variations['/calls/GT'] == filtered_gts)
        
        filter_by_qual = quality_filter_snps_fact(min_=15)
        expected_gts = numpy.array([[[0, 0], [0, 0]],
                                    [[0, 1], [0, 0]]])
        exp_snp_quals = numpy.array([15, 20])
        filtered = filter_by_qual(variations)
        assert numpy.all(filtered['/variations/qual'] == exp_snp_quals)
        assert numpy.all(filtered['/calls/GT'] == expected_gts)
        
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
        filter_monomorphic_snps = filter_monomorphic_snps_fact(0.9,
                                                               min_num_genotypes=0)
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

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'FilterTest.test_filter_mafs']
    unittest.main()
