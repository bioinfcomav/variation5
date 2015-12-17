
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import shutil
from os.path import join
from tempfile import NamedTemporaryFile

import numpy
from test.test_utils import TEST_DATA_DIR
from variation.variations import VariationsArrays, VariationsH5
from variation.variations.filters import (filter_mafs, filter_obs_het,
                                          filter_min_called_gts,
                                          set_low_qual_gts_to_missing,
                                          filter_snps_by_qual,
                                          set_low_dp_gts_to_missing,
                                          keep_biallelic,
                                          filter_monomorphic_snps,
                                          keep_biallelic_and_monomorphic,
                                          filter_samples)
from variation.iterutils import first


class FilterTest(unittest.TestCase):

    def test_filter_mafs(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        flt_chunk = filter_mafs(chunk, min_maf=0.6, min_num_genotypes=0,
                                by_chunk=False)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 182

        flt_chunk2 = filter_mafs(chunk, min_maf=0.6, min_num_genotypes=0,
                                 by_chunk=True)
        assert numpy.all(flt_chunk[path] == flt_chunk2[path])

        flt_chunk = filter_mafs(chunk)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 200

        flt_chunk = filter_mafs(chunk, max_maf=0.6)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 18

        flt_chunk = filter_mafs(chunk, min_maf=0.6, max_maf=0.9,
                                min_num_genotypes=0)
        assert flt_chunk[path].shape[0] == 125

        flt_chunk = filter_mafs(chunk, min_maf=1.1, min_num_genotypes=0)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

        flt_chunk = filter_mafs(chunk, max_maf=0, min_num_genotypes=0)
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

    def test_filter_called_gt(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]],
                           [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        variations['/calls/GT'] = gts

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
        filtered = filter_min_called_gts(variations, min_called=5, rates=False)
        assert numpy.all(filtered['/calls/GT'] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                               [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filtered = filter_min_called_gts(variations, min_called=2, rates=False)
        assert numpy.all(filtered['/calls/GT'] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                                [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                                [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filtered = filter_min_called_gts(variations, min_called=0.4,
                                         rates=True)
        assert numpy.all(filtered['/calls/GT'] == expected)

        filtered = filter_min_called_gts(variations, min_called=0, rates=True)
        assert numpy.all(filtered['/calls/GT'] == variations['/calls/GT'])

        # With hdf5 file
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_min_called_gts(hdf5, min_called=0.4, rates=True,
                                              by_chunk=True)
        filtered_h5_2 = filter_min_called_gts(hdf5, min_called=0.4, rates=True,
                                              by_chunk=False)

        res = filtered_h5_1['/calls/GT'] == filtered_h5_2['/calls/GT']
        assert numpy.all(res)

    def test_filter_mafs_2(self):
        # with some missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        variations['/calls/GT'] = gts
        filtered = filter_mafs(variations, min_num_genotypes=5)
        assert numpy.all(filtered['/calls/GT'] == gts)

        filtered = filter_mafs(variations, min_maf=0, min_num_genotypes=5)
        assert numpy.all(filtered['/calls/GT'] == gts[[0, 1, 2]])

        # without missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations['/calls/GT'] = gts

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_mafs(variations, max_maf=0.8, min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_mafs(variations, min_maf=0.6, max_maf=0.8,
                               min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == expected)

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        filtered = filter_mafs(variations, max_maf=0.5, min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == expected)

        filtered = filter_mafs(variations, min_maf=0.5, max_maf=1,
                               min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == variations['/calls/GT'])

        # With hdf5 files
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_mafs(hdf5, min_maf=0.6, max_maf=0.9,
                                    by_chunk=True)
        filtered_h5_2 = filter_mafs(hdf5, min_maf=0.6, max_maf=0.9,
                                    by_chunk=False)
        res = filtered_h5_1['/calls/GT'] == filtered_h5_2['/calls/GT']
        assert numpy.all(res)

    def test_filter_obs_het(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations['/calls/GT'] = gts

        filtered = filter_obs_het(variations, min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == gts)

        filtered = filter_obs_het(variations, min_het=0.2, min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == gts[[0, 2, 3]])

        filtered = filter_obs_het(variations, max_het=0.1, min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == gts[[1]])

        filtered = filter_obs_het(variations, max_het=0.3, min_het=0.2,
                                  min_num_genotypes=0)
        assert numpy.all(filtered['/calls/GT'] == gts[[0, 2, 3]])

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_obs_het(hdf5, min_het=0.6, max_het=0.9,
                                       by_chunk=True)
        filtered_h5_2 = filter_obs_het(hdf5, min_het=0.6, max_het=0.9,
                                       by_chunk=False)
        res = filtered_h5_1['/calls/GT'] == filtered_h5_2['/calls/GT']
        assert numpy.all(res)

    def test_filter_quality_genotype(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations['/calls/GT'] = gts
        variations['/calls/GQ'] = gqs

        set_low_qual_gts_to_missing(variations)
        filtered = variations['/calls/GT']
        assert numpy.all(filtered == gts)

        expected = numpy.array([[[0, 0], [1, 1], [-1, -1], [1, 1], [0, 0]],
                                [[0, 0], [-1, -1], [-1, -1], [0, 0],
                                 [-1, -1]]])
        set_low_qual_gts_to_missing(variations, min_qual=10, by_chunk=True)
        assert numpy.all(variations['/calls/GT'] == expected)

        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations['/calls/GT'] = gts
        variations['/calls/GQ'] = gqs
        set_low_qual_gts_to_missing(variations, min_qual=10, by_chunk=False)
        assert numpy.all(variations['/calls/GT'] == expected)

        set_low_qual_gts_to_missing(variations, min_qual=100)
        assert numpy.all(variations['/calls/GT'] == -1)

        h1_fhand = NamedTemporaryFile(suffix='.h5')
        h2_fhand = NamedTemporaryFile(suffix='.h5')
        shutil.copyfile(join(TEST_DATA_DIR, 'ril.hdf5'), h1_fhand.name)
        shutil.copyfile(join(TEST_DATA_DIR, 'ril.hdf5'), h2_fhand.name)

        h5_1 = VariationsH5(h1_fhand.name, mode='r+')
        set_low_qual_gts_to_missing(h5_1, min_qual=0, by_chunk=True)
        h5_2 = VariationsH5(h2_fhand.name, mode='r+')
        set_low_qual_gts_to_missing(h5_2, min_qual=0, by_chunk=False)
        assert numpy.all(h5_1['/calls/GT'][:] == h5_2['/calls/GT'][:])

    def test_filter_quality_snps(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1]], [[0, 1], [1, 1]],
                           [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                           [[0, 1], [0, 0]]])
        snp_quals = numpy.array([5, 10, 15, 5, 20])
        variations['/calls/GT'] = gts
        variations['/variations/qual'] = snp_quals

        filtered = filter_snps_by_qual(variations)
        filtered_qual = filtered['/variations/qual']
        filtered_gts = filtered['/calls/GT']
        assert numpy.all(variations['/variations/qual'] == filtered_qual)
        assert numpy.all(variations['/calls/GT'] == filtered_gts)

        expected_gts = numpy.array([[[0, 0], [0, 0]],
                                    [[0, 1], [0, 0]]])
        exp_snp_quals = numpy.array([15, 20])
        filtered = filter_snps_by_qual(variations, min_qual=15)
        assert numpy.all(filtered['/variations/qual'] == exp_snp_quals)
        assert numpy.all(filtered['/calls/GT'] == expected_gts)

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/variations/qual']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = filter_snps_by_qual(chunk, min_qual=530)
        assert first(flt_chunk.values()).shape[0] == 126

        flt_chunk = filter_snps_by_qual(chunk)
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = filter_snps_by_qual(chunk, max_qual=1000)
        assert first(flt_chunk.values()).shape[0] == 92

        flt_chunk = filter_snps_by_qual(chunk, min_qual=530, max_qual=1000)
        assert first(flt_chunk.values()).shape[0] == 18

        flt_chunk = filter_snps_by_qual(chunk, min_qual=586325202)
        assert first(flt_chunk.values()).shape[0] == 13

        flt_chunk = filter_snps_by_qual(chunk, max_qual=-1)
        assert first(flt_chunk.values()).shape[0] == 0

    def test_filter_quality_dp(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/DP', '/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        set_low_dp_gts_to_missing(chunk, min_dp=300)
        assert numpy.all(chunk['/calls/GT'][0][147] == [-1, -1])

        set_low_dp_gts_to_missing(chunk)
        assert numpy.all(chunk['/calls/GT'].shape[0] == 200)

    def test_filter_monomorfic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = filter_monomorphic_snps(chunk, min_maf=0.9,
                                            min_num_genotypes=0)
        assert flt_chunk.num_variations == 59

    def test_filter_biallelic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/GT']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = keep_biallelic_and_monomorphic(chunk)
        assert flt_chunk['/calls/GT'].shape == (200, 153, 2)
        flt_chunk = keep_biallelic(chunk)
        assert flt_chunk['/calls/GT'].shape == (174, 153, 2)


class FilterSamplesTest(unittest.TestCase):
    def test_filter_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        samples = ['1_14_1_gbs', '1_17_1_gbs', '1_18_4_gbs']
        varis = filter_samples(hdf5, samples=samples)
        assert varis.samples == samples
        assert varis['/calls/GT'].shape == (943, 3, 2)
        assert varis['/calls/GQ'].shape == (943, 3)
        assert varis['/variations/chrom'].shape == (943,)

        varis = filter_samples(hdf5, samples=samples, reverse=True,
                               by_chunk=True)
        assert all([sample not in samples for sample in varis.samples])
        n_samples = len(hdf5.samples)
        assert varis['/calls/GT'].shape == (943, n_samples - 3, 2)
        assert varis['/calls/GQ'].shape == (943, n_samples - 3)
        assert varis['/variations/chrom'].shape == (943,)

        varis2 = filter_samples(hdf5, samples=samples, reverse=True,
                                by_chunk=False)
        assert numpy.all(varis['/calls/GT'] == varis2['/calls/GT'])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'FilterSamplesTest']
    unittest.main()
