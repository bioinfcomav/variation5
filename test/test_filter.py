
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
from collections import Counter

import numpy

from test.test_utils import TEST_DATA_DIR
from variation.variations import VariationsArrays, VariationsH5
from variation.variations.filters import (filter_mafs, filter_macs,
                                          filter_obs_het, flt_hist_obs_het,
                                          filter_min_called_gts,
                                          set_low_qual_gts_to_missing,
                                          filter_snps_by_qual,
                                          set_low_dp_gts_to_missing,
                                          keep_biallelic,
                                          filter_monomorphic_snps,
                                          keep_biallelic_and_monomorphic,
                                          filter_samples, filter_unlinked_vars,
                                          filter_samples_by_missing,
                                          filter_high_density_snps,
                                          filter_standarized_by_sample_depth,
                                          flt_hist_standarized_by_sample_depth,
                                          flt_hist_high_density_snps,
                                          flt_hist_samples_by_missing,
                                          flt_hist_chi2_gt_2_sample_sets,
                                          MinCalledGTsFilter, FLT_VARS,
                                          MafFilter)
from variation.iterutils import first
from variation import GT_FIELD, CHROM_FIELD, POS_FIELD, GQ_FIELD, DP_FIELD


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
        variations[GT_FIELD] = gts

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
        filtered = filter_min_called_gts(variations, min_called=5, rates=False)
        assert numpy.all(filtered[GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                               [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filtered = filter_min_called_gts(variations, min_called=2, rates=False)
        assert numpy.all(filtered[GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                                [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                                [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filtered = filter_min_called_gts(variations, min_called=0.4,
                                         rates=True)
        assert numpy.all(filtered[GT_FIELD] == expected)

        filtered = filter_min_called_gts(variations, min_called=0, rates=True)
        assert numpy.all(filtered[GT_FIELD] == variations[GT_FIELD])

        # With hdf5 file
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_min_called_gts(hdf5, min_called=0.4, rates=True,
                                              by_chunk=True)
        filtered_h5_2 = filter_min_called_gts(hdf5, min_called=0.4, rates=True,
                                              by_chunk=False)

        res = filtered_h5_1[GT_FIELD] == filtered_h5_2[GT_FIELD]
        assert numpy.all(res)

    def test_filter_mafs_2(self):
        # with some missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts
        filtered = filter_mafs(variations, min_num_genotypes=5)
        assert numpy.all(filtered[GT_FIELD] == gts)

        filtered = filter_mafs(variations, min_maf=0, min_num_genotypes=5)
        assert numpy.all(filtered[GT_FIELD] == gts[[0, 1, 2]])

        # without missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_mafs(variations, max_maf=0.8, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_mafs(variations, min_maf=0.6, max_maf=0.8,
                               min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        filtered = filter_mafs(variations, max_maf=0.5, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == expected)

        filtered = filter_mafs(variations, min_maf=0.5, max_maf=1,
                               min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == variations[GT_FIELD])

        # With hdf5 files
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_mafs(hdf5, min_maf=0.6, max_maf=0.9,
                                    by_chunk=True)
        filtered_h5_2 = filter_mafs(hdf5, min_maf=0.6, max_maf=0.9,
                                    by_chunk=False)
        res = filtered_h5_1[GT_FIELD] == filtered_h5_2[GT_FIELD]
        assert numpy.all(res)

    def test_filter_macs(self):
        # with some missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts
        filtered = filter_macs(variations, min_num_genotypes=5)
        assert numpy.all(filtered[GT_FIELD] == gts)

        filtered = filter_macs(variations, min_mac=0, min_num_genotypes=5)
        assert numpy.all(filtered[GT_FIELD] == gts[[0, 1, 2]])

        # without missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_macs(variations, max_mac=4, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = filter_macs(variations, min_mac=3.5, max_mac=4,
                               min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        filtered = filter_macs(variations, max_mac=3, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == expected)

        filtered = filter_macs(variations, min_mac=2, max_mac=5,
                               min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == variations[GT_FIELD])

        # With hdf5 files
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_macs(hdf5, min_mac=130, max_mac=150,
                                    by_chunk=True)
        filtered_h5_2 = filter_macs(hdf5, min_mac=130, max_mac=150,
                                    by_chunk=False)
        res = filtered_h5_1[GT_FIELD] == filtered_h5_2[GT_FIELD]
        assert numpy.all(res)

    def test_filter_obs_het(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts

        filtered = filter_obs_het(variations, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == gts)

        filtered = filter_obs_het(variations, min_het=0.2, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == gts[[0, 2, 3]])

        filtered = filter_obs_het(variations, max_het=0.1, min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == gts[[1]])

        filtered = filter_obs_het(variations, max_het=0.3, min_het=0.2,
                                  min_num_genotypes=0)
        assert numpy.all(filtered[GT_FIELD] == gts[[0, 2, 3]])

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered_h5_1 = filter_obs_het(hdf5, min_het=0.6, max_het=0.9,
                                       by_chunk=True)
        filtered_h5_2 = filter_obs_het(hdf5, min_het=0.6, max_het=0.9,
                                       by_chunk=False)
        res = filtered_h5_1[GT_FIELD] == filtered_h5_2[GT_FIELD]
        assert numpy.all(res)

        filtered_h5_1 = filter_obs_het(hdf5, min_het=0.6, max_het=0.9,
                                       min_call_dp=5, by_chunk=True)
        filtered_h5_2 = filter_obs_het(hdf5, min_het=0.6, max_het=0.9,
                                       min_call_dp=5, by_chunk=False)
        res = filtered_h5_1[GT_FIELD] == filtered_h5_2[GT_FIELD]
        assert numpy.all(res)

        filtered_h5_3, cnts, edges = flt_hist_obs_het(hdf5, min_het=0.6,
                                                      max_het=0.9,
                                                      min_call_dp=5,
                                                      n_bins=3, range_=(0, 1))
        res = filtered_h5_1[GT_FIELD] == filtered_h5_3[GT_FIELD]
        assert numpy.all(res)
        assert numpy.all(cnts == [391, 14, 10])
        assert numpy.all(edges == [0, 1 / 3, 2 / 3, 1])

        samples = hdf5.samples[:50]
        _, cnts, _ = flt_hist_obs_het(hdf5, samples=samples, min_het=0.6,
                                      max_het=0.9, min_call_dp=5, n_bins=3,
                                      range_=(0, 1))
        assert numpy.all(cnts == [340, 14, 5])

    def test_filter_quality_genotype(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations[GT_FIELD] = gts
        variations[GQ_FIELD] = gqs

        set_low_qual_gts_to_missing(variations)
        filtered = variations[GT_FIELD]
        assert numpy.all(filtered == gts)

        expected = numpy.array([[[0, 0], [1, 1], [-1, -1], [1, 1], [0, 0]],
                                [[0, 0], [-1, -1], [-1, -1], [0, 0],
                                 [-1, -1]]])
        set_low_qual_gts_to_missing(variations, min_qual=10, by_chunk=True)
        assert numpy.all(variations[GT_FIELD] == expected)

        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations[GT_FIELD] = gts
        variations[GQ_FIELD] = gqs
        set_low_qual_gts_to_missing(variations, min_qual=10, by_chunk=False)
        assert numpy.all(variations[GT_FIELD] == expected)

        set_low_qual_gts_to_missing(variations, min_qual=100)
        assert numpy.all(variations[GT_FIELD] == -1)

        h1_fhand = NamedTemporaryFile(suffix='.h5')
        h2_fhand = NamedTemporaryFile(suffix='.h5')
        shutil.copyfile(join(TEST_DATA_DIR, 'ril.hdf5'), h1_fhand.name)
        shutil.copyfile(join(TEST_DATA_DIR, 'ril.hdf5'), h2_fhand.name)

        h5_1 = VariationsH5(h1_fhand.name, mode='r+')
        set_low_qual_gts_to_missing(h5_1, min_qual=0, by_chunk=True)
        h5_2 = VariationsH5(h2_fhand.name, mode='r+')
        set_low_qual_gts_to_missing(h5_2, min_qual=0, by_chunk=False)
        assert numpy.all(h5_1[GT_FIELD][:] == h5_2[GT_FIELD][:])

    def test_filter_quality_snps(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1]], [[0, 1], [1, 1]],
                           [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                           [[0, 1], [0, 0]]])
        snp_quals = numpy.array([5, 10, 15, 5, 20])
        variations[GT_FIELD] = gts
        variations['/variations/qual'] = snp_quals

        filtered = filter_snps_by_qual(variations)
        filtered_qual = filtered['/variations/qual']
        filtered_gts = filtered[GT_FIELD]
        assert numpy.all(variations['/variations/qual'] == filtered_qual)
        assert numpy.all(variations[GT_FIELD] == filtered_gts)

        expected_gts = numpy.array([[[0, 0], [0, 0]],
                                    [[0, 1], [0, 0]]])
        exp_snp_quals = numpy.array([15, 20])
        filtered = filter_snps_by_qual(variations, min_qual=15)
        assert numpy.all(filtered['/variations/qual'] == exp_snp_quals)
        assert numpy.all(filtered[GT_FIELD] == expected_gts)

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
        kept_fields = ['/calls/DP', GT_FIELD]
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        set_low_dp_gts_to_missing(chunk, min_dp=300)
        assert numpy.all(chunk[GT_FIELD][0][147] == [-1, -1])

        set_low_dp_gts_to_missing(chunk)
        assert numpy.all(chunk[GT_FIELD].shape[0] == 200)

    def test_filter_monomorfic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = [GT_FIELD]
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = filter_monomorphic_snps(chunk, min_maf=0.9,
                                            min_num_genotypes=0)
        assert flt_chunk.num_variations == 59

    def test_filter_biallelic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = [GT_FIELD]
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = keep_biallelic_and_monomorphic(chunk)
        assert flt_chunk[GT_FIELD].shape == (200, 153, 2)
        flt_chunk = keep_biallelic(chunk)
        assert flt_chunk[GT_FIELD].shape == (174, 153, 2)

    def test_filter_unlinked_vars(self):
        varis = VariationsArrays()
        varis[GT_FIELD] = numpy.array([[[0, 0], [0, 0], [0, 1]],
                                      [[0, 0], [0, 0], [0, 1]],
                                      [[1, 1], [0, 1], [0, 0]],
                                      [[1, 1], [0, 1], [0, 0]]])
        varis[CHROM_FIELD] = numpy.array([b'chr1'] * 4)
        varis[POS_FIELD] = numpy.array([1, 10, 100, 110])
        expected = [[[0, 0], [0, 0], [0, 1]]]
        filtered_vars = filter_unlinked_vars(varis, window_size=50,
                                             by_chunk=False)
        assert numpy.all(filtered_vars[GT_FIELD] == expected)

        filtered_vars = filter_unlinked_vars(varis, window_size=50,
                                             by_chunk=False, r2_threshold=1)
        assert numpy.all(filtered_vars[GT_FIELD] == varis[GT_FIELD])

        expected = [[[0, 0], [0, 0], [0, 1]], [[1, 1], [0, 1], [0, 0]]]
        filtered_vars = filter_unlinked_vars(varis, window_size=50,
                                             by_chunk=False, r2_threshold=.9)
        assert numpy.all(filtered_vars[GT_FIELD] == expected)

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        vars1 = filter_unlinked_vars(hdf5, 1000, by_chunk=True)
        vars2 = filter_unlinked_vars(hdf5, 1000, by_chunk=False)
        assert numpy.all(vars1[GT_FIELD] == vars2[GT_FIELD])

    def test_filter_high_density(self):
        varis = VariationsArrays()
        flt_varis = filter_high_density_snps(varis, max_density=1, window=1)
        assert not flt_varis.keys()

        varis = VariationsArrays()
        varis[CHROM_FIELD] = numpy.array([b'chr1'] * 6)
        varis[POS_FIELD] = numpy.array([1, 2, 3, 4, 10, 11])
        flt_varis = filter_high_density_snps(varis, max_density=2, window=3)
        assert list(flt_varis[POS_FIELD]) == [1, 4, 10, 11]

        flt_varis = filter_high_density_snps(varis, max_density=2, window=3,
                                             chunk_size=1)
        assert list(flt_varis[POS_FIELD]) == [1, 4, 10, 11]

        res = flt_hist_high_density_snps(varis, max_density=2, window=3,
                                         n_bins=2)
        flt_varis, counts, edges = res
        assert list(flt_varis[POS_FIELD]) == [1, 4, 10, 11]
        assert list(counts) == [4, 2]
        assert list(edges) == [2., 2.5, 3.]

    def test_filter_snp_by_std_depth(self):
        vars_ = VariationsArrays()
        dps = numpy.array([[4, 2, 2, 0, 0], [2, 1, 1, 0, 1]])
        vars_['/calls/DP'] = dps
        vars2 = filter_standarized_by_sample_depth(vars_, max_std_dp=None)
        numpy.allclose(vars2['/calls/DP'], dps)
        vars2 = filter_standarized_by_sample_depth(vars_, max_std_dp=1.5)
        numpy.allclose(vars2['/calls/DP'], dps[0, :])

        vars2, counts, edges = flt_hist_standarized_by_sample_depth(vars_,
                                                                    n_bins=2)
        numpy.allclose(vars2['/calls/DP'], dps)
        numpy.allclose(counts, [1, 1])
        numpy.allclose(edges, [1, 1.5, 2])

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        snps = VariationsArrays()
        snps.put_chunks(hdf5.iterate_chunks())
        vars2 = filter_standarized_by_sample_depth(hdf5, max_std_dp=1.5)
        vars3 = filter_standarized_by_sample_depth(snps, max_std_dp=1.5)
        assert numpy.allclose(vars2[DP_FIELD], vars3[DP_FIELD])
        vars3 = filter_standarized_by_sample_depth(snps, max_std_dp=1.5,
                                                   chunk_size=0)
        assert numpy.allclose(vars2[DP_FIELD], vars3[DP_FIELD])
        res = flt_hist_standarized_by_sample_depth(snps, max_std_dp=1.5)
        vars3 = res[0]
        assert numpy.allclose(vars2[DP_FIELD], vars3[DP_FIELD])

        flt_hist_standarized_by_sample_depth(snps, max_std_dp=1.5,
                                             samples=snps.samples[1:20])

    def test_filter_chi2_gt_sample_sets(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [1, 1], [1, 1], [1, 1]],
                           [[0, 0], [0, 0], [0, 1], [1, 1], [1, 1], [1, 1]]])
        variations[GT_FIELD] = gts
        variations.samples = [1, 2, 3, 4, 5, 6]
        samples1 = [1, 2, 3]
        samples2 = [4, 5, 6]
        res = flt_hist_chi2_gt_2_sample_sets(variations, samples1,
                                                  samples2, min_pval=0.05,
                                                  n_bins=2)
        filtered, counts, _ = res
        assert list(counts) == [2, 2]
        assert numpy.all(filtered[GT_FIELD] == gts[:2, ...])


class FilterSamplesTest(unittest.TestCase):
    def test_filter_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        samples = ['1_14_1_gbs', '1_17_1_gbs', '1_18_4_gbs']
        varis = filter_samples(hdf5, samples=samples)
        assert varis.samples == samples
        assert varis[GT_FIELD].shape == (943, 3, 2)
        assert varis[GQ_FIELD].shape == (943, 3)
        assert varis[CHROM_FIELD].shape == (943,)

        varis = filter_samples(hdf5, samples=samples, reverse=True,
                               by_chunk=True)
        assert all([sample not in samples for sample in varis.samples])
        n_samples = len(hdf5.samples)
        assert varis[GT_FIELD].shape == (943, n_samples - 3, 2)
        assert varis[GQ_FIELD].shape == (943, n_samples - 3)
        assert varis[CHROM_FIELD].shape == (943,)

        varis2 = filter_samples(hdf5, samples=samples, reverse=True,
                                by_chunk=False)
        assert numpy.all(varis[GT_FIELD] == varis2[GT_FIELD])

    def test_filter_samples_by_missing(self):
        variations = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(variations.iterate_chunks())

        new_var = filter_samples_by_missing(chunk, 0.9)
        assert len(new_var.samples) == 0

        new_var = filter_samples_by_missing(chunk, 0.1)
        assert len(new_var.samples) == len(chunk.samples)

        new_var2 = flt_hist_samples_by_missing(chunk, 0.1)[0]
        assert len(new_var.samples) == len(new_var2.samples)


class MinCalledGTTest(unittest.TestCase):
    def test_filter_called_gt(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]],
                           [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        variations[GT_FIELD] = gts

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]])
        filter_gts = MinCalledGTsFilter(min_called=5, rates=False)
        filtered = filter_gts(variations)
        assert numpy.all(filtered['flt_vars'][GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                               [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filter_gts = MinCalledGTsFilter(min_called=2, rates=False)
        filtered = filter_gts(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                                [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                                [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]]])
        filter_gts = MinCalledGTsFilter(min_called=0.4, rates=True)
        filtered = filter_gts(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        filter_gts = MinCalledGTsFilter(min_called=0, rates=True)
        filtered = filter_gts(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == variations[GT_FIELD])

        # With hdf5 file
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filter_gts = MinCalledGTsFilter(min_called=0.4, rates=True)
        filtered = filter_gts(hdf5)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert counts == {0: 89936, 1: 50473, -1: 40972, 2: 378, 3: 5}


class MafFilterTest(unittest.TestCase):

    def test_filter_mafs(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(hdf5.iterate_chunks())
        filtered = MafFilter(min_maf=0.6, min_num_genotypes=0)(chunk)
        flt_chunk = filtered[FLT_VARS]

        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 182

        filtered = MafFilter()(chunk)
        flt_chunk = filtered[FLT_VARS]
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 200

        filtered = MafFilter(max_maf=0.6)(chunk)
        flt_chunk = filtered[FLT_VARS]
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 18

        filtered = MafFilter(min_maf=0.6, max_maf=0.9,
                             min_num_genotypes=0)(chunk)
        flt_chunk = filtered[FLT_VARS]
        assert flt_chunk[path].shape[0] == 125

        filtered = MafFilter(min_maf=1.1, min_num_genotypes=0)(chunk)
        flt_chunk = filtered[FLT_VARS]
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

        filtered = MafFilter(max_maf=0, min_num_genotypes=0)(chunk)
        flt_chunk = filtered[FLT_VARS]
        path = first(chunk.keys())
        assert flt_chunk[path].shape[0] == 0

    def test_filter_mafs_2(self):
        # with some missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts
        filtered = MafFilter(min_num_genotypes=5)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts)

        filtered = MafFilter(min_num_genotypes=5, min_maf=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts[[0, 1, 2]])

        # without missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = MafFilter(min_num_genotypes=0, max_maf=0.8)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = MafFilter(min_num_genotypes=0, min_maf=0.6,
                             max_maf=0.8)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        filtered = MafFilter(min_num_genotypes=0, max_maf=0.5)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        filtered = MafFilter(min_num_genotypes=0, min_maf=0.5,
                             max_maf=1)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == variations[GT_FIELD])

        # With hdf5 files
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered = MafFilter(min_maf=0.6, max_maf=0.9)(hdf5)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert counts == {0: 57805, -1: 55792, 1: 32504, 2: 162, 3: 5}

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'FilterTest']
    unittest.main()
