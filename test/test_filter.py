
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

from collections import Counter

import numpy

from test.test_utils import TEST_DATA_DIR
from variation.variations import VariationsArrays, VariationsH5
from variation.variations.filters import (MinCalledGTsFilter, FLT_VARS, COUNTS,
                                          EDGES, MafFilter, MacFilter,
                                          ObsHetFilter, SNPQualFilter, TOT,
                                          LowDPGTsToMissingSetter, FLT_STATS,
                                          LowQualGTsToMissingSetter, N_KEPT,
                                          NonBiallelicFilter,
                                          Chi2GtFreqs2SampleSetsFilter,
                                          SampleFilter, FieldFilter,
                                          filter_samples_by_missing_rate,
                                          filter_variation_density,
                                          N_FILTERED_OUT)
from variation.iterutils import first
from variation import GT_FIELD, CHROM_FIELD, POS_FIELD, GQ_FIELD, DP_FIELD


class FilterTest(unittest.TestCase):

    def test_filter_high_density(self):
        in_vars = VariationsArrays()
        out_vars = VariationsArrays()
        res = filter_variation_density(in_vars, out_vars=out_vars,
                                       max_density=1, window=1)
        assert not out_vars.keys()
        assert not res

        varis = VariationsArrays()
        out_vars = VariationsArrays()
        varis[CHROM_FIELD] = numpy.array([b'chr1'] * 6)
        varis[POS_FIELD] = numpy.array([1, 2, 3, 4, 10, 11])
        result = filter_variation_density(varis, out_vars=out_vars,
                                          max_density=2, window=3, n_bins=2)
        assert list(out_vars[POS_FIELD]) == [1, 4, 10, 11]
        assert list(result[COUNTS]) == [4, 2]
        assert list(result[EDGES]) == [2., 2.5, 3.]
        assert result[FLT_STATS][N_KEPT] == 4
        assert result[FLT_STATS][TOT] == 6
        assert result[FLT_STATS][N_FILTERED_OUT] == 2

        varis = VariationsArrays()
        out_vars = VariationsArrays()
        varis[CHROM_FIELD] = numpy.array([b'chr1'] * 6)
        varis[POS_FIELD] = numpy.array([1, 2, 3, 4, 10, 11])
        result = filter_variation_density(varis, out_vars=out_vars,
                                          max_density=2, window=3, n_bins=2,
                                          chunk_size=1)
        assert list(out_vars[POS_FIELD]) == [1, 4, 10, 11]
        assert list(result[COUNTS]) == [4, 2]
        assert list(result[EDGES]) == [2., 2.5, 3.]
        assert result[FLT_STATS][N_KEPT] == 4
        assert result[FLT_STATS][TOT] == 6
        assert result[FLT_STATS][N_FILTERED_OUT] == 2

        varis = VariationsArrays()
        out_vars = VariationsArrays()
        varis[CHROM_FIELD] = numpy.array([b'chr1'] * 6)
        varis[POS_FIELD] = numpy.array([1, 2, 3, 4, 10, 11])
        result = filter_variation_density(varis, out_vars=out_vars,
                                          max_density=2, window=3, n_bins=2,
                                          chunk_size=None)
        assert list(out_vars[POS_FIELD]) == [1, 4, 10, 11]
        assert list(result[COUNTS]) == [4, 2]
        assert list(result[EDGES]) == [2., 2.5, 3.]
        assert result[FLT_STATS][N_KEPT] == 4
        assert result[FLT_STATS][TOT] == 6
        assert result[FLT_STATS][N_FILTERED_OUT] == 2


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
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)
        assert filtered[FLT_STATS][N_KEPT] == 1
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 3

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
        assert filtered[FLT_STATS][N_KEPT] == 182
        assert filtered[FLT_STATS][TOT] == 200
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 18

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
        assert filtered[FLT_STATS][N_KEPT] == 4
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 0

        filtered = MafFilter(min_num_genotypes=5, min_maf=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts[[0, 1, 2]])
        assert filtered[FLT_STATS][N_KEPT] == 3
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 1

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

    def test_filter_macs(self):
        # with some missing values
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts
        filtered = MacFilter(min_num_genotypes=5)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts)
        assert filtered[FLT_STATS][N_KEPT] == 4
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 0

        filtered = MacFilter(min_mac=0, min_num_genotypes=5)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts[[0, 1, 2]])
        assert filtered[FLT_STATS][N_KEPT] == 3
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 1

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
        filtered = MacFilter(max_mac=4, min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        filtered = MacFilter(min_mac=3.5, max_mac=4,
                             min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        expected = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        filtered = MacFilter(max_mac=3, min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected)

        filtered = MacFilter(min_mac=2, max_mac=5,
                             min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == variations[GT_FIELD])

        # With hdf5 files
        variations = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered = MacFilter(min_mac=130, max_mac=150)(variations)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert counts == {-1: 64530, 0: 36977, 1: 18716, 2: 35}


class ObsHetFiltterTest(unittest.TestCase):

    def test_filter_obs_het(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        variations[GT_FIELD] = gts
        variations.samples = [1, 2, 3, 4, 5]

        filtered = ObsHetFilter(min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts)
        assert filtered[FLT_STATS][N_KEPT] == 4
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 0

        filtered = ObsHetFilter(min_het=0.2, min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts[[0, 2, 3]])
        assert filtered[FLT_STATS][N_KEPT] == 3
        assert filtered[FLT_STATS][TOT] == 4
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 1

        filtered = ObsHetFilter(max_het=0.1, min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts[[1]])

        filtered = ObsHetFilter(min_het=0.2, max_het=0.3,
                                min_num_genotypes=0)(variations)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == gts[[0, 2, 3]])

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        filtered = ObsHetFilter(min_het=0.6, max_het=0.9)(hdf5)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert counts == {}

        filtered = ObsHetFilter(min_het=0.6, max_het=0.9, min_call_dp=5)(hdf5)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert counts == {0: 978, -1: 910, 1: 774, 2: 92}

        filtered = ObsHetFilter(min_het=0.6, max_het=0.9, min_call_dp=5,
                                n_bins=3, range_=(0, 1))(hdf5)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert counts == {0: 978, -1: 910, 1: 774, 2: 92}
        assert numpy.all(filtered[COUNTS] == [391, 14, 10])
        assert numpy.all(filtered[EDGES] == [0, 1 / 3, 2 / 3, 1])

        samples = hdf5.samples[:50]
        filtered = ObsHetFilter(min_het=0.6, max_het=0.9, min_call_dp=5,
                                n_bins=3, range_=(0, 1), samples=samples)(hdf5)
        counts = Counter(filtered[FLT_VARS][GT_FIELD].flat)
        assert numpy.all(filtered[COUNTS] == [339, 14, 6])


class SNPQualFilterTest(unittest.TestCase):

    def test_filter_quality_snps(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1]], [[0, 1], [1, 1]],
                           [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                           [[0, 1], [0, 0]]])
        snp_quals = numpy.array([5, 10, 15, 5, 20])
        variations[GT_FIELD] = gts
        variations['/variations/qual'] = snp_quals

        filtered = SNPQualFilter()(variations)[FLT_VARS]
        filtered_qual = filtered['/variations/qual']
        filtered_gts = filtered[GT_FIELD]
        assert numpy.all(variations['/variations/qual'] == filtered_qual)
        assert numpy.all(variations[GT_FIELD] == filtered_gts)

        expected_gts = numpy.array([[[0, 0], [0, 0]],
                                    [[0, 1], [0, 0]]])
        exp_snp_quals = numpy.array([15, 20])
        filtered = SNPQualFilter(min_qual=15)(variations)
        assert numpy.all(filtered[FLT_VARS]['/variations/qual'] ==
                         exp_snp_quals)
        assert numpy.all(filtered[FLT_VARS][GT_FIELD] == expected_gts)
        assert filtered[FLT_STATS][N_KEPT] == 2
        assert filtered[FLT_STATS][TOT] == 5
        assert filtered[FLT_STATS][N_FILTERED_OUT] == 3

        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/variations/qual']
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        flt_chunk = SNPQualFilter(min_qual=530)(chunk)[FLT_VARS]
        assert first(flt_chunk.values()).shape[0] == 126

        flt_chunk = SNPQualFilter()(chunk)[FLT_VARS]
        assert first(flt_chunk.values()).shape[0] == 200

        flt_chunk = SNPQualFilter(max_qual=1000)(chunk)[FLT_VARS]
        assert first(flt_chunk.values()).shape[0] == 92

        flt_chunk = SNPQualFilter(min_qual=530, max_qual=1000)(chunk)[FLT_VARS]
        assert first(flt_chunk.values()).shape[0] == 18

        flt_chunk = SNPQualFilter(min_qual=586325202)(chunk)[FLT_VARS]
        assert first(flt_chunk.values()).shape[0] == 0

        flt_chunk = SNPQualFilter(max_qual=-1)(chunk)[FLT_VARS]
        assert first(flt_chunk.values()).shape[0] == 0


class MissingGTSettersTest(unittest.TestCase):

    def test_set_gt_to_missing_by_dp(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = ['/calls/DP', GT_FIELD]
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)
        set_low_dp_gts_to_missing = LowDPGTsToMissingSetter(min_dp=300)
        res = set_low_dp_gts_to_missing(chunk)
        assert numpy.all(chunk[GT_FIELD][0][147] == [-1, -1])
        assert COUNTS in res

        set_low_dp_gts_to_missing(chunk)
        assert numpy.all(chunk[GT_FIELD].shape[0] == 200)

    def test_set_gt_to_missing_by_qual(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations[GT_FIELD] = gts
        variations[GQ_FIELD] = gqs
        set_low_qual_gts_to_missing = LowQualGTsToMissingSetter(min_qual=0)
        set_low_qual_gts_to_missing(variations)
        filtered = variations[GT_FIELD]
        assert numpy.all(filtered == gts)

        expected = numpy.array([[[0, 0], [1, 1], [-1, -1], [1, 1], [0, 0]],
                                [[0, 0], [-1, -1], [-1, -1], [0, 0],
                                 [-1, -1]]])
        set_low_qual_gts_to_missing = LowQualGTsToMissingSetter(min_qual=10)
        set_low_qual_gts_to_missing(variations)
        assert numpy.all(variations[GT_FIELD] == expected)

        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        gqs = numpy.array([[10, 20, 5, 20, 25],
                           [10, 2, 5, 15, 5]])
        variations[GT_FIELD] = gts
        variations[GQ_FIELD] = gqs
        set_low_qual_gts_to_missing(variations)
        assert numpy.all(variations[GT_FIELD] == expected)

        set_low_qual_gts_to_missing = LowQualGTsToMissingSetter(min_qual=100)
        set_low_qual_gts_to_missing(variations)
        assert numpy.all(variations[GT_FIELD] == -1)

        h5_1 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        set_low_qual_gts_to_missing = LowQualGTsToMissingSetter(min_qual=0)
        h5_2 = set_low_qual_gts_to_missing(h5_1)
        assert numpy.all(h5_1[GT_FIELD][:] == h5_2[FLT_VARS][GT_FIELD])


class MonoBiallelicFilterTest(unittest.TestCase):

    def test_filter_biallelic(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        kept_fields = [GT_FIELD]
        snps = hdf5.iterate_chunks(kept_fields=kept_fields)
        chunk = first(snps)

        flt_chunk = NonBiallelicFilter()(chunk)
        assert flt_chunk[FLT_VARS][GT_FIELD].shape == (174, 153, 2)
        assert flt_chunk[FLT_STATS][N_KEPT] == 174
        assert flt_chunk[FLT_STATS][TOT] == 200
        assert flt_chunk[FLT_STATS][N_FILTERED_OUT] == 26


class Chi2GtFilterTest(unittest.TestCase):

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
        flt = Chi2GtFreqs2SampleSetsFilter(samples1, samples2, min_pval=0.05,
                                           n_bins=2)
        res = flt(variations)
        assert list(res[COUNTS]) == [2, 2]
        assert numpy.all(res[FLT_VARS][GT_FIELD] == gts[:2, ...])
        assert res[FLT_STATS][N_KEPT] == 2
        assert res[FLT_STATS][TOT] == 4
        assert res[FLT_STATS][N_FILTERED_OUT] == 2


class FieldFilterTest(unittest.TestCase):
    def test_field_filter(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        orig_keys = hdf5.keys()

        varis = FieldFilter(kept_fields=[GT_FIELD])(hdf5)[FLT_VARS]
        assert [GT_FIELD] == list(varis.keys())

        varis = FieldFilter(ignored_fields=[GT_FIELD])(hdf5)[FLT_VARS]
        assert set(orig_keys).difference(varis.keys()) == set([GT_FIELD])


class FilterSamplesTest(unittest.TestCase):

    def test_filter_samples(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        samples = ['1_14_1_gbs', '1_17_1_gbs', '1_18_4_gbs']
        varis = SampleFilter(samples=samples)(hdf5)[FLT_VARS]
        # varis = filter_samples(hdf5, samples=samples)
        assert varis.samples == samples
        assert varis[GT_FIELD].shape == (943, 3, 2)
        assert varis[GQ_FIELD].shape == (943, 3)
        assert varis[CHROM_FIELD].shape == (943,)

        varis = SampleFilter(samples=samples, reverse=True)(hdf5)[FLT_VARS]
        assert all([sample not in samples for sample in varis.samples])
        n_samples = len(hdf5.samples)
        assert varis[GT_FIELD].shape == (943, n_samples - 3, 2)
        assert varis[GQ_FIELD].shape == (943, n_samples - 3)
        assert varis[CHROM_FIELD].shape == (943,)

    def test_filter_samples_by_missing(self):
        variations = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        chunk = first(variations.iterate_chunks())

        new_var = VariationsArrays()
        filter_samples_by_missing_rate(chunk, min_called_rate=0.9,
                                       out_vars=new_var)
        assert len(new_var.samples) == 0

        new_var = VariationsArrays()
        filter_samples_by_missing_rate(chunk, min_called_rate=0.1,
                                       out_vars=new_var)
        assert len(new_var.samples) == len(chunk.samples)

        # check that it works by chunk
        new_var = VariationsArrays()
        res = filter_samples_by_missing_rate(variations, min_called_rate=0.2,
                                             out_vars=new_var,
                                             do_histogram=True)
        new_var2 = VariationsArrays()
        res2 = filter_samples_by_missing_rate(variations, min_called_rate=0.2,
                                              out_vars=new_var2,
                                              chunk_size=None,
                                              do_histogram=True)

        assert res2['missing_rates'].shape[0] == len(variations.samples)
        assert res2['selected_samples'].shape[0] == len(variations.samples)
        assert new_var.samples == new_var2.samples
        assert numpy.all(new_var[GT_FIELD] == new_var2[GT_FIELD])
        assert numpy.allclose(res[EDGES], res2[EDGES])
        assert numpy.all(res[COUNTS][:] == res2[COUNTS][:])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'DepthFilterTest']
    unittest.main()
