
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join
import math

import numpy
from scipy.spatial.distance import squareform

from variation.variations.distance import (_kosman,
                                           _IndiPairwiseCalculator,
                                           calc_pairwise_distance,
                                           sel_samples_from_dist_mat,
                                           _matching, calc_pop_distance,
                                           filter_dist_matrix,
                                           calc_gst_per_loci,
                                           _calc_pop_pairwise_unbiased_nei_dists,
                                           triangular_dists_to_square,
                                           locate_cols_and_rows_with_nan_values_in_dist_matrix,
                                           calc_pairwise_distances_between_pops)
from variation.variations.vars_matrices import VariationsArrays, VariationsH5
from variation.variations.stats import GT_FIELD
from test.test_utils import TEST_DATA_DIR
from variation.iterutils import first
from variation import AD_FIELD


class IndividualDistTest(unittest.TestCase):

    def test_kosman_2_indis(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = numpy.stack((a, b), axis=1)
        abs_distance, n_snps = _kosman(gts, 0, 1, {})
        distance = abs_distance / n_snps
        assert distance == 1 / 3

        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        gts = numpy.stack((c, d), axis=1)
        abs_distance, n_snps = _kosman(gts, 0, 1, {})
        distance = abs_distance / n_snps
        assert distance == 0

        gts = numpy.stack((b, d), axis=1)
        abs_distance, n_snps = _kosman(gts, 0, 1, {})
        distance = abs_distance / n_snps
        assert distance == 0.45

    def test_kosman_missing(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = numpy.stack((a, b), axis=1)
        abs_distance, n_snps = _kosman(gts, 0, 1, {})
        distance_ab = abs_distance / n_snps

        a = numpy.array([[-1, -1], [-1, -1], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[-1, -1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = numpy.stack((a, b), axis=1)
        abs_distance, n_snps = _kosman(gts, 0, 1, {})
        distance_cd = abs_distance / n_snps

        assert distance_ab == distance_cd

    def test_kosman_pairwise(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 2]])
        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        gts = numpy.stack((a, b, c, d), axis=0)
        gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        pairwise_dist_calculator = _IndiPairwiseCalculator()
        abs_dist, n_snps = pairwise_dist_calculator.calc_dist(varis,
                                                              method='kosman')
        distance = abs_dist / n_snps
        expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.]
        assert numpy.allclose(distance, expected)

    def test_kosman_pairwise_by_chunk(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        gts = numpy.stack((a, b, c, d), axis=0)
        gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
        variations = VariationsArrays()
        variations['/calls/GT'] = gts
        expected = [0.33333333, 0.75, 0.75, 0.45, 0.45, 0.]
        distance = calc_pairwise_distance(variations, chunk_size=None,
                                          min_num_snps=1)
        assert numpy.allclose(distance, expected)

        distance = calc_pairwise_distance(variations, chunk_size=2)
        assert numpy.allclose(distance, expected)

        distance = calc_pairwise_distance(variations, chunk_size=None,
                                          min_num_snps=11)
        assert numpy.sum(numpy.isnan(distance)) == 5

        # With all missing
        a = numpy.full(shape=(10, 2), fill_value=-1, dtype=numpy.int16)
        b = numpy.full(shape=(10, 2), fill_value=-1, dtype=numpy.int16)
        gts = numpy.stack((a, b), axis=0)
        gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
        variations = VariationsArrays()
        variations['/calls/GT'] = gts
        distance = calc_pairwise_distance(variations)
        assert numpy.isnan(distance[0])

        # With missing in some chunks only
        variations['/calls/GT'][:5, 0, :] = 1
        variations['/calls/GT'][:5, 1, :] = 0
        assert calc_pairwise_distance(variations)[0] == 1
        assert calc_pairwise_distance(variations, chunk_size=3)[0] == 1

    def test_kosman_pairwise_between_pops_by_chunk(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        gts = numpy.stack((a, b, c, d), axis=0)
        gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
        variations = VariationsArrays()
        variations.samples = [1, 2, 3, 4]
        variations['/calls/GT'] = gts
        expected = [[0., 0.33333333, 0.75, 0.75],
                    [0.33333333, 0., 0.45, 0.45],
                    [0.75, 0.45, 0., 0.],
                    [0.75, 0.45, 0., 0.]]
        distance = calc_pairwise_distances_between_pops(variations,
                                                        chunk_size=None,
                                                        min_num_snps=1,
                                                        pop1_samples=[1, 2, 3, 4],
                                                        pop2_samples=[1, 2, 3, 4])
        assert numpy.allclose(distance, expected)

        expected = [[0., 0.33333333, 0.75, 0.75]]
        distance = calc_pairwise_distances_between_pops(variations,
                                                        chunk_size=None,
                                                        min_num_snps=1,
                                                        pop1_samples=[1],
                                                        pop2_samples=[1, 2, 3, 4])
        assert numpy.allclose(distance, expected)

        expected = [[0.75, 0.75],
                    [0.45, 0.45]]
        distance = calc_pairwise_distances_between_pops(variations,
                                                        chunk_size=None,
                                                        min_num_snps=1,
                                                        pop1_samples=[1, 2],
                                                        pop2_samples=[3, 4])

    def test_select_samples_from_distance_matrix(self):
        distances = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.]
        sel_samples = [0, 1, 3]
        expected = [0.33333333, 0.75, 0.5]
        selected_distances = sel_samples_from_dist_mat(distances,
                                                       sel_samples)
        assert numpy.all(selected_distances == expected)

    def test_matching_2_indis(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = numpy.stack((a, b), axis=1)
        distance = _matching(gts, 0, 1, {})
        assert distance == 1 - 4 / 9

        f = numpy.array([[0, 0], [0, 1], [0, 0], [0, 0]])
        g = numpy.array([[0, 0], [1, 0], [1, 1], [1, 0]])
        gts = numpy.stack((f, g), axis=1)
        distance = _matching(gts, 0, 1, {})
        assert distance == 0.5

        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        gts = numpy.stack((c, d), axis=1)
        distance = _matching(gts, 0, 1, {})
        assert distance == 0

        e = numpy.full(shape=(11, 2), fill_value=0, dtype=numpy.int16)
        gts = numpy.stack((c, e), axis=1)
        distance = _matching(gts, 0, 1, {})
        assert distance == 1

    def test_matching_pairwise_by_chunk(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        gts = numpy.stack((a, b, c, d), axis=0)
        gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
        variations = VariationsArrays()
        variations['/calls/GT'] = gts
        expected = [0.444444, 0, 0, 0.3, 0.3, 1]
        distance = calc_pairwise_distance(variations, chunk_size=None,
                                          method='matching')
        assert numpy.allclose(distance, expected)

        distance = calc_pairwise_distance(variations, chunk_size=2,
                                          method='matching')
        assert numpy.allclose(distance, expected)

        # With all missing
        a = numpy.full(shape=(10, 2), fill_value=-1, dtype=numpy.int16)
        b = numpy.full(shape=(10, 2), fill_value=-1, dtype=numpy.int16)
        gts = numpy.stack((a, b), axis=0)
        gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
        variations = VariationsArrays()
        variations['/calls/GT'] = gts
        distance = calc_pairwise_distance(variations, method='matching')
        assert numpy.isnan(distance[0])


class PopDistTest(unittest.TestCase):

    def test_nei_dist(self):
        gts = [[[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 0]],
               [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]],
               [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]]]
        snps = VariationsArrays()
        snps['/calls/GT'] = numpy.array(gts)
        snps.samples = [1, 2, 3, 4, 5, 6, 7]

        pops = [[1, 2, 3], [4, 5, 6, 7]]
        dists = calc_pop_distance(snps, populations=pops, method='nei',
                                  min_num_genotypes=0)
        assert dists[0] - 3.14019792 < 0.001
        pops = [[1, 2, 3], [1, 2, 3]]
        dists = calc_pop_distance(snps, populations=pops, method='nei',
                                  min_num_genotypes=0)
        assert dists[0] - 0 < 0.001
        pops = [[1, 2, 3], [1, 4, 5, 6, 7]]
        dists = calc_pop_distance(snps, populations=pops, method='nei',
                                  min_num_genotypes=0)
        assert dists[0] - 1.23732507 < 0.001

        # by chunk
        pops = [[1, 2, 3], [4, 5, 6, 7]]
        dists = calc_pop_distance(snps, populations=pops, method='nei',
                                  chunk_size=2, min_num_genotypes=0)
        assert dists[0] - 3.14019792 < 0.001
        pops = [[1, 2, 3], [1, 2, 3]]
        dists = calc_pop_distance(snps, populations=pops, method='nei',
                                  chunk_size=2, min_num_genotypes=0)
        assert dists[0] - 0 < 0.001
        pops = [[1, 2, 3], [1, 4, 5, 6, 7]]
        dists = calc_pop_distance(snps, populations=pops, method='nei',
                                  chunk_size=2, min_num_genotypes=0)
        assert dists[0] - 1.23732507 < 0.001


class NeiUnbiasedDistTest(unittest.TestCase):

    def test_nei_dist(self):

        gts = numpy.array([[[1, 1], [5, 2], [2, 2], [3, 2]],
                           [[1, 1], [1, 2], [2, 2], [2, 1]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4]
        pops = [[1, 2], [3, 4]]
        dists = _calc_pop_pairwise_unbiased_nei_dists(varis,
                                                      populations=pops,
                                                      min_num_genotypes=1)
        assert math.isclose(dists[0], 0.3726315908494797)

        dists = _calc_pop_pairwise_unbiased_nei_dists(varis,
                                                      populations=pops,
                                                      min_num_genotypes=1,
                                                      chunk_size=1)
        assert math.isclose(dists[0], 0.3726315908494797)

        # all missing
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4]
        pops = [[1, 2], [3, 4]]
        dists = _calc_pop_pairwise_unbiased_nei_dists(varis,
                                                      populations=pops,
                                                      min_num_genotypes=1)
        assert math.isnan(dists[0])

        # min_num_genotypes
        gts = numpy.array([[[1, 1], [5, 2], [2, 2], [3, 2]],
                           [[1, 1], [1, 2], [2, 2], [2, 1]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4]
        pops = [[1, 2], [3, 4]]
        dists = _calc_pop_pairwise_unbiased_nei_dists(varis,
                                                      populations=pops,
                                                      min_num_genotypes=1)
        assert math.isclose(dists[0], 0.3726315908494797)

        dists = _calc_pop_pairwise_unbiased_nei_dists(varis,
                                                      populations=pops,
                                                      chunk_size=1)
        assert math.isnan(dists[0])


class DJostDistTest(unittest.TestCase):
    def test_dest_jost_distance(self):

        gts = [[(1, 1), (1, 3), (1, 2), (1, 4), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [(1, 3), (1, 1), (1, 1), (1, 3), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)]]
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        pops = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]

        snps = VariationsArrays()
        snps['/calls/GT'] = numpy.array(gts)
        snps.samples = samples

        dists = calc_pop_distance(snps, populations=pops, method='dest',
                                  min_num_genotypes=0)
        assert numpy.allclose(dists, [0.65490196])

        dists = calc_pop_distance(snps, populations=pops, method='dest',
                                  min_num_genotypes=0, chunk_size=1)
        assert numpy.allclose(dists, [0.65490196])

        dists = calc_pop_distance(snps, populations=pops, method='dest',
                                  min_num_genotypes=6, chunk_size=1)
        assert numpy.all(numpy.isnan(dists))

    def test_empty_pop(self):
        missing = (-1, -1)
        gts = [[(1, 1), (1, 3), (1, 2), (1, 4), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [(1, 3), (1, 1), (1, 1), (1, 3), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               ]
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        pops = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]

        snps = VariationsArrays()
        snps['/calls/GT'] = numpy.array(gts)
        snps.samples = samples

        dists = calc_pop_distance(snps, populations=pops, method='dest',
                                  min_num_genotypes=0)
        assert numpy.allclose(dists, [0.65490196])

        gts = [[missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               ]
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        pops = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]

        snps = VariationsArrays()
        snps['/calls/GT'] = numpy.array(gts)
        snps.samples = samples

        dists = calc_pop_distance(snps, populations=pops, method='dest',
                                  min_num_genotypes=0)
        assert numpy.isnan(dists[0])


class FilterDistTest(unittest.TestCase):
    def test_filter_dists(self):
        dist = [[0, 1, 2],
                [1, 0, 3],
                [2, 3, 0]]

        dist = squareform(numpy.array(dist))
        assert numpy.allclose(filter_dist_matrix(dist, [0, 1]), [1])
        assert numpy.allclose(filter_dist_matrix(dist, [1, 2]), [3])


class GstDistTest(unittest.TestCase):
    def test_gst_basic(self):
        ad = [[[10, 3, -1], [11, 2, -1]],
              [[10, 0, -1], [10, 0, -1]],
              [[10, 10, -1], [11, 11, -1]],
              [[-1, 2, 10], [-1, 10, 2]]]

        snps = VariationsArrays()
        snps.samples = [1, 2]
        populations = [[1], [2]]
        snps[AD_FIELD] = numpy.array(ad)
        dist = calc_gst_per_loci(snps, populations)
        expected = numpy.array([0.00952381, 0, 0, 0.44444444])
        numpy.testing.assert_almost_equal(dist, expected)

    def test_gst(self):
        h5 = VariationsH5(join(TEST_DATA_DIR, 'limon.h5'), mode='r')
        # flt = SampleFilter(['V51'])
        # v51 = flt(h5)[FLT_VARS]
        chunk = first(h5.iterate_chunks())

        dists = calc_gst_per_loci(chunk, populations=[['V51'], ['F49']])
        assert dists[0] == 0


class TriangularToSquareTest(unittest.TestCase):

    def test_trinagular_to_square(self):
        dists = [1., 2., 3.]
        names = ['a', 'b', 'c']
        square_dists = triangular_dists_to_square(dists, names)
        assert math.isclose(square_dists['a']['b'], 1.)
        assert math.isclose(square_dists['a']['c'], 2.)
        assert math.isclose(square_dists['b']['c'], 3.)

        assert numpy.allclose(square_dists.values, [[0, 1, 2],
                                                    [1, 0, 3],
                                                    [2, 3, 0]])


class TestLocateNans(unittest.TestCase):

    def test_locate_cols_and_rows_with_nans(self):
        dists = numpy.array([1., 2., 3., 4., 5., math.nan])
        cols_with_nans = locate_cols_and_rows_with_nan_values_in_dist_matrix(dists)
        assert list(cols_with_nans) == [False, False, True, True]

        dists = numpy.array([1., 2., 3., 4., 5., 6.])
        cols_with_nans = locate_cols_and_rows_with_nan_values_in_dist_matrix(dists)
        assert list(cols_with_nans) == [False, False, False, False]


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'DJostDistTest.test_empty_pop']
    unittest.main()
