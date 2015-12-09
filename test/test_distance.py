
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest

import numpy

from variation.variations.distance import (_indi_pairwise_dist, _kosman,
                                           calc_pairwise_distance)
from variation.variations.vars_matrices import VariationsArrays


class IndividualDistTest(unittest.TestCase):
    def test_kosman_2_indis(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 2]])
        abs_distance, n_snps = _kosman(a, b)
        distance = abs_distance / n_snps
        assert distance == 1/3

        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        abs_distance, n_snps = _kosman(c, d)
        distance = abs_distance / n_snps
        assert distance == 0

        abs_distance, n_snps = _kosman(b, d)
        distance = abs_distance / n_snps
        assert distance == 0.5

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
        abs_distance, n_snps = _indi_pairwise_dist(gts)
        distance = abs_distance / n_snps
        expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.]
        assert numpy.allclose(distance, expected)

    def test_kosman_pairwise_by_chunk(self):
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
        variations = VariationsArrays()
        variations['/calls/GT'] = gts
        expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.]
        distance = calc_pairwise_distance(variations, chunk_size=2)
        assert numpy.allclose(distance, expected)
        distance = calc_pairwise_distance(variations, chunk_size=None)
        assert numpy.allclose(distance, expected)

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
        variations['/calls/GT'][:5,0,:] = 1
        variations['/calls/GT'][:5,1,:] = 0
        assert calc_pairwise_distance(variations)[0] == 1
        assert calc_pairwise_distance(variations, chunk_size=3)[0] == 1


if __name__ == "__main__":
#     import sys;sys.argv = ['', 'IndividualDistTest.test_kosman_pairwise_by_chunk']
    unittest.main()
