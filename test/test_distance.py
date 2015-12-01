
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest

import numpy
from scipy.spatial.distance import pdist

from variation.variations.distance import indi_pairwise_dist, _kosman


class IndividualDistTest(unittest.TestCase):
    def test_kosman_2_indis(self):
        a = numpy.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = numpy.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 2]])
        distance = _kosman(a, b)
        assert distance == 1/3

        c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
        distance = _kosman(c, d)
        assert distance == 0
        
        distance = _kosman(b, d)
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
        distance = indi_pairwise_dist(gts)
        expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.]
        assert numpy.allclose(distance, expected)

if __name__ == "__main__":
#     import sys;sys.argv = ['', 'FilterTest.test_filter_missing_varArray']
    unittest.main()
