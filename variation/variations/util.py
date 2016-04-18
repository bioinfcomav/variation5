
import numpy

from variation import DP_FIELD


def add_mock_depth(variations, depth):
    shape = (variations.num_variations, len(variations.samples))
    depths = numpy.full(shape, depth, dtype=numpy.int8)
    variations[DP_FIELD] = depths
