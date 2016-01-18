
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest

import numpy

from variation.variations import VariationsArrays
from variation.variations.index import PosIndex


class IndexTest(unittest.TestCase):
    def test_index(self):
        snps = VariationsArrays()
        chroms = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        pos = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 4, 6]
        snps['/variations/chrom'] = numpy.array(chroms)
        snps['/variations/pos'] = numpy.array(pos)
        index = PosIndex(snps)
        assert index.index_pos(1, 1) == 0
        assert index.index_pos(2, 1) == 3
        assert index.index_pos(3, 1) == 6
        assert index.index_pos(4, 1) == 9
        assert index.index_pos(4, 2) == 9
        assert index.index_pos(4, 3) == 10
        assert index.index_pos(4, 4) == 10
        assert index.get_chrom_range(1) == (0, 2)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PlotTest.test_manhattan_plot']
    unittest.main()
