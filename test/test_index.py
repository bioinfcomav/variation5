
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest

import numpy

from variation.variations import VariationsArrays
from variation.variations.index import (PosIndex, var_bisect_right,
                                        var_bisect_left, index,
                                        find_le, find_ge)
from variation import CHROM_FIELD, POS_FIELD


class IndexTest(unittest.TestCase):
    def test_index(self):
        snps = VariationsArrays()
        chroms = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        pos = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 4, 6]
        snps[CHROM_FIELD] = numpy.array(chroms)
        snps[POS_FIELD] = numpy.array(pos)

        index = PosIndex(snps)
        assert index.index_pos(1, 1) == 0
        assert index.index_pos(2, 1) == 3
        assert index.index_pos(3, 1) == 6
        assert index.index_pos(4, 1) == 9
        assert index.index_pos(4, 2) == 9
        assert index.index_pos(4, 3) == 10
        assert index.index_pos(4, 4) == 10
        assert index.get_chrom_range_index(1) == (0, 2)
        assert index.get_chrom_range_pos(1) == (1, 3)
        assert index.covered_length == 10

    def test_find(self):

        snps = VariationsArrays()
        chroms = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        pos = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 4, 6]
        snps[CHROM_FIELD] = numpy.array(chroms)
        snps[POS_FIELD] = numpy.array(pos)

        # find_le

        # find_re

    def test_bisect(self):

        snps = VariationsArrays()
        chroms = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        pos = [1, 3, 5, 1, 2, 3, 1, 2, 3, 2, 4, 6]
        snps[CHROM_FIELD] = numpy.array(chroms)
        snps[POS_FIELD] = numpy.array(pos)

        # bisect right
        assert var_bisect_right(snps, 1, 0) == 0
        assert var_bisect_right(snps, 1, 1) == 1
        assert var_bisect_right(snps, 1, 3) == 2
        assert var_bisect_right(snps, 1, 5) == 3
        assert var_bisect_right(snps, 2, 1) == 4
        assert var_bisect_right(snps, 4, 7) == 12

        assert var_bisect_left(snps, 1, 0) == 0
        assert var_bisect_left(snps, 1, 1) == 0
        assert var_bisect_left(snps, 1, 3) == 1
        assert var_bisect_left(snps, 1, 5) == 2
        assert var_bisect_left(snps, 2, 1) == 3
        assert var_bisect_left(snps, 4, 7) == 12


        # find_re
        try:
            find_le(snps, 1, 0)
            self.fail('Value Error expected')
        except ValueError:
            pass

        assert find_le(snps, 1, 1) == 0
        assert find_le(snps, 1, 2) == 0
        assert find_le(snps, 1, 3) == 1
        assert find_le(snps, 1, 7) == 2
        assert find_le(snps, 2, 1) == 3
        assert find_le(snps, 2, 2) == 4
        assert find_le(snps, 4, 7) == 11

        assert find_ge(snps, 1, 0) == 0
        assert find_ge(snps, 1, 1) == 0
        assert find_ge(snps, 1, 2) == 1
        assert find_ge(snps, 1, 7) == 3
        assert find_ge(snps, 2, 1) == 3
        assert find_ge(snps, 2, 2) == 4
        try:
            find_ge(snps, 4, 7)
            self.fail('Value Error expected')
        except ValueError:
            pass

        assert index(snps, 1, 1) == 0
        try:
            index(snps, 0, 7)
            self.fail('Index Error expected')
        except IndexError:
            pass
        try:
            index(snps, 4, 3)
            self.fail('Index Error expected')
        except IndexError:
            pass
        assert index(snps, 4, 2) == 9

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PlotTest.test_manhattan_plot']
    unittest.main()
