# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

import numpy

from variation.variations.merge import (_group_overlaping_vars,
                                        VarMerger, _sort_iterators,
                                        _get_overlapping_region,
                                        _pos_lt_tuples)
from variation.iterutils import PeekableIterator
from variation.variations.vars_matrices import VariationsH5, VariationsArrays

from test.test_utils import TEST_DATA_DIR
from variation.variations.stats import _remove_nans
from collections import Counter


class MockList(list):
    pass


class MockMerger(dict):
    def __init__(self, gt_shape):
        self._gt_shape = gt_shape
        self._gt_dtype = numpy.int32
        self.max_field_lens = {'alt': 3}
        self._n_samples1 = 2
        self._n_samples2 = 2
        self.log = Counter()

MockMerger._get_alleles = VarMerger._get_alleles
MockMerger._get_qual = VarMerger._get_qual
MockMerger._snps_are_mergeable = VarMerger._snps_are_mergeable


class MergeTest(unittest.TestCase):

    def test_sort_iterators(self):
        iter1 = [{'chrom': '1', 'pos': 1}, {'chrom': '1', 'pos': 5},
                 {'chrom': '1', 'pos': 7}, {'chrom': '1', 'pos': 10},
                 {'chrom': '2', 'pos': 11}, {'chrom': '2', 'pos': 12}]
        iter1 = PeekableIterator(iter(iter1))
        iter2 = [{'chrom': '1', 'pos': 0}, {'chrom': '1', 'pos': 3},
                 {'chrom': '1', 'pos': 6}, {'chrom': '2', 'pos': 5},
                 {'chrom': '2', 'pos': 11}, {'chrom': '2', 'pos': 13}]
        iter2 = PeekableIterator(iter(iter2))
        expected = [{'chrom': '1', 'pos': 0}, {'chrom': '1', 'pos': 1},
                    {'chrom': '1', 'pos': 3}, {'chrom': '1', 'pos': 5},
                    {'chrom': '1', 'pos': 6}, {'chrom': '1', 'pos': 7},
                    {'chrom': '1', 'pos': 10}, {'chrom': '2', 'pos': 5},
                    {'chrom': '2', 'pos': 11}, {'chrom': '2', 'pos': 11},
                    {'chrom': '2', 'pos': 12}, {'chrom': '2', 'pos': 13}]
        for x, y in zip(_sort_iterators(iter1, iter2), expected):
            assert x == y

    def test_group_overlaping_vars(self):
        vars1 = [{'chrom': '1', 'pos': 1, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '1', 'pos': 5, 'ref': 'AA', 'alt': 'T'},
                 {'chrom': '1', 'pos': 7, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '1', 'pos': 10, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '2', 'pos': 11, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '2', 'pos': 12, 'ref': 'A', 'alt': 'T'}]
        vars2 = [{'chrom': '1', 'pos': 0, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '1', 'pos': 3, 'ref': 'AAA', 'alt': 'T'},
                 {'chrom': '1', 'pos': 6, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '2', 'pos': 5, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '2', 'pos': 11, 'ref': 'A', 'alt': 'T'},
                 {'chrom': '2', 'pos': 13, 'ref': 'A', 'alt': 'T'}]
        exp = [(0, 1), (1, 0), (1, 2), (1, 0), (1, 0), (0, 1), (1, 1), (1, 0),
               (0, 1)]
        for (snps1, snps2), exp_ in zip(_group_overlaping_vars(vars1, vars2),
                                        exp):
            assert (len(snps1), len(snps2)) == exp_

    def test_pos_lt_tuples(self):
        assert _pos_lt_tuples(('1', 3), ('1', 4))
        assert _pos_lt_tuples(('1', 3), ('2', 1))
        assert not _pos_lt_tuples(('2', 3), ('1', 4))
        assert not _pos_lt_tuples(('2', 3), ('2', 3))

    def test_get_overlapping_region(self):
        sorted_vars = [{'chrom': '1', 'pos': 0, 'ref': 'A', 'alt': 'T'},
                       {'chrom': '2', 'pos': 1, 'ref': 'A', 'alt': 'T'}]
        region = _get_overlapping_region(sorted_vars)
        assert region == (('1', 0), ('1', 0))
        sorted_vars = [{'chrom': '1', 'pos': 0, 'ref': 'ATTT', 'alt': 'T'},
                       {'chrom': '2', 'pos': 1, 'ref': 'A', 'alt': 'T'}]
        region = _get_overlapping_region(sorted_vars)
        assert region == (('1', 0), ('1', 3))

        sorted_vars = [{'chrom': '1', 'pos': 0, 'ref': 'TAAA', 'alt': 'T'},
                       {'chrom': '1', 'pos': 1, 'ref': 'A', 'alt': 'T'},
                       {'chrom': '1', 'pos': 3, 'ref': 'A', 'alt': 'C'},
                       {'chrom': '1', 'pos': 5, 'ref': 'G', 'alt': 'C'}]
        region = _get_overlapping_region(sorted_vars)
        assert region == (('1', 0), ('1', 3))

        sorted_vars = [{'chrom': '1', 'pos': 0, 'ref': 'TAAA', 'alt': 'T'},
                       {'chrom': '1', 'pos': 1, 'ref': 'A', 'alt': 'T'},
                       {'chrom': '1', 'pos': 3, 'ref': 'A', 'alt': 'C'}]
        region = _get_overlapping_region(sorted_vars)
        assert region == (('1', 0), ('1', 3))

        sorted_vars = [{'chrom': '1', 'pos': 0, 'ref': 'TA', 'alt': 'T'},
                       {'chrom': '1', 'pos': 1, 'ref': 'AAA', 'alt': 'T'},
                       {'chrom': '1', 'pos': 3, 'ref': 'A', 'alt': 'C'}]
        region = _get_overlapping_region(sorted_vars)
        assert region == (('1', 0), ('1', 3))

    def var_is_equal(self, var1, var2):
        assert var1['chrom'] == var2['chrom']
        assert var1['pos'] == var2['pos']
        assert var1['ref'] == var2['ref']
        assert numpy.all(var1['alt'] == var2['alt'])
        assert var1['qual'] == var2['qual']
        assert numpy.all(var1['gts'] == var2['gts'])

    def test_merge_simple_var(self):

        vars1 = MockList([{'chrom': '1', 'pos': 1, 'ref': b'A', 'alt': [b'T'],
                           'gts': numpy.array([[0, 0], [1, 1]]), 'qual': 34}])
        vars2 = MockList([{'chrom': '1', 'pos': 1, 'ref': b'A', 'alt': [b'T'],
                           'gts': numpy.array([[0, 0], [1, 1]]), 'qual': 35}])
        vars1.samples = ['a', 'b']
        vars2.samples = ['c', 'd']
        merger = MockMerger(gt_shape=(4, 2))

        variation = VarMerger._merge_vars(merger, vars1[0], vars2[0])
#         merger = VarMerger(vars1, vars2)
#         variations = merger.variations
#         variation = next(variations)
        exp = {'gts': [[0, 0], [1, 1], [0, 0], [1, 1]], 'pos': 1,
               'ref': b'A', 'chrom': '1', 'alt': [b'T'], 'qual': 34}
        self.var_is_equal(exp, variation)

        vars1 = MockList([{'chrom': '1', 'pos': 1, 'ref': b'A', 'alt': [b'T'],
                           'gts': numpy.array([[0, 0], [1, 1]]), 'qual': 21}])
        vars2 = MockList([{'chrom': '1', 'pos': 2, 'ref': b'A', 'alt': [b'T'],
                           'gts': numpy.array([[0, 0], [1, 1]]), 'qual': 21}])
        vars1.samples = ['a', 'b']
        vars2.samples = ['c', 'd']

        variation = VarMerger._merge_vars(merger, vars1[0], None)
        exp = {'gts': [[0, 0], [1, 1], [-1, -1], [-1, -1]], 'pos': 1,
               'ref': b'A', 'chrom': '1', 'alt': [b'T'], 'qual': 21}
        self.var_is_equal(exp, variation)

        variation = VarMerger._merge_vars(merger, None, vars2[0])
        exp = {'gts': [[-1, -1], [-1, -1], [0, 0], [1, 1]], 'pos': 2,
               'ref': b'A', 'chrom': '1', 'alt': [b'T'], 'qual': 21}
        self.var_is_equal(exp, variation)

    def test_merge_complex_var(self):
        vars1 = MockList([{'chrom': '1', 'pos': 1, 'ref': b'ATT',
                           'alt': [b'T'], 'gts': numpy.array([[0, 0], [1, 1]]),
                           'qual': 21}])
        vars2 = MockList([{'chrom': '1', 'pos': 2, 'ref': b'T', 'alt': [b'A'],
                           'gts': numpy.array([[0, 0], [1, 1]]),
                           'qual': None}])
        vars1.samples = ['a', 'b']
        vars2.samples = ['c', 'd']
        merger = MockMerger(gt_shape=(4, 2))
        variation = VarMerger._merge_vars(merger, vars1[0], vars2[0])

        exp = {'gts': [[0, 0], [1, 1], [0, 0], [2, 2]], 'pos': 1,
               'ref': b'ATT', 'chrom': '1', 'alt': [b'T', b'AAT'],
               'qual': None}
        self.var_is_equal(exp, variation)

    def test_snos_are_mergeable(self):
        vars1 = MockList([{'chrom': '1', 'pos': 1, 'ref': b'ATT',
                           'alt': [b'T'],
                           'gts': numpy.array([[0, 0], [1, 1]])}])
        vars2 = MockList([{'chrom': '1', 'pos': 2, 'ref': b'TT', 'alt': [b'A'],
                           'gts': numpy.array([[0, 0], [1, 1]])}])
        vars1.samples = ['a', 'b']
        vars2.samples = ['c', 'd']

        merger = MockMerger((2, 2))

        assert not merger._snps_are_mergeable(vars1[0], vars2[0])

    def test_merge_variations(self):
        h5_1 = VariationsH5(join(TEST_DATA_DIR, 'csv', 'format.h5'), "r")
        h5_2 = VariationsH5(join(TEST_DATA_DIR, 'format_def.h5'), "r")
        merger = VarMerger(h5_1, h5_2, max_field_lens={'alt': 3},
                           ignore_complex_overlaps=True,
                           check_ref_matches=False)
        assert merger.ploidy == 2
        assert merger.samples == [b'TS-1', b'TS-11', b'TS-21', b'NA00001',
                                  b'NA00002', b'NA00003']
        expected_h5 = VariationsH5(join(TEST_DATA_DIR, 'expected_merged.h5'),
                                   'r')
        new_vars = VariationsArrays()
        new_vars.put_vars(merger)
        for field in new_vars.keys():
            if 'float' in str(new_vars[field][:].dtype):
                assert numpy.all(_remove_nans(expected_h5[field][:]) ==
                                 _remove_nans(new_vars[field][:]))
            else:
                result = new_vars[field][:]
                assert numpy.all(expected_h5[field][:] == result)

        # Change the order
        h5_1 = VariationsH5(join(TEST_DATA_DIR, 'csv', 'format.h5'), "r")
        h5_2 = VariationsH5(join(TEST_DATA_DIR, 'format_def.h5'), "r")
        merger = VarMerger(h5_2, h5_1, max_field_lens={'alt': 3},
                           ignore_complex_overlaps=True,
                           check_ref_matches=False)
        assert merger.ploidy == 2
        assert merger.samples == [b'NA00001', b'NA00002', b'NA00003',
                                  b'TS-1', b'TS-11', b'TS-21']
        expected_h5 = VariationsH5(join(TEST_DATA_DIR, 'expected_merged2.h5'),
                                   'r')
        new_vars = VariationsArrays()
        new_vars.put_vars(merger)
        for field in new_vars.keys():
            if 'float' in str(new_vars[field][:].dtype):
                assert numpy.all(_remove_nans(expected_h5[field][:]) ==
                                 _remove_nans(new_vars[field][:]))
            else:
                result = new_vars[field][:]
                assert numpy.all(expected_h5[field][:] == result)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'MergeTest.test_merge_variations']
    unittest.main()
