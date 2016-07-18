
from collections import OrderedDict
from bisect import bisect_left, bisect_right
from functools import total_ordering

import numpy

from variation import POS_FIELD, CHROM_FIELD


class PosIndex():
    def __init__(self, variations):
        self.variations = variations
        self._index = self._create_dict()

    @property
    def chroms(self):
        return iter(self._index.keys())

    def get_chrom_range_index(self, chrom):
        try:
            chrom_locs = self._index[chrom]
        except IndexError:
            raise IndexError('No snps for chrom: ' + str(chrom))

        return chrom_locs['start'], chrom_locs['end'] - 1

    def get_chrom_range_pos(self, chrom):
        if chrom not in self._index:
            raise IndexError('No snps for chrom: ' + str(chrom))
        chrom_locs = self._index[chrom]
        pos_in_chroms = self.variations[POS_FIELD]
        return (pos_in_chroms[chrom_locs['start']],
                pos_in_chroms[chrom_locs['end'] - 1])

    @property
    def covered_length(self):
        pos = self.variations[POS_FIELD]
        length = 0
        for chrom_locs in self._index.values():
            start = pos[chrom_locs['start']]
            end = pos[chrom_locs['end'] - 1]
            length += end - start
        return length

    def index_pos(self, chrom, pos):
        return self._bisect(self.variations[POS_FIELD], pos,
                            lo=self._index[chrom]['start'],
                            hi=self._index[chrom]['end'])

    def _create_dict(self):
        idx = OrderedDict()
        snps = self.variations
        chrom_mat = snps[CHROM_FIELD]
        for chrom in numpy.unique(chrom_mat):
            start = bisect_left(chrom_mat, chrom)
            end = bisect_right(chrom_mat, chrom)
            if chrom_mat[start] != chrom:
                raise RuntimeError('Maybe SNPs are not sorted')
            idx[chrom] = {'start': start, 'end': end}
        return idx

    def _bisect(self, chrom_positions, pos, lo=0, hi=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(chrom_positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if chrom_positions[mid] < pos:
                lo = mid + 1
            else:
                hi = mid
        return lo


def var_bisect_right(variations, chrom, pos, lo=0, hi=None):

    chroms = variations[CHROM_FIELD]
    poss = variations[POS_FIELD]

    if lo < 0:
        raise ValueError('lo must be equal or bigger than 0')

    if hi is None:
        hi = chroms.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if (chrom < chroms[mid]) or (chrom == chroms[mid] and pos < poss[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def var_bisect_left(variations, chrom, pos, lo=0, hi=None):

    chroms = variations[CHROM_FIELD]
    poss = variations[POS_FIELD]

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = chroms.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2

        if chroms[mid] < chrom or (chroms[mid] == chrom and poss[mid] < pos):
            lo = mid + 1
        else:
            hi = mid

    return lo


def find_le(variations, chrom, pos):
    'Find rightmost value less than or equal to x'
    idx = var_bisect_right(variations, chrom, pos)
    if idx:
        return idx - 1
    raise ValueError


def find_ge(variations, chrom, pos):
    'Find leftmost item greater than or equal to x'
    idx = var_bisect_left(variations, chrom, pos)
    if idx != variations.num_variations:
        return idx
    raise ValueError


def _raise_index_error(chrom, pos):
    msg = 'chrom and pos not found in variations : '
    msg += str(chrom) + ' ' + str(pos)
    raise IndexError(msg)


def index(variations, chrom, pos):
    try:
        idx = find_le(variations, chrom, pos)
    except ValueError:
        _raise_index_error(chrom, pos)
    # print(variations[CHROM_FIELD][idx], variations[POS_FIELD][idx])
    if (chrom != variations[CHROM_FIELD][idx] or
       variations[POS_FIELD][idx] != pos):
        _raise_index_error(chrom, pos)
    return idx
