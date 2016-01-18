
from collections import OrderedDict
from bisect import bisect_left, bisect_right

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
        chrom_locs = self._index[chrom]
        return chrom_locs['start'], chrom_locs['end'] - 1

    def get_chrom_range_pos(self, chrom):
        chrom_locs = self._index[chrom]
        pos_in_chroms = self.variations[POS_FIELD]
        return (pos_in_chroms[chrom_locs['start']],
                pos_in_chroms[chrom_locs['end'] - 1])

    def index_pos(self, chrom, pos):
        return self._bisect(self.variations[POS_FIELD], pos,
                            lo=self._index[chrom]['start'],
                            hi=self._index[chrom]['end'])

    def _create_dict(self):
        idx = OrderedDict()
        snps = self.variations
        for chrom in numpy.unique(snps[CHROM_FIELD]):
            start = bisect_left(snps[CHROM_FIELD], chrom)
            end = bisect_right(snps[CHROM_FIELD], chrom)
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
