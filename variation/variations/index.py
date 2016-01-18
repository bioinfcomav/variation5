
from collections import OrderedDict
from bisect import bisect_left, bisect_right

import numpy


class PosIndex():
    def __init__(self, variations):
        self.variations = variations
        self._index = self._create_dict()

    @property
    def chroms(self):
        return iter(self._index.keys())

    def get_chrom_range(self, chrom):
        chrom_locs = self._index[chrom]
        return chrom_locs['start'], chrom_locs['end'] - 1

    def index_pos(self, chrom, pos):
        return self._bisect(self.variations['/variations/pos'], pos,
                            lo=self._index[chrom]['start'],
                            hi=self._index[chrom]['end'])

    def _create_dict(self):
        idx = OrderedDict()
        snps = self.variations
        for chrom in numpy.unique(snps['/variations/chrom']):
            start = bisect_left(snps['/variations/chrom'], chrom)
            end = bisect_right(snps['/variations/chrom'], chrom)
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
