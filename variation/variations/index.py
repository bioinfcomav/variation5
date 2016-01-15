from array import array
from collections import OrderedDict

CHUNK_SIZE = 200


class PosIndex():
    def __init__(self, array):
        self.array = array
        self._index = self._create_dict()

    @property
    def chroms(self):
        return iter(self._index.keys())

    def get_chrom_range(self, chrom):
        chrom_locs = self._index[chrom]
        return chrom_locs[0], chrom_locs[-1]

    def index_pos(self, chrom, pos):
        return self._bisect(self._index[chrom], pos)

    def _create_dict(self):
        idx = OrderedDict()
        snps = self.array
        n_snps = snps['/variations/chrom'].shape[0]
        for start in range(0, n_snps, CHUNK_SIZE):
            stop = start + CHUNK_SIZE
            if stop > n_snps:
                stop = n_snps
            array_chrom = snps['/variations/chrom'][start:stop]
            first_chrom = array_chrom[0]
            last_chrom = array_chrom[-1]
            # if the chunk has different chroms
            if first_chrom != last_chrom:
                chroms = set(array_chrom)
                # create and fill the arrays per chrom
                while len(chroms) > 0:
                    chrom = chroms.pop()
                    is_chrom = array_chrom == chrom
                    chrom_positions = array('l')
                    chrom_positions.extend(snps['/variations/pos'][start:stop][is_chrom])
                    # if chrom is in the dictionary and is not the first chunk
                    try:
                        idx[chrom].extend(chrom_positions)
                    except KeyError:
                        idx[chrom] = chrom_positions
            else:
                chrom_positions = array('l')
                chrom_positions.extend(snps['/variations/pos'][start:stop])
                if first_chrom in idx:
                    idx[first_chrom].extend(chrom_positions)
                else:
                    idx[first_chrom] = chrom_positions
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
