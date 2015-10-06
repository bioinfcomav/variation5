from array import array

CHUNK_SIZE = 200
MIN_NUM_GENOTYPES_FOR_POP_STAT = 10


class PosIndex():
    # Use tabix?
    def __init__(self, array):
        self.array = array
        self.dictionary = self._create_dict

    def index_pos(self, chrom, pos):
        first_chrom = self.dictionary[chrom][0]
        if pos < first_chrom:
            msg = 'The are not any chrom in this position.'
            msg += 'The first position in the chrom is '+first_chrom.astype(str)
            raise ValueError(msg)
        else:
            return self._bisect(self.dictionary[chrom], pos)

    @property
    def _create_dict(self):
        idx = {}
        snps = self.array
        n_snps = snps['/variations/chrom'].shape[0]
        for start in range(0, n_snps, CHUNK_SIZE):
            stop = start + CHUNK_SIZE
            if stop > n_snps:
                stop = n_snps
            array_chrom = snps['/variations/chrom'][start:stop]
            first_chrom = array_chrom[0]
            last_chrom = array_chrom[-1]
            #if the chunk has differents chroms
            if first_chrom != last_chrom:
                chroms = set(array_chrom)
                #create and fill the arrays per chrom
                while len(chroms) > 0:
                    chrom = chroms.pop()
                    is_chrom = array_chrom == chrom
                    chrom_positions = array('l')
                    chrom_positions.extend(snps['/variations/pos'][start:stop][is_chrom])
                    #if chrom is in the dictionary and is not the first chunk
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
            mid = (lo+hi)//2
            if chrom_positions[mid] < pos: lo = mid+1
            else: hi = mid
        return lo



