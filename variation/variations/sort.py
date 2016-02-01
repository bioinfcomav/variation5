
import itertools
from array import array

import numpy

from variation.matrix.methods import is_dataset


def _calc_sort_order(variations):
    chrom = variations['/variations/chrom']
    pos = variations['/variations/pos']
    idx_order = numpy.lexsort((pos, chrom))
    return idx_order


def _calc_sort_order_by_chrom(variations):
    chrom = variations['/variations/chrom']
    if is_dataset(chrom):
        chrom = chrom[:]
    pos = variations['/variations/pos']
    chrom_names = numpy.sort(numpy.unique(chrom))
    for chrom_name in chrom_names:
        mask = chrom == chrom_name
        snps_in_chrom_idx = numpy.where(mask)[0]
        pos_chrom = pos[mask]
        sorted_idx = numpy.lexsort((pos_chrom,), axis=0)
        sorted_snps_in_chrom_idx = snps_in_chrom_idx[sorted_idx]
        yield sorted_snps_in_chrom_idx


def _sorted_chunks(n_snps, chunk_size, idx_order, variations):
    for _ in range(0, n_snps, chunk_size):
        idx = array('L', itertools.islice(idx_order, chunk_size))
        yield variations.get_chunk(idx)


def sort_variations(variations, output_variations):
    if True:
        idx_order = itertools.chain(*_calc_sort_order_by_chrom(variations))
        n_snps = variations.num_variations
        chunk_size = variations._vars_in_chunk
        chunks = _sorted_chunks(n_snps, chunk_size, idx_order, variations)
        output_variations.put_chunks(chunks)
    else:
        # This algorithm sorts all snps at the same time, it does not
        # iterate over the chromosomes
        # We haven't tested if it is faster or slower than iterating over the
        # chromosomes. What we do know is that iterating over the chromosomes
        # should take less memory
        idx_order = _calc_sort_order(variations)
        n_snps = variations.num_variations
        chunk_size = variations._vars_in_chunk
        chunks = (variations.get_chunk(idx_order[i: i + chunk_size])
                  for i in range(0, n_snps, chunk_size))
        output_variations.put_chunks(chunks)
