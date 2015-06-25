

import numpy

from variation.inout import _copy_chunk_with_new_data
from variation.stats import calc_mafs

# Missing docstring
# pylint: disable=C0111


def select_dset_chunks_for_field(dsets_chunks, field):
    for dsets_chunk in dsets_chunks:
        yield dsets_chunk[field]


def keep_only_data_from_dset_chunks(dset_chunks):
    return (dset_chunk.data for dset_chunk in dset_chunks)


def _filter_no_row(var_matrix):
    n_snps = var_matrix.data.shape[0]
    selector = numpy.ones((n_snps,), dtype=numpy.bool_)
    return selector


def _filter_all_rows(var_matrix):
    n_snps = var_matrix.data.shape[0]
    selector = numpy.zeros((n_snps,), dtype=numpy.bool_)
    return selector


def _filter_no_gts_in_dsets_chunk(dsets_chunk):
    gt_mat = dsets_chunk['GT']
    return _filter_no_row(gt_mat)


def _filter_all_gts_in_dsets_chunk(dsets_chunk):
    gt_mat = dsets_chunk['GT']
    return _filter_all_rows(gt_mat)


class DsetsChunksVarFilterByMaf():
    def __init__(self, min_maf=None, max_maf=None, min_num_genotypes=10):
        self.min_maf = min_maf
        self.max_maf = max_maf
        self.min_num_genotypes = min_num_genotypes

    def __call__(self, dsets_chunk):
        genotypes = dsets_chunk['GT']
        mafs = calc_mafs(genotypes.data,
                         min_num_genotypes=self.min_num_genotypes)
        ok_mafs1 = None if self.min_maf is None else mafs >  self.min_maf
        ok_mafs2 = None if self.max_maf is None else mafs < self.max_maf

        if ok_mafs1 is not None and ok_mafs2 is not None:
            ok_mafs = ok_mafs1 & ok_mafs2
        elif ok_mafs1 is not None:
            ok_mafs = ok_mafs1
        elif ok_mafs2 is not None:
            ok_mafs = ok_mafs2

        not_maf = numpy.isnan(mafs)
        ok_mafs = ok_mafs & ~not_maf
        return ok_mafs



def filter_dsets_chunks(selector_function, dsets_chunks):
    for dsets_chunk in dsets_chunks:
        bool_selection = selector_function(dsets_chunk)
        flt_dsets_chunk = {}
        for field, dset_chunk in dsets_chunk.items():
            flt_data = numpy.compress(bool_selection, dset_chunk.data, axis=0)
            flt_dsets_chunk[field] = _copy_chunk_with_new_data(dset_chunk,
                                                               flt_data)
        yield flt_dsets_chunk
