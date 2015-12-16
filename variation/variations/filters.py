import numpy
from _functools import partial

from variation.variations.stats import (calc_maf, calc_obs_het, GT_FIELD,
                                        calc_called_gt, GQ_FIELD, DP_FIELD,
                                        MIN_NUM_GENOTYPES_FOR_POP_STAT)
from variation.variations.vars_matrices import VariationsArrays
from variation import MISSING_VALUES


def _filter_dsets_chunks(selector_function, dsets_chunks):
    for dsets_chunk in dsets_chunks:
        bool_selection = selector_function(dsets_chunk)
        flt_dsets_chunk = {}
        for field, dset_chunk in dsets_chunk.items():
            flt_data = numpy.compress(bool_selection, dset_chunk.data, axis=0)
            flt_dsets_chunk[field] = flt_data
        yield flt_dsets_chunk


def _filter_all_rows(chunk):
    n_snps = chunk.num_variations
    selector = numpy.zeros((n_snps,), dtype=numpy.bool_)
    return selector


def _filter_no_row(chunk):
    n_snps = chunk.num_variations
    selector = numpy.ones((n_snps,), dtype=numpy.bool_)
    return selector


def _filter_chunk(chunk, fields, filter_funct):
    selected_rows = filter_funct(chunk)
    return _filter_chunk2(chunk, selected_rows)


def _filter_chunk2(chunk, selected_rows):
    flt_chunk = VariationsArrays()
    for path in chunk.keys():
        matrix = chunk[path]
        try:
            array = matrix.data
        except:
            array = matrix
        flt_data = numpy.compress(selected_rows, array, axis=0)
        flt_chunk[path] = flt_data
    flt_chunk.metadata = chunk.metadata
    return flt_chunk


def _filter_chunk_samples(chunk, selected_cols):
    flt_chunk = VariationsArrays()
    for path in chunk.keys():
        if 'calls' in path:
            matrix = chunk[path]
            try:
                array = matrix.data
            except:
                array = matrix
            flt_data = numpy.compress(selected_cols, array, axis=1)
            flt_chunk[path] = flt_data
        else:
            flt_chunk[path] = matrix
    flt_chunk.metadata = chunk.metadata
    return flt_chunk


def _calc_rows_by_min(array, min_):
    selected_rows = None if min_ is None else array >= min_
    return selected_rows


def _filter_all(chunk):
    flt_chunk = _filter_chunk(chunk, [GT_FIELD], _filter_all_rows)
    return flt_chunk


def _filter_none(chunk):
    return _filter_chunk(chunk, [GT_FIELD], _filter_no_row)


def _filter_mafs(chunk, min_=None, max_=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    mafs = calc_maf(chunk, min_num_genotypes=min_num_genotypes)
    with numpy.errstate(invalid='ignore'):
        selector_max = None if max_ is None else mafs <= max_
        selector_min = None if min_ is None else mafs >= min_

    if selector_max is None and selector_min is not None:
        selected_rows = selector_min
    elif selector_max is not None and selector_min is None:
        selected_rows = selector_max
    elif selector_max is not None and selector_min is not None:
        selected_rows = selector_min & selector_max
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def mafs_filter_fact(min_=None, max_=None,
                     min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    return partial(_filter_mafs, min_=min_, max_=max_,
                   min_num_genotypes=min_num_genotypes)


def _filter_obs_het(chunk, min_=None, max_=None,
                    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    obs_het = calc_obs_het(chunk, min_num_genotypes=min_num_genotypes)
    with numpy.errstate(invalid='ignore'):
        selector_max = None if max_ is None else obs_het <= max_
        selector_min = None if min_ is None else obs_het >= min_

    if selector_max is None and selector_min is not None:
        selected_rows = selector_min
    elif selector_max is not None and selector_min is None:
        selected_rows = selector_max
    elif selector_max is not None and selector_min is not None:
        selected_rows = selector_min & selector_max
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def obs_het_filter_fact(min_=None, max_=None,
                        min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    return partial(_filter_obs_het, min_=min_, max_=max_,
                   min_num_genotypes=min_num_genotypes)


def _min_called_gts_filter(chunk, min_=None, rates=True):
    called_gts = calc_called_gt(chunk, rates=rates)
    if min_ is not None:
        selected_rows = _calc_rows_by_min(called_gts, min_)
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def min_called_gts_filter_fact(min_=None, rates=True):
    return partial(_min_called_gts_filter, min_=min_, rates=rates)


def _filter_gt(chunk, min_, path):
    genotypes_field = chunk[path]
    gts = chunk[GT_FIELD].copy()
    if min_ is not None:
        gts[genotypes_field < min_] = MISSING_VALUES[int]
    return gts


def _quality_filter_gt(chunk, min_=None):
    return _filter_gt(chunk, min_, GQ_FIELD)


def quality_filter_genotypes_fact(min_=None):
    return partial(_quality_filter_gt, min_=min_)


def _quality_filter_snps(chunk, min_=None, max_=None):
    snps_qual = chunk['/variations/qual']
    with numpy.errstate(invalid='ignore'):
        selector_max = None if max_ is None else snps_qual <= max_
        selector_min = None if min_ is None else snps_qual >= min_

    if selector_max is None and selector_min is not None:
        selected_rows = selector_min
    elif selector_max is not None and selector_min is None:
        selected_rows = selector_max
    elif selector_max is not None and selector_min is not None:
        selected_rows = selector_min & selector_max
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def quality_filter_snps_fact(min_=None, max_=None):
    return partial(_quality_filter_snps, min_=min_, max_=max_)


def _filter_gts_by_dp(chunk, min_=None):
    return _filter_gt(chunk, min_, DP_FIELD)


def filter_gts_by_dp_fact(min_=None):
    return partial(_filter_gts_by_dp, min_=min_)


def filter_monomorphic_snps_fact(min_maf=None,
                                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    return partial(_filter_mafs, min_=min_maf,
                   min_num_genotypes=min_num_genotypes)


def _biallelic_filter(chunk, keep_monomorphic):
    gts = chunk[GT_FIELD]
    shape = gts.shape

    # we count how many different alleles are per row
    # we do it adding a complex part to each number. The complex part is
    # related with the row. Then we use unique
    weight = 1j * numpy.arange(0, shape[0])
    weight = numpy.repeat(weight, shape[1] * shape[2]).reshape(shape)
    b = gts + weight
    _, ind = numpy.unique(b, return_index=True)
    b = numpy.zeros_like(gts)
    c = numpy.ones_like(gts)
    numpy.put(b, ind, c.flat[ind])
    c = numpy.sum(b, axis=(2, 1))

    # we remove the missing values from the count
    rows_with_missing = numpy.any(gts == -1, axis=(1, 2))
    c -= rows_with_missing

    if keep_monomorphic:
        selected_rows = (c <= 2)
    else:
        selected_rows = (c == 2)

    return _filter_chunk2(chunk, selected_rows)


def filter_biallelic_and_polymorphic(chunk):
    return _biallelic_filter(chunk, keep_monomorphic=False)


def filter_biallelic(chunk):
    return _biallelic_filter(chunk, keep_monomorphic=True)
