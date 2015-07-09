import numpy
from _functools import partial

from variation.iterutils import first
from variation.vcfh5.chunk_stats import (calc_mafs,
                                         missing_gt_rate,
                                         called_gt_counts)


def _filter_dsets_chunks(selector_function, dsets_chunks):
    for dsets_chunk in dsets_chunks:
        bool_selection = selector_function(dsets_chunk)
        flt_dsets_chunk = {}
        for field, dset_chunk in dsets_chunk.items():
            flt_data = numpy.compress(bool_selection, dset_chunk.data, axis=0)
            flt_dsets_chunk[field] = flt_data
        yield flt_dsets_chunk


def _get_num_snps_from_chunk(chunk):
    array = first(chunk.values())
    return array.shape[0]


def _filter_all_rows(arrays):
    n_snps = _get_num_snps_from_chunk(arrays)
    selector = numpy.zeros((n_snps,), dtype=numpy.bool_)
    return selector


def _filter_no_row(arrays):
    n_snps = _get_num_snps_from_chunk(arrays)
    selector = numpy.ones((n_snps,), dtype=numpy.bool_)
    return selector


def _filter_chunk(chunk, fields, filter_funct):
    arrays = {field: chunk[field].data for field in fields}
    selected_rows = filter_funct(arrays)
    return _filter_chunk2(chunk, selected_rows)



def _filter_chunk2(chunk, selected_rows):
    flt_dsets_chunk = {}
    for field, dset_chunk in chunk.items():
        flt_data = numpy.compress(selected_rows, dset_chunk.data, axis=0)
        flt_dsets_chunk[field] = flt_data
    return flt_dsets_chunk


def _filter_all(chunk):
    flt_chunk = _filter_chunk(chunk, ['/calls/GT'], _filter_all_rows)
    return flt_chunk


def _filter_none(chunk):
    return _filter_chunk(chunk, ['/calls/GT'], _filter_no_row)


def _filter_mafs(chunk, min_=None, max_=None):
    mafs = calc_mafs(chunk)
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


def mafs_filter_fact(min_=None, max_=None):
    return partial(_filter_mafs, min_=min_, max_=max_)


def _missing_rate_filter(chunk, min_=None):
    rates = missing_gt_rate(chunk)
    if min_ is not None:
        selected_rows = _calc_min(rates, min_)
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def missing_rate_filter_fact(min_=None):
    return partial(_missing_rate_filter, min_=min_)


def _min_called_gts_filter(chunk, min_=None):
    called_gts = called_gt_counts(chunk)
    if min_ is not None:
        selected_rows = _calc_min(called_gts, min_)
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def min_called_gts_filter_fact(min_=None):
    return partial(_min_called_gts_filter, min_=min_)


def _calc_min(array, min_):
    selected_rows = None if min_ is None else array >= min_
    return selected_rows