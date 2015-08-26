import numpy
from _functools import partial

from variation.iterutils import first
from variation.vars_matrices.stats import (calc_mafs,
                                           missing_gt_rate,
                                           called_gt_counts,
                                           counts_by_row,
                                           calc_quality_genotypes,
                                           calc_quality_rd)
from variation.vars_matrices import VariationsArrays


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


def _quality_filter_g(chunk, min_=None, max_=None):
    genotypes_qual = calc_quality_genotypes(chunk)
    selector_max = None if max_ is None else genotypes_qual <= max_
    selector_min = None if min_ is None else genotypes_qual >= min_

    if selector_max is None and selector_min is not None:
        selected_rows = selector_min
    elif selector_max is not None and selector_min is None:
        selected_rows = selector_max
    elif selector_max is not None and selector_min is not None:
        selected_rows = selector_min & selector_max
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def quality_filter_genotypes_fact(min_=None, max_=None):
    return partial(_quality_filter_g, min_=min_, max_=max_)


def _quality_filter_snps(chunk, min_=None, max_=None):
    snps_qual = chunk['/variations/qual']
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


def _quality_filter_dp(chunk, min_=None, max_=None):
    snps_qual = calc_quality_rd(chunk)
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


def quality_filter_snps_dp(min_=None, max_=None):
    return partial(_quality_filter_dp, min_=min_, max_=max_)

def monomorfic_filter(chunk):
    data = chunk["/calls/GT"]
    mafs = counts_by_row(data)
    array = (mafs == 0)
    result = numpy.logical_xor.reduce(array,1)
    return numpy.sum(result == True)

def billelic_filter(chunk):
    return _no_monomorfic_filter(chunk)

def disorderly_allelic(chunk):
    return _no_monomorfic_filter(chunk)

def heterozygosity_filter(chunk):
    return _no_monomorfic_filter(chunk)

def _no_monomorfic_filter(chunk):
    dim = chunk["/calls/GT"].shape[0]
    return dim-monomorfic_filter(chunk)

