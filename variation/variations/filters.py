import numpy
from _functools import partial

from variation.variations.stats import (_MafCalculator,
                                        _MissingGTCalculator,
                                        _CalledGTCalculator)
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


def _filter_quality_GT(chunk, selected_rows):
    matrix = chunk['calls/GT']
    try:
        array = matrix.data
    except:
        array = matrix
    flt_data = numpy.extract(selected_rows,array)
    print(flt_data.shape)
    chunk['calls/GT'].data = flt_data
    print(len(chunk['calls/GT']))


def _filter_all(chunk):
    flt_chunk = _filter_chunk(chunk, ['/calls/GT'], _filter_all_rows)
    return flt_chunk


def _filter_none(chunk):
    return _filter_chunk(chunk, ['/calls/GT'], _filter_no_row)


def _filter_mafs(chunk, min_=None, max_=None):
    calc_mafs = _MafCalculator()
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
    calc_missing_gt = _MissingGTCalculator()
    rates = calc_missing_gt(chunk)
    if min_ is not None:
        selected_rows = _calc_min(rates, min_)
    else:
        selected_rows = _filter_no_row(chunk)
    return _filter_chunk2(chunk, selected_rows)


def missing_rate_filter_fact(min_=None):
    return partial(_missing_rate_filter, min_=min_)


def _min_called_gts_filter(chunk, min_=None):
    calc_called_gt = _CalledGTCalculator(rate=False)
    called_gts = calc_called_gt(chunk)
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


def _filter_gt(chunk, min_, path):
    genotypes_qual = chunk[path]
    gts = chunk['/calls/GT'].copy()
    if min_ is not None:
        gts[genotypes_qual < min_] = MISSING_VALUES[int]
    return gts


def _quality_filter_gt(chunk, min_=None):
    return _filter_gt(chunk, min_, '/calls/GQ')


def quality_filter_genotypes_fact(min_=None):
    return partial(_quality_filter_gt, min_=min_)


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


def _filter_gts_by_dp(chunk, min_=None):
    return _filter_gt(chunk, min_, '/calls/DP')


def filter_gts_by_dp_fact(min_=None):
    return partial(_filter_gts_by_dp, min_=min_)


def filter_monomorphic_snps_fact(min_maf=None):
    return partial(_filter_mafs, min_=min_maf)


def _biallelic_filter(chunk, keep_monomorphic):
    gts = chunk['/calls/GT']
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
    c = numpy.sum(b, axis=(2,1))

    # we remove the missing values from the count
    rows_with_missing = numpy.any(gts == -1, axis=(1,2))
    c -= rows_with_missing

    if keep_monomorphic:
        selected_rows = (c <= 2)
    else:
        selected_rows = (c == 2)

    return _filter_chunk2(chunk, selected_rows)


def biallelic_and_polymorphic_filter(chunk):
    return _biallelic_filter(chunk, keep_monomorphic=False)


def biallelic_filter(chunk):
    return _biallelic_filter(chunk, keep_monomorphic=True)
