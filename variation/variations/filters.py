from functools import partial
from itertools import chain

import numpy

from allel.chunked.util import get_blen_array
from allel.model.ndarray import GenotypeArray
from allel.opt.stats import gn_locate_unlinked_int8

from variation.variations.stats import (calc_maf, calc_obs_het, GT_FIELD,
                                        calc_called_gt, GQ_FIELD, DP_FIELD,
                                        MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                        calc_mac)
from variation.variations.vars_matrices import VariationsArrays
from variation import MISSING_INT, SNPS_PER_CHUNK
from variation.matrix.methods import (append_matrix, is_dataset,
                                      iterate_matrix_chunks)
from variation.iterutils import first


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


def _filter_chunk2(chunk, filtered_chunk, selected_rows):
    if filtered_chunk is None:
        filtered_chunk = VariationsArrays()

    for path in chunk.keys():
        matrix = chunk[path]
        try:
            array = matrix.data
        except:
            array = matrix
        flt_data = numpy.compress(selected_rows, array, axis=0)
        try:
            out_mat = filtered_chunk[path]
            append_matrix(out_mat, flt_data)
        except KeyError:
            filtered_chunk[path] = flt_data
    filtered_chunk.metadata = chunk.metadata
    filtered_chunk.samples = chunk.samples
    return filtered_chunk


def _filter_mafs(variations, filtered_vars=None, min_=None, max_=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    mafs = calc_maf(variations, min_num_genotypes=min_num_genotypes)

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
        selected_rows = _filter_no_row(variations)

    return _filter_chunk2(variations, filtered_vars, selected_rows)


def _filter_by_chunk(variations, filtered_vars, filter_funct):

    for chunk in variations.iterate_chunks():
        filtered_vars = filter_funct(chunk, filtered_vars)
    return filtered_vars


def filter_mafs(variations, filtered_vars=None, min_maf=None, max_maf=None,
                min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                by_chunk=True):
    if by_chunk:
        filter_funct = partial(_filter_mafs, min_=min_maf, max_=max_maf,
                               min_num_genotypes=min_num_genotypes)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return _filter_mafs(variations, filtered_vars=filtered_vars,
                            min_=min_maf, max_=max_maf,
                            min_num_genotypes=min_num_genotypes)


def _filter_macs(variations, filtered_vars=None, min_=None, max_=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    mafs = calc_mac(variations, min_num_genotypes=min_num_genotypes)

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
        selected_rows = _filter_no_row(variations)

    return _filter_chunk2(variations, filtered_vars, selected_rows)


def filter_macs(variations, filtered_vars=None, min_mac=None, max_mac=None,
                min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                by_chunk=True):
    if by_chunk:
        filter_funct = partial(_filter_macs, min_=min_mac, max_=max_mac,
                               min_num_genotypes=min_num_genotypes)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return _filter_macs(variations, filtered_vars=filtered_vars,
                            min_=min_mac, max_=max_mac,
                            min_num_genotypes=min_num_genotypes)


def _filter_obs_het(variations, filtered_vars, min_=None, max_=None,
                    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                    min_call_dp=0):
    obs_het = calc_obs_het(variations, min_num_genotypes=min_num_genotypes,
                           min_call_dp=min_call_dp)
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
        selected_rows = _filter_no_row(variations)
    return _filter_chunk2(variations, filtered_vars, selected_rows)


def filter_obs_het(variations, filtered_vars=None, min_het=None, max_het=None,
                   min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                   min_call_dp=0, by_chunk=True):
    if by_chunk:
        filter_funct = partial(_filter_obs_het, min_=min_het, max_=max_het,
                               min_num_genotypes=min_num_genotypes,
                               min_call_dp=min_call_dp)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return _filter_obs_het(variations, filtered_vars=filtered_vars,
                               min_=min_het, max_=max_het,
                               min_num_genotypes=min_num_genotypes,
                               min_call_dp=min_call_dp)


def _filter_min_called_gts(variations, filtered_vars=None, min_=None,
                           rates=True):
    called_gts = calc_called_gt(variations, rates=rates)
    if min_ is not None:
        selected_rows = None if min_ is None else called_gts > min_
    else:
        selected_rows = _filter_no_row(variations)
    return _filter_chunk2(variations, filtered_vars, selected_rows)


def filter_min_called_gts(variations, filtered_vars=None, min_called=None,
                          rates=True, by_chunk=True):
    no_chunk_flt_funct = _filter_min_called_gts
    if by_chunk:
        filter_funct = partial(no_chunk_flt_funct, min_=min_called,
                               rates=rates)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return no_chunk_flt_funct(variations, filtered_vars=filtered_vars,
                                  min_=min_called, rates=rates)


def _filter_gt_by_chunk(variations, filter_funct, field_path):

    gt_mat = variations[GT_FIELD]
    idx = 0
    for chunk in variations.iterate_chunks():
        filtered_gt_chunk = filter_funct(chunk)
        n_snps_in_chunk = filtered_gt_chunk.shape[0]
        gt_mat[idx: idx + n_snps_in_chunk] = filtered_gt_chunk
        idx += n_snps_in_chunk


def _filter_gt_no_chunk(variations, field_path, min_):
    mat_to_check = variations[field_path]
    if is_dataset(mat_to_check):
        mat_to_check = mat_to_check[:]
    gts = variations[GT_FIELD]

    if min_ is not None:
        if is_dataset(variations[GT_FIELD]):
            gts = variations[GT_FIELD][:]
            gts[mat_to_check < min_] = MISSING_INT
            variations[GT_FIELD][:] = gts
        else:
            gts[mat_to_check < min_] = MISSING_INT
    return gts


def _filter_gt(variations, min_, field_path, by_chunk=True):
    no_chunk_flt_funct = _filter_gt_no_chunk
    if by_chunk:
        filter_funct = partial(no_chunk_flt_funct, min_=min_,
                               field_path=field_path)
        _filter_gt_by_chunk(variations, filter_funct, field_path=field_path)
    else:
        no_chunk_flt_funct(variations, min_=min_, field_path=field_path)


def set_low_qual_gts_to_missing(variations, min_qual=None, by_chunk=True):
    _filter_gt(variations, min_qual, field_path=GQ_FIELD, by_chunk=by_chunk)


def set_low_dp_gts_to_missing(variations, min_dp=None, by_chunk=True):
    _filter_gt(variations, min_dp, field_path=DP_FIELD, by_chunk=by_chunk)


def _filter_snps_by_qual(variations, filtered_vars=None, min_=None, max_=None):
    snps_qual = variations['/variations/qual']
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
        selected_rows = _filter_no_row(variations)
    return _filter_chunk2(variations, filtered_vars, selected_rows)


def filter_snps_by_qual(variations, filtered_vars=None, min_qual=None,
                        max_qual=None, by_chunk=True):
    no_chunk_flt_funct = _filter_snps_by_qual
    if by_chunk:
        filter_funct = partial(no_chunk_flt_funct, min_=min_qual,
                               max_=max_qual)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return no_chunk_flt_funct(variations, filtered_vars=filtered_vars,
                                  min_=min_qual, max_=max_qual)


def _filter_monomorphic_snps(variations, filtered_vars=None, min_maf=None,
                             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    return filter_mafs(variations, filtered_vars=filtered_vars,
                       min_maf=min_maf, min_num_genotypes=min_num_genotypes)


def filter_monomorphic_snps(variations, filtered_vars=None, min_maf=None,
                            min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                            by_chunk=True):
    no_chunk_flt_funct = _filter_monomorphic_snps
    if by_chunk:
        filter_funct = partial(no_chunk_flt_funct, min_maf=min_maf,
                               min_num_genotypes=min_num_genotypes)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return no_chunk_flt_funct(variations, filtered_vars=filtered_vars,
                                  min_maf=min_maf,
                                  min_num_genotypes=min_num_genotypes)


def _biallelic_filter(chunk, filtered_vars, keep_monomorphic):
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

    return _filter_chunk2(chunk, filtered_vars, selected_rows)


def _filter_non_biallelic(variations, filtered_vars=None,
                          keep_monomorphic=False, by_chunk=True):
    no_chunk_flt_funct = _biallelic_filter
    if by_chunk:
        filter_funct = partial(no_chunk_flt_funct,
                               keep_monomorphic=keep_monomorphic)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)
    else:
        return no_chunk_flt_funct(variations, filtered_vars=filtered_vars,
                                  keep_monomorphic=keep_monomorphic)


def keep_biallelic(variations, filtered_vars=None, by_chunk=True):
    return _filter_non_biallelic(variations, filtered_vars=filtered_vars,
                                 keep_monomorphic=False, by_chunk=by_chunk)


def keep_biallelic_and_monomorphic(variations, filtered_vars=None,
                                   by_chunk=True):
    return _filter_non_biallelic(variations, filtered_vars=filtered_vars,
                                 keep_monomorphic=True, by_chunk=by_chunk)


def _filter_samples_by_index(variations, sample_cols, filtered_vars=None,
                             reverse=False):
    if filtered_vars is None:
        filtered_vars = VariationsArrays()

    samples = variations.samples
    if reverse:
        sample_cols = [idx for idx in range(len(samples))
                       if idx not in sample_cols]

    for path in variations.keys():
        matrix = variations[path]
        if is_dataset(matrix):
            matrix = matrix[:]
        if 'calls' in path:
            flt_data = matrix[:, sample_cols]
            # flt_data = numpy.compress(sample_cols, , axis=1)
            filtered_vars[path] = flt_data
        else:
            filtered_vars[path] = matrix
    filtered_vars.metadata = variations.metadata
    filtered_vars.samples = [samples[idx] for idx in sample_cols]
    return filtered_vars


def filter_samples_by_index(variations, sample_cols, filtered_vars=None,
                            reverse=False, by_chunk=True):
    if by_chunk:
        if filtered_vars is None:
            filtered_vars = VariationsArrays()
        chunks = (_filter_samples_by_index(chunk, sample_cols, reverse=reverse)
                  for chunk in variations.iterate_chunks())
        chunk = first(chunks)
        chunks = chain([chunk], chunks)
        filtered_vars.put_chunks(chunks)
        filtered_vars.metadata = chunk.metadata
        filtered_vars.samples = chunk.samples
        return filtered_vars
    else:
        return _filter_samples_by_index(variations, sample_cols,
                                        filtered_vars=filtered_vars,
                                        reverse=reverse)


def filter_samples(variations, samples, filtered_vars=None,
                   reverse=False, by_chunk=True):
    var_samples = variations.samples
    if len(set(var_samples)) != len(var_samples):
        raise ValueError('Some samples in the given variations are repeated')
    if len(set(samples)) != len(samples):
        raise ValueError('Some samples in the given samples are repeated')
    samples_not_int_vars = set(samples).difference(var_samples)
    if samples_not_int_vars:
        msg = 'Samples not found in variations: '
        msg += ','.join(samples_not_int_vars)
        raise ValueError(msg)

    idx_to_keep = [var_samples.index(sample) for sample in samples]

    return filter_samples_by_index(variations, idx_to_keep, reverse=reverse,
                                   by_chunk=by_chunk)


def locate_unlinked(gts, window_size=100, step=20, threshold=.1,
                    chunk_size=None, gts_to_gns=False):
    """modified from https://github.com/cggh/scikit-allel"""
    # check inputs
    if not gts_to_gns:
        if not hasattr(gts, 'shape') or not hasattr(gts, 'dtype'):
            gn = numpy.asarray(gts, dtype='i1')
        if gn.ndim != 2:
            raise ValueError('gts must have two dimensions')

    # setup output
    loc = numpy.ones(gts.shape[0], dtype='u1')

    # compute in chunks to avoid loading big arrays into memory
    chunk_size = get_blen_array(gts, chunk_size)
    # avoid too small chunks
    chunk_size = max(chunk_size, 10 * window_size)
    n_variants = gts.shape[0]
    for i in range(0, n_variants, chunk_size):
        # N.B., ensure overlap with next window
        j = min(n_variants, i + chunk_size + window_size)
        chunk_gts = numpy.asarray(gts[i:j], dtype='i1')
        if gts_to_gns:
            chunk_gts = GenotypeArray(chunk_gts)
            chunk_gns = chunk_gts.to_n_alt(fill=MISSING_INT)
        else:
            chunk_gns = chunk_gts
        locb = loc[i:j]
        gn_locate_unlinked_int8(chunk_gns, locb, window_size, step, threshold)

    return loc.astype('b1')


def filter_unlinked_vars(variations, window_size, step=20, filtered_vars=None,
                         r2_threshold=0.1, by_chunk=True,
                         chunk_size=SNPS_PER_CHUNK):
    gts = variations[GT_FIELD]
    if filtered_vars is None:
        filtered_vars = VariationsArrays()

    unlinked_mask = locate_unlinked(gts, window_size, step, r2_threshold,
                                    chunk_size=chunk_size, gts_to_gns=True)

    if by_chunk:
        selected_rows_chunks = iterate_matrix_chunks(unlinked_mask,
                                                     chunk_size=chunk_size)
        chunks = variations.iterate_chunks(chunk_size=chunk_size)
        filtered_chunks = (_filter_chunk2(chunk, filtered_chunk=None,
                                          selected_rows=sel_row)
                           for chunk, sel_row in zip(chunks,
                                                     selected_rows_chunks))
        filtered_vars.put_chunks(filtered_chunks)
    else:
        filtered_vars = _filter_chunk2(variations, filtered_vars,
                                       selected_rows=unlinked_mask)
    return filtered_vars
