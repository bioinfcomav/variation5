from functools import partial
import array

import numpy
from scipy.stats import chi2_contingency

from allel.chunked.util import get_blen_array
from allel.model.ndarray import GenotypeArray
from allel.opt.stats import gn_locate_unlinked_int8

from variation.variations.stats import (calc_maf, calc_obs_het, GT_FIELD,
                                        calc_called_gt, GQ_FIELD, DP_FIELD,
                                        MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                        calc_mac, calc_snp_density,
                                        _calc_standarized_by_sample_depth,
                                        histogram, DEF_NUM_BINS)
from variation.variations.vars_matrices import VariationsArrays
from variation import MISSING_INT, SNPS_PER_CHUNK, MISSING_FLOAT
from variation.matrix.methods import (append_matrix, is_dataset,
                                      iterate_matrix_chunks)
from variation.iterutils import first, group_in_packets
from variation.matrix.stats import row_value_counter_fact


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


def _filter_by_chunk(variations, filtered_vars, filter_funct,
                     chunk_size=SNPS_PER_CHUNK):

    for chunk in variations.iterate_chunks(chunk_size=chunk_size):
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


def _filter_macs2(variations, filtered_vars=None, min_=None, max_=None,
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

    return _filter_chunk2(variations, filtered_vars, selected_rows), mafs


def _filter_macs(variations, filtered_vars=None, min_=None, max_=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    return _filter_macs2(variations, filtered_vars=filtered_vars, min_=min_,
                         max_=max_, min_num_genotypes=min_num_genotypes)[0]


def flt_hist_mac(variations, filtered_vars=None, min_=None, max_=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                 n_bins=DEF_NUM_BINS, range_=None):
    res = _filter_macs2(variations, filtered_vars=filtered_vars, min_=min_,
                        max_=max_, min_num_genotypes=min_num_genotypes)
    variations, stat = res
    counts, edges = histogram(stat, n_bins=n_bins, range_=range_)
    return variations, counts, edges


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


def _filter_obs_het2(variations, filtered_vars, min_=None, max_=None,
                     min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                     min_call_dp=0, samples=None):
    if samples is None:
        vars_for_stat = variations
    else:
        vars_for_stat = filter_samples(variations, samples, by_chunk=False)

    obs_het = calc_obs_het(vars_for_stat, min_num_genotypes=min_num_genotypes,
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
    flt_vars = _filter_chunk2(variations, filtered_vars, selected_rows)
    return flt_vars, obs_het


def _filter_obs_het(variations, filtered_vars, min_=None, max_=None,
                    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                    min_call_dp=0):
    res = _filter_obs_het2(variations, filtered_vars=filtered_vars,
                           min_=min_, max_=max_,
                           min_num_genotypes=min_num_genotypes,
                           min_call_dp=min_call_dp)
    return res[0]


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


def flt_hist_obs_het(variations, filtered_vars=None,
                     min_het=None, max_het=None,
                     min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                     min_call_dp=0, samples=None, n_bins=DEF_NUM_BINS,
                     range_=None):
    res = _filter_obs_het2(variations, filtered_vars=filtered_vars,
                           min_=min_het, max_=max_het,
                           min_num_genotypes=min_num_genotypes,
                           min_call_dp=min_call_dp, samples=samples)
    variations, stat = res
    counts, edges = histogram(stat, n_bins=n_bins, range_=range_)
    return variations, counts, edges


def _filter_min_called_gts2(variations, filtered_vars=None, min_=None,
                            rates=True):
    called_gts = calc_called_gt(variations, rates=rates)
    if min_ is not None:
        selected_rows = None if min_ is None else called_gts > min_
    else:
        selected_rows = _filter_no_row(variations)
    return _filter_chunk2(variations, filtered_vars, selected_rows), called_gts


def _filter_min_called_gts(variations, filtered_vars=None, min_=None,
                           rates=True):
    return _filter_min_called_gts2(variations, filtered_vars=filtered_vars,
                                   min_=min_, rates=rates)[0]


def flt_hist_min_called_gts(variations, filtered_vars=None, min_=None,
                            rates=True, n_bins=DEF_NUM_BINS, range_=None):
    res = _filter_min_called_gts2(variations, filtered_vars=filtered_vars,
                                  min_=min_, rates=rates)
    variations, stat = res
    counts, edges = histogram(stat, n_bins=n_bins, range_=range_)
    return variations, counts, edges


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


def _filter_standarized_by_sample_depth(variations, filtered_vars=None,
                                        max_std_dp=None, samples=None):
    if samples is None:
        vars_for_stat = variations
    else:
        vars_for_stat = filter_samples(variations, samples, by_chunk=False)

    stat = _calc_standarized_by_sample_depth(vars_for_stat)

    if max_std_dp:
        selected_rows = stat <= max_std_dp
    else:
        selected_rows = numpy.full_like(stat, True, dtype=numpy.bool_)

    return _filter_chunk2(variations, filtered_vars, selected_rows), stat


def _filter_standarized_by_sample_depth2(variations, filtered_vars=None,
                                         max_std_dp=None, samples=None):
    return _filter_standarized_by_sample_depth(variations,
                                               filtered_vars=filtered_vars,
                                               max_std_dp=max_std_dp,
                                               samples=samples)[0]


def filter_standarized_by_sample_depth(variations, filtered_vars=None,
                                       max_std_dp=None,
                                       chunk_size=SNPS_PER_CHUNK,
                                       samples=None):
    no_chunk_flt_funct = _filter_standarized_by_sample_depth2

    if chunk_size is None:
        return no_chunk_flt_funct(variations, filtered_vars=filtered_vars,
                                  max_std_dp=max_std_dp, samples=samples)
    else:
        filter_funct = partial(no_chunk_flt_funct, max_std_dp=max_std_dp,
                               samples=samples)
        return _filter_by_chunk(variations, filtered_vars, filter_funct)


def flt_hist_standarized_by_sample_depth(variations, filtered_vars=None,
                                         max_std_dp=None, n_bins=DEF_NUM_BINS,
                                         range_=None, samples=None):
    res = _filter_standarized_by_sample_depth(variations,
                                              filtered_vars=filtered_vars,
                                              max_std_dp=max_std_dp,
                                              samples=samples)
    variations, stat = res
    counts, edges = histogram(stat, n_bins=n_bins, range_=range_)
    return variations, counts, edges


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


def flt_hist_high_density_snps(variations, max_density, window,
                               filtered_vars=None, n_bins=DEF_NUM_BINS,
                               range_=None):
    res = _filter_high_density_snps(variations, max_density, window,
                                    filtered_vars=filtered_vars)
    variations, stat = res
    counts, edges = histogram(stat, n_bins=n_bins, range_=range_)
    return variations, counts, edges


def _filter_high_density_snps(variations, max_density, window,
                              filtered_vars=None):
    densities = calc_snp_density(variations, window)
    densities = numpy.array(array.array('I', densities))
    selected_rows = [dens <= max_density for dens in densities]
    filtered_vars = _filter_chunk2(variations, filtered_vars, selected_rows)
    if filtered_vars is None:
        filtered_vars = VariationsArrays()
    return filtered_vars, densities


def filter_high_density_snps(variations, max_density, window,
                             filtered_vars=None, chunk_size=SNPS_PER_CHUNK):
    densities = calc_snp_density(variations, window)
    densities = group_in_packets(densities, chunk_size)
    for chunk_idx0, chunk_densities in zip(range(0, variations.num_variations,
                                                 chunk_size),
                                           densities):
        chunk_idx1 = chunk_idx0 + chunk_size
        chunk = variations.get_chunk(slice(chunk_idx0, chunk_idx1))
        selected_rows = [dens <= max_density for dens in chunk_densities]
        filtered_vars = _filter_chunk2(chunk, filtered_vars, selected_rows)
    if filtered_vars is None:
        filtered_vars = VariationsArrays()
    return filtered_vars


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
    try:
        dtype = sample_cols.dtype
        is_bool = numpy.issubdtype(dtype, numpy.bool)
    except AttributeError:
        item = first(iter(sample_cols))
        is_bool = isinstance(item, bool)
    if not is_bool:
        sample_cols = [idx in sample_cols for idx in range(len(samples))]

    if 'shape' not in dir(sample_cols):
        sample_cols = numpy.array(sample_cols, dtype=numpy.bool)

    if reverse:
        sample_cols = numpy.logical_not(sample_cols)

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
    kept_samples = [samples[idx] for idx, keep in enumerate(sample_cols)
                    if keep]
    filtered_vars.samples = kept_samples
    return filtered_vars


def filter_samples_by_index(variations, sample_cols, filtered_vars=None,
                            reverse=False, by_chunk=True):
    if by_chunk:
        if filtered_vars is None:
            filtered_vars = VariationsArrays()
        chunks = (_filter_samples_by_index(chunk, sample_cols, reverse=reverse)
                  for chunk in variations.iterate_chunks())
        chunk = first(chunks)
        filtered_vars.put_chunks([chunk])
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


def flt_hist_samples_by_missing(variations, min_called_rate,
                                n_bins=DEF_NUM_BINS, range_=None):
    missing_rates = calc_called_gt(variations, rates=True, axis=0)
    counts, edges = histogram(missing_rates, n_bins=n_bins, range_=range_)
    idx_to_keep = missing_rates > min_called_rate

    variations = filter_samples_by_index(variations, idx_to_keep,
                                         by_chunk=False)
    return variations, counts, edges


def filter_samples_by_missing(variations, min_called_rate, by_chunk=True):
    missing_rates = calc_called_gt(variations, rates=True, axis=0)
    idx_to_keep = missing_rates > min_called_rate

    return filter_samples_by_index(variations, idx_to_keep,
                                   by_chunk=by_chunk)


COUNTS = 'counts'
EDGES = 'edges'
FLT_VARS = 'flt_vars'


class _BaseFilter:

    def __init__(self, n_bins=DEF_NUM_BINS, range_=None, do_filtering=True,
                 do_histogram=None, samples=None, can_be_in_pipeline=True):
        if do_histogram is None:
            if range_ is not None or n_bins != DEF_NUM_BINS:
                do_histogram = True
            else:
                do_histogram = False
        self.do_filtering = do_filtering
        self.do_histogram = do_histogram
        self.can_be_in_pipeline = can_be_in_pipeline
        self.n_bins = n_bins
        self.range = range_
        self.samples = samples

    def _filter(self, variations, stat):
        min_ = getattr(self, 'min', None)
        max_ = getattr(self, 'max', None)

        with numpy.errstate(invalid='ignore'):
            selector_max = None if max_ is None else stat <= max_
            selector_min = None if min_ is None else stat >= min_

        if selector_max is None and selector_min is not None:
            selected_rows = selector_min
        elif selector_max is not None and selector_min is None:
            selected_rows = selector_max
        elif selector_max is not None and selector_min is not None:
            selected_rows = selector_min & selector_max
        else:
            selected_rows = _filter_no_row(variations)
        return variations.get_chunk(selected_rows)

    def _filter_samples_for_stats(self, variations):
        if self.samples is None:
            vars_for_stat = variations
        else:
            vars_for_stat = filter_samples(variations, self.samples)
        return vars_for_stat

    def _calc_stat_for_filtered_samples(self, variations):
        vars_for_stat = self._filter_samples_for_stats(variations)
        return self._calc_stat(vars_for_stat)

    def __call__(self, variations):
        stats = self._calc_stat_for_filtered_samples(variations)
        result = {}

        if self.do_histogram:
            counts, edges = histogram(stats, n_bins=self.n_bins,
                                      range_=self.range)
            result[COUNTS] = counts
            result[EDGES] = edges

        if self.do_filtering:
            result[FLT_VARS] = self._filter(variations, stats)

        return result


class MinCalledGTsFilter(_BaseFilter):
    # def __init__(self, min_called=None, rates=True, n_bins=DEF_NUM_BINS,
    #             range_=None, do_filtering=True, do_histogram=True):
    def __init__(self, min_called=None, rates=True, **kwargs):
        self.rates = rates
        self.min = min_called
        super().__init__(**kwargs)

    def _calc_stat(self, variations):
        return calc_called_gt(variations, rates=self.rates)


class MafFilter(_BaseFilter):
    def __init__(self, min_maf=None, max_maf=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                 **kwargs):
        self.min = min_maf
        self.max = max_maf
        self.min_num_genotypes = min_num_genotypes

        super().__init__(**kwargs)

    def _calc_stat(self, variations):
        return calc_maf(variations, min_num_genotypes=self.min_num_genotypes)


class MacFilter(_BaseFilter):
    def __init__(self, min_mac=None, max_mac=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, **kwargs):
        self.min = min_mac
        self.max = max_mac
        self.min_num_genotypes = min_num_genotypes

        super().__init__(**kwargs)

    def _calc_stat(self, variations):
        return calc_mac(variations, min_num_genotypes=self.min_num_genotypes)


class ObsHetFilter(_BaseFilter):
    def __init__(self, min_het=None, max_het=None,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                 min_call_dp=0, **kwargs):
        self.min = min_het
        self.max = max_het
        self.min_num_genotypes = min_num_genotypes
        self.min_call_dp = min_call_dp

        super().__init__(**kwargs)

    def _calc_stat(self, variations):
        return calc_obs_het(variations,
                            min_num_genotypes=self.min_num_genotypes,
                            min_call_dp=self.min_call_dp)


class SNPQualFilter(_BaseFilter):
    def __init__(self, min_qual=None, max_qual=None, **kwargs):
        self.min = min_qual
        self.max = max_qual

        super().__init__(**kwargs)

    def _calc_stat_for_filtered_samples(self, variations):
        if self.samples is not None:
            raise ValueError('SNPQualFilter does not support samples')
        return self._calc_stat(variations)

    def _calc_stat(self, variations):
        stat = variations['/variations/qual']
        if is_dataset(stat):
            stat = stat[:]
        return stat


class _GTsToMissingSetter:
    def __init__(self, min_, field_path):
        self.min = min_
        self.field_path = field_path
        self.can_be_in_pipeline = True

    @property
    def do_filtering(self):
        return True

    @property
    def do_histogram(self):
        return False

    def __call__(self, variations):

        gts = variations[GT_FIELD][:]
        mat_to_check = variations[self.field_path]

        if is_dataset(variations[GT_FIELD]):
            mat_to_check = mat_to_check[:]
            gts[mat_to_check < self.min] = MISSING_INT
        else:
            gts[mat_to_check < self.min] = MISSING_INT

        copied_vars = variations.get_chunk(slice(None, None),
                                           ignored_fields=[GT_FIELD])
        copied_vars[GT_FIELD] = gts

        return {FLT_VARS: copied_vars}


class LowDPGTsToMissingSetter(_GTsToMissingSetter):
    def __init__(self, min_dp):
        super().__init__(min_=min_dp, field_path=DP_FIELD)


class LowQualGTsToMissingSetter(_GTsToMissingSetter):
    def __init__(self, min_qual):
        super().__init__(min_=min_qual, field_path=GQ_FIELD)


class NonBiallelicFilter(_BaseFilter):

    def __init__(self, samples=None):
        self.keep_monomorphic = False
        self.samples = samples
        self.can_be_in_pipeline = True

    @property
    def do_filtering(self):
        return True

    @property
    def do_histogram(self):
        return False

    def _select_mono(self, chunk):
        keep_monomorphic = self.keep_monomorphic

        gts = chunk[GT_FIELD]
        if is_dataset(gts):
            gts = gts[:]

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
        return selected_rows

    def __call__(self, variations):
        vars_for_stat = self._filter_samples_for_stats(variations)

        selected_rows = self._select_mono(vars_for_stat)
        result = {}
        if self.do_filtering:
            result[FLT_VARS] = variations.get_chunk(selected_rows)

        return result


class StdDepthFilter(_BaseFilter):
    def __init__(self, max_std_dp, **kwargs):
        self.max = max_std_dp
        kwargs['can_be_in_pipeline'] = False
        super().__init__(**kwargs)

    def _calc_stat(self, variations):
        return _calc_standarized_by_sample_depth(variations)


def _calc_fisher_for_gts(variations, samples1, samples2):
    snps1 = filter_samples(variations, samples1, by_chunk=False)
    snps2 = filter_samples(variations, samples2, by_chunk=False)
    gt_counts1 = _count_gts(snps1)
    gt_counts2 = _count_gts(snps2)

    genotypes = list(sorted(set(gt_counts1.keys()).union(gt_counts2.keys())))
    counts1 = {}
    counts2 = {}
    for gt in genotypes:
        wgs_counts_for_gt = gt_counts1.get(gt, None)
        gbs_counts_for_gt = gt_counts2.get(gt, None)
        if wgs_counts_for_gt is None:
            wgs_counts_for_gt = numpy.zeros_like(gbs_counts_for_gt)
        if gbs_counts_for_gt is None:
            gbs_counts_for_gt = numpy.zeros_like(wgs_counts_for_gt)
        counts1[gt] = wgs_counts_for_gt
        counts2[gt] = gbs_counts_for_gt

    chi2_vals = array.array('f')
    p_vals = array.array('f')
    counts = []
    for snp_idx in range(counts1[genotypes[0]].shape[0]):
        counts1_for_snp = [counts1[gt][snp_idx] for gt in genotypes]
        counts2_for_snp = [counts2[gt][snp_idx] for gt in genotypes]

        counts_for_snp = numpy.array([counts1_for_snp, counts2_for_snp])
        counts_for_snp = counts_for_snp[:, numpy.sum(counts_for_snp, axis=0) > 0]
        counts.append(counts_for_snp)
        try:
            chi2, pvalue, _, _ = chi2_contingency(counts_for_snp)
        except ValueError:
            chi2 = MISSING_FLOAT
            pvalue = MISSING_FLOAT
        chi2_vals.append(chi2)
        p_vals.append(pvalue)
    return numpy.array(chi2_vals), numpy.array(p_vals), counts


def _packed_gt_to_tuple(gt, ploidy):
    gt_len = ploidy * 2
    gt_fmt = '%0' + str(gt_len) + 'd'
    gt = gt_fmt % gt
    gt = tuple(sorted([int(gt[idx: idx + 2]) for idx in range(0, len(gt), 2)]))
    return gt


def _count_gts(variations):

    if variations[GT_FIELD].shape[0] == 0:
        return numpy.array([]), numpy.array([]), numpy.array([])

    gts = variations[GT_FIELD]
    gts = gts[...]

    # get rid of genotypes with missing alleles
    missing_alleles = gts == MISSING_INT
    miss_gts = numpy.any(missing_alleles, axis=2)

    # We pack the genotype of a sample that is in third axes as two
    # integers as one integer: 1, 1 -> 11 0, 1 -> 01, 0, 0-> 0
    gts_per_haplo = [(gts[:, :, idx].astype(numpy.int16)) * (100 ** idx) for idx in range(gts.shape[2])]
    packed_gts = None
    for gts_ in gts_per_haplo:
        if packed_gts is None:
            packed_gts = gts_
        else:
            packed_gts += gts_
    packed_gts[miss_gts] = MISSING_INT

    different_gts = numpy.unique(packed_gts)
    # Count genotypes, homo, het and alleles
    ploidy = gts.shape[2]
    counts = {}
    for gt in different_gts:
        count_gt_by_row = row_value_counter_fact(gt)
        gt_counts = count_gt_by_row(packed_gts)
        if gt == MISSING_INT:
            continue
        unpacked_gt = _packed_gt_to_tuple(gt, ploidy)
        if unpacked_gt not in counts:
            counts[unpacked_gt] = gt_counts
        else:
            counts[unpacked_gt] += gt_counts
    return counts


class Chi2GtFreqs2SampleSetsFilter(_BaseFilter):
    def __init__(self, samples1, samples2, min_pval, **kwargs):
        self.min = min_pval
        self.samples1 = samples1
        self.samples2 = samples2

        super().__init__(**kwargs)

    def _calc_stat_for_filtered_samples(self, variations):
        if self.samples is not None:
            msg = 'Chi2GtFreqs2SampleSetsFilter does not support samples'
            raise ValueError(msg)
        return self._calc_stat(variations)

    def _calc_stat(self, variations):
        _, p_vals, _ = _calc_fisher_for_gts(variations, self.samples1,
                                            self.samples2)
        return p_vals
