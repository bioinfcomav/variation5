
import array
from collections import Counter
import itertools

import numpy
from scipy.stats import chi2_contingency

from variation.variations.stats import (calc_maf, calc_obs_het, GT_FIELD,
                                        calc_called_gt, GQ_FIELD, DP_FIELD,
                                        MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                        calc_mac, calc_snp_density,
                                        _calc_standarized_by_sample_depth,
                                        histogram, DEF_NUM_BINS)
from variation.variations.vars_matrices import VariationsArrays
from variation import MISSING_INT, SNPS_PER_CHUNK, MISSING_FLOAT, POS_FIELD, \
    REF_FIELD, CHROM_FIELD
from variation.matrix.methods import append_matrix, is_dataset
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
        self._samples = samples
        self._filter_samples = None

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

    @property
    def samples(self):
        return self._samples

    def _get_sample_filter(self):
        if self._filter_samples is not None:
            return self._filter_samples

        filter_samples = SampleFilter(self.samples, self._samples)
        self._filter_samples = filter_samples
        return filter_samples

    def _filter_samples_for_stats(self, variations):
        if self.samples is None:
            vars_for_stat = variations
        else:
            filter_samples = self._get_sample_filter()
            vars_for_stat = filter_samples(variations)[FLT_VARS]
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
        self._samples = samples
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
    snps1 = SampleFilter(samples1)(variations)[FLT_VARS]
    snps2 = SampleFilter(samples2)(variations)[FLT_VARS]
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


class SamplesFilterByIndex:
    def __init__(self, samples_col_idxs, reverse=False):
        self.samples_col_idxs = samples_col_idxs
        self.reverse = reverse
        self.can_be_in_pipeline = True

    @property
    def do_filtering(self):
        return True

    @property
    def do_histogram(self):
        return False

    def __call__(self, variations):
        flt_vars = _filter_samples_by_index(variations, self.samples_col_idxs,
                                            reverse=self.reverse)
        return {FLT_VARS: flt_vars}


class SampleFilter:
    def __init__(self, samples, reverse=False):
        self.samples = samples
        self.reverse = reverse
        self.can_be_in_pipeline = True

    @property
    def do_filtering(self):
        return True

    @property
    def do_histogram(self):
        return False

    def __call__(self, variations):
        samples = self.samples
        var_samples = variations.samples
        if len(set(var_samples)) != len(var_samples):
            repeated_samples = [item for item, cnt in
                                Counter(var_samples).items() if cnt > 1]
            msg = 'Some samples in the given variations are repeated: '
            msg += ','.join(repeated_samples)
            raise ValueError(msg)
        if len(set(samples)) != len(samples):
            raise ValueError('Some samples in the given samples are repeated')
        samples_not_int_vars = set(samples).difference(var_samples)
        if samples_not_int_vars:
            msg = 'Samples not found in variations: '
            msg += ','.join(samples_not_int_vars)
            raise ValueError(msg)

        idx_to_keep = [var_samples.index(sample) for sample in samples]

        filter_samples = SamplesFilterByIndex(idx_to_keep,
                                              reverse=self.reverse)
        return filter_samples(variations)


class MissingRateSampleFilter(_BaseFilter):
    def __init__(self, min_called_rate, all_variations=None,
                 chunk_size=SNPS_PER_CHUNK, **kwargs):
        self.min = min_called_rate
        self.all_variations = all_variations
        self.chunk_size = chunk_size

        self.how_to_be_in_pipeline = 'Add all_variations to __init__'

        can_be_in_pipeline = False if all_variations is None else True
        kwargs['can_be_in_pipeline'] = can_be_in_pipeline

        super().__init__(**kwargs)

    def _calc_stat_for_filtered_samples(self, variations):
        if self.samples is not None:
            msg = self.__class__.__name__
            msg += ' does not support samples'
            raise ValueError(msg)
        return super()._calc_stat_for_filtered_samples(variations)

    def _calc_stats_one_chunk(self, chunk):
        return calc_called_gt(chunk, rates=True, axis=0)

    def _calc_stats_by_chunk(self):
        variations = self.all_variations
        chunks = variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=self.chunk_size)

        missing = None
        for chunk in chunks:
            chunk_missing = calc_called_gt(chunk, rates=False, axis=0)
            if missing is None:
                missing = chunk_missing
            else:
                missing += chunk_missing
        rates = missing / variations.num_variations
        if self.range is None:
            self.range = min(rates), max(rates)
        return rates

    def _calc_stat(self, variations):
        if self.all_variations is None:
            missing_rates = self._calc_stats_one_chunk(variations)
        else:
            missing_rates = self._calc_stats_by_chunk()
        return missing_rates

    def _filter(self, variations, stat):
        idx_to_keep = stat > self.min
        filter_samples = SamplesFilterByIndex(idx_to_keep)
        return filter_samples(variations)[FLT_VARS]


def _calc_range_for_var_density(variations, window, chunk_size):

    min_, max_ = None, None
    for stats in group_in_packets(calc_snp_density(variations, window),
                                  chunk_size):
        stats = array.array('I', stats)
        this_min = min(stats)
        if min_ is None or min_ > this_min:
            min_ = this_min
        this_max = max(stats)
        if max_ is None or max_ < this_max:
            max_ = this_max
    return min_, max_


def filter_variation_density(in_vars, max_density, window, out_vars=None,
                             chunk_size=SNPS_PER_CHUNK, n_bins=DEF_NUM_BINS,
                             range_=None, do_histogram=None):
    if (n_bins != DEF_NUM_BINS or range_ is not None) and do_histogram is None:
        do_histogram = True
    elif do_histogram is None:
        do_histogram = False

    num_vars = in_vars.num_variations
    if not num_vars:
        if do_histogram:
            return {EDGES: [], COUNTS: []}
        else:
            return {}

    if chunk_size is None:
        chunk_size = num_vars

    do_filtering = False if out_vars is None else True

    if do_histogram and range_ is None:
        range_ = _calc_range_for_var_density(in_vars, window, chunk_size)

    stats = calc_snp_density(in_vars, window)
    edges, counts = None, None
    for chunk in in_vars.iterate_chunks(chunk_size=chunk_size):
        stats_for_chunk = itertools.islice(stats, chunk.num_variations)
        stats_for_chunk = numpy.array(array.array('I', stats_for_chunk))

        if do_filtering:
            selected_rows = stats_for_chunk <= max_density
            out_vars.put_chunks([chunk.get_chunk(selected_rows)])

        if do_histogram:
            this_counts, this_edges = histogram(stats_for_chunk, n_bins=n_bins,
                                                range_=range_)
            if edges is None:
                edges = this_edges
                counts = this_counts
            else:
                counts += this_counts
                if not numpy.allclose(edges, this_edges):
                    msg = 'Bin edges do not match in a chunk iteration'
                    raise RuntimeError(msg)
    return {EDGES: edges, COUNTS: counts}
