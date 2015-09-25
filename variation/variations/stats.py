import numpy
from functools import reduce
import operator

from variation import MISSING_VALUES
from variation.matrix.stats import counts_by_row, row_value_counter_fact
from variation.matrix.methods import append_matrix, calc_min_max, fill_array
from variation.variations.index import PosIndex

CHUNK_SIZE = 200
MIN_NUM_GENOTYPES_FOR_POP_STAT = 10


def _remove_nans(mat):
    return mat[~numpy.isnan(mat)]


def _remove_infs(mat):
    return mat[~numpy.isinf(mat)]


def _calc_items_in_row(mat):
    num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
    return num_items_per_row


# def plot_hist_mafs(var_mat, fhand=None, no_interactive_win=False):
#     mafs = calc_mafs(var_mat)
#     mafs = _remove_nans(mafs)
#     return plot_histogram(mafs, fhand=fhand,
#                           no_interactive_win=no_interactive_win)


def calc_stat_by_chunk(var_matrices, function, reduce_funct=None, axis=None):
    stats = None
    chunks = var_matrices.iterate_chunks(kept_fields=function.required_fields)
    for chunk in chunks:
        chunk_stats = function(chunk)
        if stats is None:
            stats = numpy.copy(chunk_stats)
        elif reduce_funct is not None:
            stats = reduce_funct(stats, chunk_stats)
        else:
            append_matrix(stats, chunk_stats)
    return stats


class MafCalculator:
    def __init__(self, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
        self.required_fields = ['/calls/GT']
        self.min_num_genotypes = min_num_genotypes

    def __call__(self, variations):
        gts = variations['/calls/GT']
        gt_counts = counts_by_row(gts, missing_value=MISSING_VALUES[int])
        max_ = numpy.amax(gt_counts, axis=1)
        sum_ = numpy.sum(gt_counts, axis=1)
        mafs_gt = max_ / sum_
        mafs_gt[sum_ < self.min_num_genotypes] = numpy.nan
        return mafs_gt


class MissingGTCalculator:
    def __init__(self, rate=True):
        self.required_fields = ['/calls/GT']
        self.rate = rate

    def __call__(self, chunk):
        gts = chunk['/calls/GT']
        missing = _missing_gt_counts(gts)
        if self.rate:
            num_items_per_row = _calc_items_in_row(gts)
            result = missing / num_items_per_row
        else:
            result = missing
        return result


def _missing_gt_counts(chunk):
    count_missing = row_value_counter_fact(MISSING_VALUES[int], ratio=False)
    return count_missing(chunk)


class _ObsHetCalculator:

    def __call__(self, chunk):
        # TODO min_num_genotypes
        gts = chunk['/calls/GT']
        is_het = numpy.logical_xor.reduce(gts, axis=2)
        rows_with_missing = numpy.any(gts == -1, axis=2)
        is_het[rows_with_missing == True] = False
        gts_missing = numpy.sum(rows_with_missing == 0, axis=self.axis)
        het = numpy.divide(numpy.sum(is_het, axis=self.axis), gts_missing)
        return het


class ObsHetCalculator(_ObsHetCalculator):
    def __init__(self, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
        self.axis = 1
        self.required_fields = ['/calls/GT']
        self.min_num_genotypes = min_num_genotypes


class ObsHetCalculatorBySample(_ObsHetCalculator):
    def __init__(self):
        self.axis = 0
        self.required_fields = ['/calls/GT']


class DepthDistributionCalculator():

    def __init__(self, max_depth):
        self.required_fields = ['/calls/DP']
        self.max_depth = max_depth

    def __call__(self, variations):
        dps = variations['/calls/DP']
        dps_sample = dps.transpose()
        counts_matrix = None
        for sample_dps in dps_sample:
            sample_dps[sample_dps == MISSING_VALUES[int]] = 0
            sample_dps = sample_dps[sample_dps <= self.max_depth]
            sample_dps_counts = numpy.bincount(sample_dps)
            if sample_dps_counts.shape[0] <= self.max_depth:
                sample_dps_counts = fill_array(sample_dps_counts,
                                               self.max_depth+1)
            if counts_matrix is None:
                counts_matrix = numpy.array([sample_dps_counts])
            else:
                counts_matrix = numpy.append(counts_matrix, [sample_dps_counts],
                                             axis=0)
        return counts_matrix


def _calc_depth_distribution_per_sample(variations, max_depth):
    calc_depth_distr = DepthDistributionCalculator(max_depth=max_depth)
    return calc_stat_by_chunk(variations, calc_depth_distr,
                              reduce_funct=numpy.add)


def calc_depth_cumulative_distribution_per_sample(variations, max_depth=None):
    if max_depth is None:
        _, max_depth = calc_min_max(variations['/calls/DP'])
    distributions = _calc_depth_distribution_per_sample(variations, max_depth)
    dp_cumulative_distr = numpy.cumsum(distributions, axis=1)
    return distributions, dp_cumulative_distr


class CalledGTCalculator:
    def __init__(self, rate=True):
        self.required_fields = ['/calls/GT']
        self.rate = rate

    def __call__(self, chunk):
        gts = chunk['/calls/GT']
        gt_counts = _called_gt(gts)
        gt = numpy.sum(gt_counts)
        if self.rate:
            gt_result = numpy.divide(gt_counts, gt)
        else:
            gt_result = gt_counts
        return gt_result


def _called_gt(gts):
    return _calc_items_in_row(gts) - _missing_gt_counts(gts)


def calc_snp_density(variations, window):
    dens = []
    index_right = 0
    dic_index = PosIndex(variations)
    n_snps = variations['/variations/chrom'].shape
    for i in range(n_snps[0]):
        chrom = variations['/variations/chrom'][i]
        pos = variations['/variations/pos'][i]
        pos_right = pos - window
        pos_left = window + pos
        if pos_right > variations['/variations/pos'][0]:
            index_right = dic_index.index_pos(chrom, pos_right)
        index_left = dic_index.index_pos(chrom, pos_left)
        dens.append(index_left - index_right + 1)
    return dens
