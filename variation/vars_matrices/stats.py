
import numpy

from variation import MISSING_VALUES
from variation.plot import plot_histogram
from variation.matrix.stats import counts_by_row, row_value_counter_fact
from functools import reduce
import operator


def plot_hist_missing_rate(var_mat, fhand=None, no_interactive_win=False):
    missing = missing_gt_rate(var_mat)
    return plot_histogram(missing, fhand=fhand,
                          no_interactive_win=no_interactive_win)


def _remove_nans(mat):
    return mat[~numpy.isnan(mat)]


def plot_hist_mafs(var_mat, fhand=None, no_interactive_win=False):
    mafs = calc_mafs(var_mat)
    mafs = _remove_nans(mafs)
    return plot_histogram(mafs, fhand=fhand,
                          no_interactive_win=no_interactive_win)


def calc_mafs(chunk, min_num_genotypes=10):
    genotypes = chunk['/calls/GT']
    allele_counts = counts_by_row(genotypes, missing_value=MISSING_VALUES[int])
    max_ = numpy.amax(allele_counts, axis=1)
    sum_ = numpy.sum(allele_counts, axis=1)
    mafs = max_ / sum_
    mafs[sum_ < min_num_genotypes] = numpy.nan
    return mafs


def _calc_items_in_row(mat):
    num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
    return num_items_per_row


def missing_gt_rate(chunk):
    missing = _missing_gt_counts(chunk)
    genotypes = chunk['/calls/GT']
    num_items_per_row =_calc_items_in_row(genotypes)
    result = missing / num_items_per_row
    return result


def _missing_gt_counts(chunk):
    genotypes = chunk['/calls/GT']
    count_missing = row_value_counter_fact(MISSING_VALUES[int], ratio=False)
    return count_missing(genotypes)


def called_gt_counts(chunk):
    genotypes = chunk['/calls/GT']
    return _calc_items_in_row(genotypes) - _missing_gt_counts(chunk)
