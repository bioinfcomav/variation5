# Missing docstring
# pylint: disable=C0111

from functools import reduce, partial
import operator

import numpy
import matplotlib.pyplot as plt

from variation.matrix.methods import iterate_matrix_chunks


def _row_value_counter_array(array, value, axes):
    return (array == value).sum(axis=axes)


def row_counter(mat):
    mat[numpy.isnan(mat)] = 0
    result = numpy.sum(mat, axis=1)
    num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
    result = result / num_items_per_row
    return result


def _concat_arrays(arrays):
    concat = None
    for array in arrays:
        if concat is None:
            concat = array
            continue
        n_new_snps = array.shape[0]
        nsnps = concat.shape[0] + n_new_snps
        shape = list(concat.shape)
        shape[0] = nsnps
        concat.resize(shape, refcheck=False)
        concat[-n_new_snps:] = array
    return concat


def _row_value_counter(mat, value, ratio=False):
    ndims = len(mat.shape)
    if ndims == 1:
        raise ValueError('The matrix has to have at least 2 dimensions')
    elif ndims == 2:
        axes = 1
    else:
        axes = tuple([i + 1 for i in range(ndims - 1)])

    chunks = iterate_matrix_chunks(mat)
    result = numpy.zeros(mat.shape[0])
    start = 0
    for chunk in chunks:
        chunk_result = _row_value_counter_array(chunk, value, axes)
        end = start + chunk_result.shape[0]
        result[start:end] = chunk_result
        start = end

    if ratio:
        num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
        result = result / num_items_per_row
    return result


def row_value_counter_fact(value, ratio=False):
    return partial(_row_value_counter, value=value, ratio=ratio)


def counts_by_row(mat, missing_value=None):
    return counts_and_allels_by_row(mat, missing_value=missing_value)[0]


def counts_and_allels_by_row(mat, missing_value=None):

    alleles = sorted(numpy.unique(mat))
    allele_counts = None
    # This algorithm is suboptimal, it would be better to go row per row
    # the problem is a for snp in gts is very slow because the for in
    # python is slow
    for allele in alleles:
        if allele == missing_value:
            continue
        allele_counter = row_value_counter_fact(allele)
        counts = allele_counter(mat)
        if allele_counts is None:
            allele_counts = counts
        else:
            allele_counts = numpy.column_stack((allele_counts, counts))

    if allele_counts is None:
        return None, None

    if len(allele_counts.shape) == 1:
        # it happens if there is only one non missing allele
        new_shape = allele_counts.shape[0], 1
        dtype = allele_counts.dtype
        allele_counts = numpy.copy(allele_counts.reshape(new_shape))
        allele_counts = allele_counts.astype(dtype)

    return allele_counts, alleles


def plot_hist(hist, bins, print_plot=False):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    if print_plot:
        plt.show()
