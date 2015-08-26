# Missing docstring
# pylint: disable=C0111

from functools import reduce, partial
import operator

import numpy


from variation.matrix.methods import iterate_matrix_chunks


def _row_value_counter_array(array, value, axes):
    return (array == value).sum(axis=axes)


def _crow_value_counter(mat, value, ratio=False):
    ndims = len(mat.shape)
    if ndims == 1:
        raise ValueError('The matrix has to have at least 2 dimensions')
    elif ndims == 2:
        axes = 1
    else:
        axes = tuple([i +1 for i in range(ndims - 1)])

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

def row_counter(mat):
    mat[numpy.isnan(mat)]=0
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
        axes = tuple([i +1 for i in range(ndims - 1)])

    chunks = iterate_matrix_chunks(mat)
    chunk_results = (_row_value_counter_array(chunk, value, axes) for chunk in chunks)
    result = _concat_arrays(chunk_results)

    if ratio:
        num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
        result = result / num_items_per_row
    return result


def row_value_counter_fact(value, ratio=False):
    return partial(_crow_value_counter, value=value, ratio=ratio)


def counts_by_row(mat, missing_value=None):

    alleles = (numpy.unique(mat))
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

    return allele_counts

