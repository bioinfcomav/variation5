

import functools
import operator

import numpy

from variation import MISSING_GT


class RowValueCounter():

    def __init__(self, value, ratio=False):
        self.value = value  # value to count
        self.ratio = ratio

    def __call__(self, mat):
        ndims = len(mat.shape)
        if ndims == 1:
            raise ValueError('The matrix has to have at least 2 dimensions')
        elif ndims == 2:
            axes = 1
        else:
            axes = tuple([i +1 for i in range(ndims - 1)])
        result = (mat == self.value).sum(axis=axes)
        if self.ratio:
            num_items_per_row = functools.reduce(operator.mul, mat.shape[1:],
                                                 1)
            result = result / num_items_per_row
        return result


def calc_mafs(genotypes, min_num_genotypes=10):

    alleles = (numpy.unique(genotypes))
    allele_counts = None
    for allele in alleles:
        if allele == MISSING_GT:
            continue
        allele_counter = RowValueCounter(allele)
        counts = allele_counter(genotypes)
        if allele_counts is None:
            allele_counts = counts
        else:
            allele_counts = numpy.vstack((allele_counts, counts))

    max_ = numpy.amax(allele_counts, axis=0)
    sum_ = numpy.sum(allele_counts, axis=0)
    mafs = max_ / sum_
    mafs[sum_ < min_num_genotypes] = numpy.nan
    return mafs
