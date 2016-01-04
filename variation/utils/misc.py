
import numpy


def remove_nans(mat):
    return mat[~numpy.isnan(mat)]


def remove_inf(mat):
    return mat[~numpy.isinf(mat)]
