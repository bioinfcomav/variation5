
import numpy


def remove_nans(mat):
    return mat[~numpy.isnan(mat)]