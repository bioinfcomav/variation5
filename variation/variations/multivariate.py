
from collections import Counter

from numpy.linalg import eigh, svd
from numpy import newaxis, add, transpose, sqrt, argsort, array, dot
from numpy import sum as nsum
import numpy

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform

import allel

from variation.variations.stats import GT_FIELD
from variation.variations import VariationsArrays
from variation.variations.filters import keep_biallelic
from variation import MISSING_INT
from variation.matrix.stats import counts_by_row
from variation.matrix.methods import is_dataset


def non_param_multi_dim_scaling(dists, n_dims=3, n_threads=None, metric=True):
    mds = MDS(n_components=n_dims, metric=metric, n_jobs=n_threads,
              dissimilarity='precomputed')
    mds.fit(squareform(dists))
    projs = mds.embedding_
    res = {'stress': mds.stress_,
           'projections': projs}
    return res


def _make_f_matrix(matrix):
    """It takes an E matrix and returns an F matrix

    The input is the output of make_E_matrix

    For each element in matrix subtract mean of corresponding row and
    column and add the mean of all elements in the matrix
    """
    num_rows, num_cols = matrix.shape
    # make a vector of the means for each row and column
    # column_means = (add.reduce(E_matrix) / num_rows)
    column_means = (add.reduce(matrix) / num_rows)[:, newaxis]
    trans_matrix = transpose(matrix)
    row_sums = add.reduce(trans_matrix)
    row_means = row_sums / num_cols
    # calculate the mean of the whole matrix
    matrix_mean = nsum(row_sums) / (num_rows * num_cols)
    # adjust each element in the E matrix to make the F matrix

    matrix -= row_means
    matrix -= column_means
    matrix += matrix_mean

    return matrix


def do_pcoa(dists):
    'It does a Principal Coordinate Analysis on a distance matrix'
    # the code for this function is taken from pycogent metric_scaling.py
    # Principles of Multivariate analysis: A User's Perspective.
    # W.J. Krzanowski Oxford University Press, 2000. p106.

    dists = squareform(dists)

    e_matrix = (dists * dists) / -2.0
    f_matrix = _make_f_matrix(e_matrix)

    eigvals, eigvecs = eigh(f_matrix)
    eigvecs = eigvecs.transpose()
    # drop imaginary component, if we got one
    eigvals, eigvecs = eigvals.real, eigvecs.real

    # convert eigvals and eigvecs to point matrix
    # normalized eigenvectors with eigenvalues

    # get the coordinates of the n points on the jth axis of the Euclidean
    # representation as the elements of (sqrt(eigvalj))eigvecj
    # must take the absolute value of the eigvals since they can be negative
    pca_matrix = eigvecs * sqrt(abs(eigvals))[:, newaxis]

    # output
    # get order to output eigenvectors values. reports the eigvecs according
    # to their cooresponding eigvals from greatest to least
    vector_order = list(argsort(eigvals))
    vector_order.reverse()

    eigvals = eigvals[vector_order]

    # eigenvalues
    pcnts = (eigvals / nsum(eigvals)) * 100.0

    # the outputs
    # eigenvectors in the original pycogent implementation, here we name them
    # princoords
    # I think that we're doing: if the eigenvectors are written as columns,
    # the rows of the resulting table are the coordinates of the objects in
    # PCO space
    projections = []
    for name_i in range(dists.shape[0]):
        eigvect = [pca_matrix[vec_i, name_i] for vec_i in vector_order]
        projections.append(eigvect)
    projections = array(projections)

    return {'projections': projections,
            'var_percentages': pcnts}


def do_pcax(variations, n_components=10):

    # keep biallelic snps
    # TODO: We should select only the GT matrix
    snps = VariationsArrays(variations)
    keep_biallelic(variations, snps)

    genotypes = allel.GenotypeArray(snps[GT_FIELD])
    # print(genotypes.shape)
    # print(genotypes)

    # transform the genotype data into a 2-dimensional matrix where each cell
    # has the number of non-reference alleles per call

    geno_allele_counts = genotypes.to_n_alt()
    print(geno_allele_counts)
    print(geno_allele_counts.shape)

    coords, model = allel.stats.pca(geno_allele_counts,
                                    n_components=n_components,
                                    scaler='patterson')
    return {'projections': coords}


def _center_matrix(matrix):
    'It centers the matrix'
    means = matrix.mean(axis=0)
    return matrix - means


def _standarize_matrix(matrix):
    'It centers the matrix'
    means = matrix.mean(axis=0)
    std_devs = matrix.std(axis=0)
    # center the matrix
    matrix = (matrix - means) / std_devs
    return matrix


def do_pca(variations):
    'It does a Principal Component Analysis'

    # transform the genotype data into a 2-dimensional matrix where each cell
    # has the number of non-reference alleles per call

    matrix = variations.gts_as_mat012

    n_rows, n_cols = matrix.shape
    if n_rows < n_cols:
        # This restriction is in the matplotlib implementation, but I don't
        # know the reason
        msg = 'The implementation requires more rows than columns'
        raise RuntimeError(msg)

    # Implementation based on the matplotlib PCA class
    cen_matrix = _center_matrix(matrix)
    # The following line should be added from a example to get the correct
    # variances
    # cen_scaled_matrix = cen_matrix / math.sqrt(n_rows - 1)
    cen_scaled_matrix = cen_matrix

    singular_vals, princomps = svd(cen_scaled_matrix, full_matrices=False)[1:]
    eig_vals = singular_vals ** 2
    pcnts = eig_vals / eig_vals.sum() * 100.0
    projections = dot(princomps, cen_matrix.T).T

    return {'projections': projections,
            'var_percentages': pcnts,
            'princomps': princomps}
