
import itertools

import numpy

from variation.matrix.methods import is_missing

from variation.matrix.methods import iterate_matrix_chunks


def _kosman(gts, sample_i, sample_j, indi_cache):
    '''It calculates the distance between two individuals using the Kosman dist

    The Kosman distance is explain in DOI: 10.1111/j.1365-294X.2005.02416.x
    '''

    if sample_i in indi_cache:
        indi1, is_missing_1 = indi_cache[sample_i]
    else:
        indi1 = gts[:, sample_i]
        is_missing_1 = is_missing(indi1)
        indi_cache[sample_i] = indi1, is_missing_1

    if sample_j in indi_cache:
        indi2, is_missing_2 = indi_cache[sample_j]
    else:
        indi2 = gts[:, sample_j]
        is_missing_2 = is_missing(indi2)
        indi_cache[sample_j] = indi2, is_missing_2

    is_called = numpy.logical_not(numpy.logical_or(is_missing_1, is_missing_2))

    indi1 = indi1[is_called]
    indi2 = indi2[is_called]

    assert issubclass(indi1.dtype.type, numpy.integer)
    assert issubclass(indi2.dtype.type, numpy.integer)

    if indi1.shape[1] != 2:
        raise ValueError('Only diploid are allowed')

    alleles_comparison1 = indi1 == indi2.transpose()[:, :, None]
    alleles_comparison2 = indi2 == indi1.transpose()[:, :, None]

    result = numpy.add(numpy.any(alleles_comparison2, axis=2).sum(axis=0),
                       numpy.any(alleles_comparison1, axis=2).sum(axis=0))

    result2 = numpy.full(result.shape, fill_value=0.5)
    result2[result == 0] = 1
    result2[result == 4] = 0
    return result2.sum(), result2.shape[0]


def _indi_pairwise_dist(gts, indi_cache):
    n_samples = gts.shape[1]
    dists = numpy.zeros(int((n_samples ** 2 - n_samples) / 2))
    n_snps_matrix = numpy.zeros(int((n_samples ** 2 - n_samples) / 2))
    index = 0
    for sample_i, sample_j in itertools.combinations(range(n_samples), 2):
        dist, n_snps = _kosman(gts, sample_i, sample_j, indi_cache)
        dists[index] = dist
        n_snps_matrix[index] = n_snps
        index += 1
    return dists, n_snps_matrix


def _calc_pairwise_distance_by_chunk(gts, chunk_size):
    chunks = iterate_matrix_chunks(gts, chunk_size=chunk_size)

    abs_distances, n_snps_matrix = None, None
    for chunk in chunks:
        indi_cache = {}
        chunk_abs_distances, n_snps_chunk = _indi_pairwise_dist(chunk,
                                                                indi_cache)
        if abs_distances is None and n_snps_matrix is None:
            abs_distances = chunk_abs_distances.copy()
            n_snps_matrix = n_snps_chunk
        else:
            abs_distances = numpy.add(abs_distances, chunk_abs_distances)
            n_snps_matrix = numpy.add(n_snps_matrix, n_snps_chunk)
    return abs_distances / n_snps_matrix


def calc_pairwise_distance(variations, chunk_size=None):
    '''It calculates the distance between individuals using the Kosman distance.

    The Kosman distance is explain in DOI: 10.1111/j.1365-294X.2005.02416.x
    '''
    gts = variations['/calls/GT']
    if chunk_size:
        distance = _calc_pairwise_distance_by_chunk(gts, chunk_size)
    else:
        abs_dist, n_snps = _indi_pairwise_dist(gts, {})
        with numpy.errstate(invalid='ignore'):
            distance = abs_dist / n_snps
    return distance
