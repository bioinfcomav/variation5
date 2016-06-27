
import itertools

import numpy

from variation.matrix.methods import is_missing

from scipy.spatial.distance import squareform
from variation.variations.stats import GT_FIELD


def _get_sample_gts(gts, sample_i, sample_j, indi_cache):
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

    return indi1, indi2


def _gt_matching(gts, sample_i, sample_j, indi_cache):
    '''This is a very naive distance

    It is implemented just because we were asked to do it in a project
    '''
    indi1, indi2 = _get_sample_gts(gts, sample_i, sample_j, indi_cache)
    comparison1 = numpy.sum(indi1 == indi2, axis=1) == 2

    # now we allow the possibility of non matching phases 1, 0 vs 0, 1
    # we're assuming a ploidy of 2
    if indi1.shape[1] != 2:
        raise NotImplementedError('We assume diploidy')
    comparison2 = numpy.sum(indi1[:, ::-1] == indi2, axis=1) == 2

    result = numpy.logical_or(comparison1, comparison2)
    return numpy.sum(result), result.shape[0]


def _matching(gts, sample_i, sample_j, indi_cache):
    matching_gts, tot_gts = _gt_matching(gts, sample_i, sample_j, indi_cache)
    if tot_gts:
        result = 1 - matching_gts / tot_gts
    else:
        result = None
    return result


def _calc_matching_pairwise_distance_by_chunk(variations, chunk_size):

    matching_snps, tot_snps = 0, 0
    for chunk in variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=chunk_size):
        indi_cache = {}
        res = _indi_pairwise_dist(chunk, indi_cache, method='matching')
        if res is None:
            continue
        matching_snps, tot_snps = res
        matching_snps += res[0]
        tot_snps += res[1]

    if tot_snps:
        return matching_snps / tot_snps
    else:
        return None


def sel_samples_from_dist_mat(distance_matrix, sample_idxs):
    selected = squareform(distance_matrix)[sample_idxs][:, sample_idxs]
    return squareform(selected)


def _kosman(gts, sample_i, sample_j, indi_cache):
    '''It calculates the distance between two individuals using the Kosman dist

    The Kosman distance is explain in DOI: 10.1111/j.1365-294X.2005.02416.x
    '''

    indi1, indi2 = _get_sample_gts(gts, sample_i, sample_j, indi_cache)

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


_PAIRWISE_DISTANCES = {'kosman': _kosman, 'matching': _gt_matching}


def _calc_matching_pairwise_distance(variations, chunk_size=None):
    '''This is a very naive distance

    It is implemented just because we were asked to do it in a project
    '''

    abs_distances, n_snps_matrix = None, None
    for chunk in variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=chunk_size):
        indi_cache = {}
        res = _indi_pairwise_dist(chunk, indi_cache, method='matching')
        chunk_abs_distances, n_snps_chunk = res
        if abs_distances is None and n_snps_matrix is None:
            abs_distances = chunk_abs_distances.copy()
            n_snps_matrix = n_snps_chunk
        else:
            abs_distances = numpy.add(abs_distances, chunk_abs_distances)
            n_snps_matrix = numpy.add(n_snps_matrix, n_snps_chunk)
    with numpy.errstate(invalid='ignore'):
        return abs_distances / n_snps_matrix


def _indi_pairwise_dist(variations, indi_cache, method='kosman'):
    gts = variations[GT_FIELD]
    n_samples = gts.shape[1]
    dists = numpy.zeros(int((n_samples ** 2 - n_samples) / 2))
    n_snps_matrix = numpy.zeros(int((n_samples ** 2 - n_samples) / 2))
    index = 0
    dist_funct = _PAIRWISE_DISTANCES[method]
    for sample_i, sample_j in itertools.combinations(range(n_samples), 2):
        dist, n_snps = dist_funct(gts, sample_i, sample_j, indi_cache)
        dists[index] = dist
        n_snps_matrix[index] = n_snps
        index += 1
    return dists, n_snps_matrix


def _calc_kosman_pairwise_distance_by_chunk(variations, chunk_size):

    abs_distances, n_snps_matrix = None, None
    for chunk in variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=chunk_size):
        indi_cache = {}
        res = _indi_pairwise_dist(chunk, indi_cache, method='kosman')
        chunk_abs_distances, n_snps_chunk = res
        if abs_distances is None and n_snps_matrix is None:
            abs_distances = chunk_abs_distances.copy()
            n_snps_matrix = n_snps_chunk
        else:
            abs_distances = numpy.add(abs_distances, chunk_abs_distances)
            n_snps_matrix = numpy.add(n_snps_matrix, n_snps_chunk)
    with numpy.errstate(invalid='ignore'):
        return abs_distances / n_snps_matrix


def _calc_kosman_pairwise_distance(variations, chunk_size=None):
    '''It calculates the distance between individuals using the Kosman
    distance.

    The Kosman distance is explain in DOI: 10.1111/j.1365-294X.2005.02416.x
    '''
    if chunk_size:
        distance = _calc_kosman_pairwise_distance_by_chunk(variations,
                                                           chunk_size)
    else:
        abs_dist, n_snps = _indi_pairwise_dist(variations, {},
                                               method='kosman')
        with numpy.errstate(invalid='ignore'):
            distance = abs_dist / n_snps
    return distance


DISTANCES = {'kosman': _calc_kosman_pairwise_distance,
             'matching': _calc_matching_pairwise_distance}


def calc_pairwise_distance(variations, chunk_size=None, method='kosman'):
    return DISTANCES[method](variations, chunk_size=chunk_size)
