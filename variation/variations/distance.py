
import itertools
import math

import numpy
from pandas import DataFrame

from scipy.spatial.distance import squareform

from variation.matrix.methods import is_missing
from variation.variations.stats import (calc_allele_freq,
                                        calc_allele_freq_by_depth)
from variation.variations.filters import SampleFilter, FLT_VARS
from variation import GT_FIELD, MIN_NUM_GENOTYPES_FOR_POP_STAT, MISSING_INT


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


def _calc_matching_pairwise_distance(variations, chunk_size=None,
                                     min_num_snps=None):
    '''This is a very naive distance

    It is implemented just because we were asked to do it in a project
    '''

    if min_num_snps is not None:
        raise NotImplemented("It's easy, really, just copy the kosman implementation")

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


def _calc_kosman_pairwise_distance_by_chunk(variations, chunk_size,
                                            min_num_snps=None):

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

    if min_num_snps:
        n_snps_matrix[n_snps_matrix < min_num_snps] = numpy.nan

    with numpy.errstate(invalid='ignore'):
        return abs_distances / n_snps_matrix


def _calc_kosman_pairwise_distance(variations, chunk_size=None,
                                   min_num_snps=None):
    '''It calculates the distance between individuals using the Kosman
    distance.

    The Kosman distance is explain in DOI: 10.1111/j.1365-294X.2005.02416.x
    '''
    if chunk_size:
        distance = _calc_kosman_pairwise_distance_by_chunk(variations,
                                                           chunk_size,
                                                           min_num_snps=min_num_snps)
    else:
        abs_dist, n_snps = _indi_pairwise_dist(variations, {},
                                               method='kosman')
        if min_num_snps is not None:
            n_snps[n_snps < min_num_snps] = numpy.nan
        with numpy.errstate(invalid='ignore'):
            distance = abs_dist / n_snps
    return distance


def _calc_nei_pop_distance(variations, populations, chunk_size=None,
                           min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks(kept_fields=[GT_FIELD])

    pop_flts = [SampleFilter(pop) for pop in populations]

    jxy = {}
    jxx = {}
    jyy = {}
    for chunk in chunks:
        alleles = sorted(numpy.unique(chunk[GT_FIELD]))
        for pop_i, pop_j in itertools.combinations(range(len(populations)), 2):
            chunk_pop_i = pop_flts[pop_i](chunk)[FLT_VARS]
            chunk_pop_j = pop_flts[pop_j](chunk)[FLT_VARS]

            freq_al_i = calc_allele_freq(chunk_pop_i, alleles=alleles,
                                         min_num_genotypes=min_num_genotypes)
            freq_al_j = calc_allele_freq(chunk_pop_j, alleles=alleles,
                                         min_num_genotypes=min_num_genotypes)

            chunk_jxy = numpy.nansum(freq_al_i * freq_al_j)
            chunk_jxx = numpy.nansum(freq_al_i ** 2)
            chunk_jyy = numpy.nansum(freq_al_j ** 2)

            pop_idx = pop_i, pop_j
            if pop_idx not in jxy:
                jxy[pop_idx] = 0
                jxx[pop_idx] = 0
                jyy[pop_idx] = 0

            # The real Jxy is usually divided by num_snps, but it does not
            # not matter for the calculation
            jxy[pop_idx] += chunk_jxy
            jxx[pop_idx] += chunk_jxx
            jyy[pop_idx] += chunk_jyy
            # print(freq_al_i)
            # print(freq_al_j)
            # print(chunk_jxy, chunk_jxx, chunk_jyy)

    n_pops = len(populations)
    dists = numpy.zeros(int((n_pops ** 2 - n_pops) / 2))
    index = 0
    for pop_idx in itertools.combinations(range(len(populations)), 2):
        pjxy = jxy[pop_idx]
        pjxx = jxx[pop_idx]
        pjyy = jyy[pop_idx]

        try:
            nei = math.log(pjxy / math.sqrt(pjxx * pjyy))
            if nei != 0:
                nei = -nei
        except ValueError:
            nei = float('inf')

        dists[index] = nei
        index += 1

    return dists


def calc_gst_per_loci(variations, populations, chunk_size=None,
                      min_num_genotypes=None):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks()

    gst = numpy.array([])
    for chunk in chunks:
        het_exp_by_pop = []
        allele_freqs_by_pop = []
        for pop in populations:
            chunk_pop = SampleFilter(pop)(chunk)[FLT_VARS]
            allele_freq = calc_allele_freq_by_depth(chunk_pop)
            het_exp = 1 - numpy.sum(allele_freq ** 2, axis=1)
            het_exp_by_pop.append(het_exp)
            allele_freqs_by_pop.append(allele_freq)

        hs = numpy.sum(het_exp_by_pop, axis=0) / len(het_exp_by_pop)
        allele_freq_averages = numpy.sum(allele_freqs_by_pop, axis=0) / len(allele_freqs_by_pop)
        ht = 1 - numpy.sum(allele_freq_averages ** 2, axis=1)

        with numpy.errstate(invalid='ignore'):
            gst_ = (ht - hs) / ht
        gst_ = numpy.nan_to_num(gst_)
        gst = numpy.append(gst, gst_)

        # print(hs[:10])
        # print(ht[:10])
#         print(gst)
    return gst


def _calc_nei_pop_distance_by_depth(variations, populations, chunk_size=None,
                                    min_num_genotypes=None):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks()

    pop_flts = [SampleFilter(pop) for pop in populations]

    jxy = {}
    jxx = {}
    jyy = {}
    for chunk in chunks:
        for pop_i, pop_j in itertools.combinations(range(len(populations)), 2):
            chunk_pop_i = pop_flts[pop_i](chunk)[FLT_VARS]
            chunk_pop_j = pop_flts[pop_j](chunk)[FLT_VARS]

            freq_al_i = calc_allele_freq_by_depth(chunk_pop_i)
            freq_al_j = calc_allele_freq_by_depth(chunk_pop_j)

            chunk_jxy = numpy.nansum(freq_al_i * freq_al_j)
            chunk_jxx = numpy.nansum(freq_al_i ** 2)
            chunk_jyy = numpy.nansum(freq_al_j ** 2)

            pop_idx = pop_i, pop_j
            if pop_idx not in jxy:
                jxy[pop_idx] = 0
                jxx[pop_idx] = 0
                jyy[pop_idx] = 0

            # The real Jxy is usually divided by num_snps, but it does not
            # not matter for the calculation
            jxy[pop_idx] += chunk_jxy
            jxx[pop_idx] += chunk_jxx
            jyy[pop_idx] += chunk_jyy
            # print(freq_al_i)
            # print(freq_al_j)
            # print(chunk_jxy, chunk_jxx, chunk_jyy)

    n_pops = len(populations)
    dists = numpy.zeros(int((n_pops ** 2 - n_pops) / 2))
    index = 0
    for pop_idx in itertools.combinations(range(len(populations)), 2):
        pjxy = jxy[pop_idx]
        pjxx = jxx[pop_idx]
        pjyy = jyy[pop_idx]

        try:
            nei = math.log(pjxy / math.sqrt(pjxx * pjyy))
            if nei != 0:
                nei = -nei
        except ValueError:
            nei = float('inf')

        dists[index] = nei
        index += 1

    return dists


def _get_different_alleles(variations):
    different_alleles = set(numpy.unique(variations[GT_FIELD]))
    different_alleles = sorted(different_alleles.difference([MISSING_INT]))
    return different_alleles


def _calc_allele_freq_and_unbiased_J_per_locus(variations, alleles,
                                               min_num_genotypes):
    try:
        allele_freq = calc_allele_freq(variations, alleles=alleles,
                                       min_num_genotypes=min_num_genotypes)
    except ValueError:
        allele_freq = None
        xUb_per_locus = None

    if allele_freq is not None:
        n_indi = variations[GT_FIELD].shape[1]
        xUb_per_locus = ((2 * n_indi * numpy.sum(allele_freq ** 2, axis=1)) - 1) / (2 * n_indi - 1)

    return allele_freq, xUb_per_locus


def _calc_j_stats_per_locus(variations1, variations2,
                            min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    different_alleles = set(numpy.unique(variations1[GT_FIELD]))
    different_alleles.update(numpy.unique(variations2[GT_FIELD]))
    different_alleles = sorted(different_alleles.difference([MISSING_INT]))

    if not different_alleles:
        return None, None, None

    res = _calc_allele_freq_and_unbiased_J_per_locus(variations1,
                                                     alleles=different_alleles,
                                                     min_num_genotypes=min_num_genotypes)
    allele_freq1, xUb_per_locus = res

    res = _calc_allele_freq_and_unbiased_J_per_locus(variations2,
                                                     alleles=different_alleles,
                                                     min_num_genotypes=min_num_genotypes)
    allele_freq2, yUb_per_locus = res

    if allele_freq1 is None or allele_freq2 is None:
        return None, None, None

    Jxy_per_locus = numpy.sum(allele_freq1 * allele_freq2, axis=1)

    return xUb_per_locus, yUb_per_locus, Jxy_per_locus


def _accumulate_j_stats(variations1, variations2,
                        Jxy, uJx, uJy, pop_name1, pop_name2,
                        min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    res = _calc_j_stats_per_locus(variations1, variations2,
                                  min_num_genotypes=min_num_genotypes)
    xUb_per_locus, yUb_per_locus, Jxy_per_locus = res
    # print('per locus')
    # print(xUb_per_locus, yUb_per_locus, Jxy_per_locus)

    if xUb_per_locus is None:
        return

    # sum over all loci
    if Jxy[pop_name1][pop_name2] is None:
        Jxy[pop_name1][pop_name2] = numpy.nansum(Jxy_per_locus)
        uJx[pop_name1][pop_name2] = numpy.nansum(xUb_per_locus)
        uJy[pop_name1][pop_name2] = numpy.nansum(yUb_per_locus)
    else:
        Jxy[pop_name1][pop_name2] += numpy.nansum(Jxy_per_locus)
        uJx[pop_name1][pop_name2] += numpy.nansum(xUb_per_locus)
        uJy[pop_name1][pop_name2] += numpy.nansum(yUb_per_locus)


def _calc_pop_pairwise_unbiased_nei_dists(variations, populations=None,
                                          min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                          chunk_size=None):

    pop_sample_filters = [SampleFilter(pop_samples) for pop_samples in populations]
    pop_ids = list(range(len(populations)))

    Jxy = {}
    uJx = {}
    uJy = {}
    for pop_id1, pop_id2 in itertools.combinations(pop_ids, 2):
        if pop_id1 not in Jxy:
            Jxy[pop_id1] = {}
        if pop_id1 not in uJx:
            uJx[pop_id1] = {}
        if pop_id1 not in uJy:
            uJy[pop_id1] = {}

        Jxy[pop_id1][pop_id2] = None
        uJx[pop_id1][pop_id2] = None
        uJy[pop_id1][pop_id2] = None

    chunks = variations.iterate_chunks(kept_fields=[GT_FIELD],
                                       chunk_size=chunk_size)

    for chunk in chunks:
        for pop_id1, pop_id2 in itertools.combinations(pop_ids, 2):
            vars_for_pop1 = pop_sample_filters[pop_id1](chunk)[FLT_VARS]
            vars_for_pop2 = pop_sample_filters[pop_id2](chunk)[FLT_VARS]
            _accumulate_j_stats(vars_for_pop1, vars_for_pop2, Jxy, uJx, uJy,
                                pop_id1, pop_id2,
                                min_num_genotypes=min_num_genotypes)

    n_pops = len(populations)
    dists = numpy.empty(int((n_pops ** 2 - n_pops) / 2))
    dists[:] = numpy.nan
    for idx, (pop_id1, pop_id2) in enumerate(itertools.combinations(pop_ids, 2)):
        if Jxy[pop_id1][pop_id2] is None:
            unbiased_nei_identity = math.nan
        else:
            with numpy.errstate(invalid='ignore'):
                unbiased_nei_identity = Jxy[pop_id1][pop_id2] / math.sqrt(uJx[pop_id1][pop_id2] * uJy[pop_id1][pop_id2])
        nei_unbiased_distance = -math.log(unbiased_nei_identity)
        if nei_unbiased_distance < 0:
            nei_unbiased_distance = 0
        dists[idx] = nei_unbiased_distance
    return dists


DISTANCES = {'kosman': _calc_kosman_pairwise_distance,
             'matching': _calc_matching_pairwise_distance,
             'nei': _calc_nei_pop_distance,
             'nei_unbiased': _calc_pop_pairwise_unbiased_nei_dists,
             'nei_depth': _calc_nei_pop_distance_by_depth,
             'gst_per_loci': calc_gst_per_loci}


def calc_pairwise_distance(variations, chunk_size=None, method='kosman',
                           min_num_snps=None):
    return DISTANCES[method](variations, chunk_size=chunk_size,
                             min_num_snps=min_num_snps)


def calc_pop_distance(variations, populations, method, chunk_size=None,
                      min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    return DISTANCES[method](variations, populations, chunk_size=chunk_size,
                             min_num_genotypes=min_num_genotypes)


def filter_dist_matrix(dists, idxs_to_keep, squareform_checks=True):
    dists = squareform(dists, checks=squareform_checks)
    dists = dists[:, idxs_to_keep][idxs_to_keep, :]
    dists = squareform(dists, checks=squareform_checks)
    return dists


def triangular_dists_to_square(dists, col_names):
    return DataFrame(squareform(dists), index=col_names, columns=col_names)
