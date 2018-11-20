
import numpy
import math
import itertools

from variation import MISSING_INT, SNPS_PER_CHUNK, POS_FIELD, CHROM_FIELD


def _bivmom(vec0, vec1):
    """
    Calculate means, variances, the covariance, from two data vectors.
    On entry, vec0 and vec1 should be vectors of numeric values and
    should have the same length.  Function returns m0, v0, m1, v1,
    cov, where m0 and m1 are the means of vec0 and vec1, v0 and v1 are
    the variances, and cov is the covariance.
    """
    m0 = m1 = v0 = v1 = cov = 0
    for x, y in zip(vec0, vec1):
        m0 += x
        m1 += y
        v0 += x * x
        v1 += y * y
        cov += x * y
    n = len(vec0)
    assert n == len(vec1)
    n = float(n)
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n

    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    return m0, v0, m1, v1, cov


def _get_r(Y, Z):
    """
    Estimates r w/o info on gametic phase.  Also works with gametic
    data, in which case Y and Z should be vectors of 0/1 indicator
    variables.
    Uses the method of Rogers and Huff 2008.
    """
    mY, vY, mZ, vZ, cov = _bivmom(Y, Z)
    return cov / math.sqrt(vY * vZ)


def _calc_rogers_huff_r_for_snp_pair(gts_snp1, gts_snp2, min_num_gts=10,
                                     debug=False):

    gts = numpy.array([gts_snp1, gts_snp2])

    rows_with_no_missing = numpy.logical_not((gts == MISSING_INT).any(axis=0))
    gts = gts[:, rows_with_no_missing]
    if gts.shape[1] < min_num_gts:
        result = numpy.nan
    else:
        covar = numpy.cov(gts, ddof=0)
        variances = numpy.diag(covar)
        covar = covar[0, 1]
        result = covar / numpy.sqrt(variances[0] * variances[1])
    return result


def _calc_rogers_huff_r(gts, debug=False):
    # means = numpy.nanmean(gts, axis=1)
    # var = numpy.nanvar(gts, axis=1)
    covar = numpy.cov(gts, ddof=0)
    variances = numpy.diag(covar)
    covar_indices = numpy.tril_indices(covar.shape[0], -1)
    covars = covar[covar_indices]
    if debug:
        print(covar)
        print('vars:', variances)
        print(covar_indices)
        print('covars:', covars)
    vars1 = variances[covar_indices[0]]
    vars2 = variances[covar_indices[1]]
    rogers_huff_r = covars / numpy.sqrt(vars1 * vars2)
    if debug:
        print('r', rogers_huff_r)
    return rogers_huff_r


def _calc_rogers_huff_r2_no_nans(gts1, gts2, debug=False):
    # means = numpy.nanmean(gts, axis=1)
    # var = numpy.nanvar(gts, axis=1)

    covars = numpy.cov(gts1, gts2, ddof=0)
    n_vars1 = gts1.shape[0]
    n_vars2 = gts2.shape[0]
    if debug:
        print('nvars', n_vars1, n_vars2)
    variances = numpy.diag(covars)
    vars1 = variances[:n_vars1]
    vars2 = variances[n_vars1:]
    if debug:
        print('vars1', vars1)
        print('vars2', vars2)

    covars = covars[:n_vars1, n_vars1:]
    if debug:
        print('covars', covars)

    vars1 = numpy.repeat(vars1, n_vars2).reshape((n_vars1, n_vars2))
    vars2 = numpy.tile(vars2, n_vars1).reshape((n_vars1, n_vars2))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        rogers_huff_r = covars / numpy.sqrt(vars1 * vars2)
    # print(vars1)
    # print(vars2)
    return rogers_huff_r


def calc_rogers_huff_r(gts1, gts2, min_num_gts=10, debug=False):
    if not (numpy.any(gts1 == MISSING_INT) or numpy.any(gts2 == MISSING_INT)):
        return _calc_rogers_huff_r2_no_nans(gts1, gts2, debug=debug)

    rogers_huff_r = numpy.empty((gts1.shape[0], gts2.shape[0]),
                                dtype=numpy.float16)
    for idx1, gts1_snp_gts in enumerate(gts1):
        for idx2, gts2_snp_gts in enumerate(gts2):
            result = _calc_rogers_huff_r_for_snp_pair(gts1_snp_gts,
                                                      gts2_snp_gts,
                                                      min_num_gts=min_num_gts,
                                                      debug=debug)
            rogers_huff_r[idx1, idx2] = result
    return rogers_huff_r


def _calc_ld_between_chunks(chunk_pair, min_num_gts=10):
    chunk1 = chunk_pair['chunk1']
    chunk2 = chunk_pair['chunk2']
    lds_for_pair = calc_rogers_huff_r(chunk1.gts_as_mat012,
                                      chunk2.gts_as_mat012,
                                      min_num_gts=min_num_gts)
    pos1 = chunk1[POS_FIELD]
    pos2 = chunk2[POS_FIELD]

    pos1_repeated = numpy.repeat(pos1, pos2.size).reshape((pos1.size, pos2.size))
    pos2_repeated = numpy.tile(pos2, pos1.size).reshape((pos1.size, pos2.size))
    physical_dist = numpy.abs(pos1_repeated - pos2_repeated).astype(float)
    assert lds_for_pair.shape == physical_dist.shape

    chrom1 = chunk1[CHROM_FIELD]
    chrom2 = chunk2[CHROM_FIELD]
    chrom1_repeated = numpy.repeat(chrom1, chrom2.size).reshape((chrom1.size, chrom2.size))
    chrom2_repeated = numpy.tile(chrom2, chrom1.size).reshape((chrom1.size, chrom2.size))

    physical_dist[chrom1_repeated != chrom2_repeated] = numpy.nan

    yield zip(lds_for_pair.flat, physical_dist.flat)


def calc_ld_along_genome(variations, max_dist, min_num_gts=10,
                         chunk_size=SNPS_PER_CHUNK):
    chunk_pairs = variations.iterate_chunk_pairs(max_dist=max_dist,
                                                 chunk_size=chunk_size)
    for result in itertools.chain.from_iterable(_calc_ld_between_chunks(chunk_pair, min_num_gts=min_num_gts) for chunk_pair in chunk_pairs):
        for ld, physical_dist in result:
            yield ld, physical_dist


