
from functools import reduce
from itertools import combinations_with_replacement, permutations
import operator

import numpy
from scipy.stats.stats import chisquare

from variation import MISSING_VALUES, SNPS_PER_CHUNK, DEF_MIN_DEPTH
from variation.matrix.stats import counts_by_row
from variation.matrix.methods import (is_missing, fill_array, calc_min_max,
                                      is_dataset, iterate_matrix_chunks)
from variation.utils.misc import remove_nans
from variation.variations.index import PosIndex


MIN_NUM_GENOTYPES_FOR_POP_STAT = 10
DEF_NUM_BINS = 20
GT_FIELD = '/calls/GT'
GQ_FIELD = '/calls/GQ'
ALT_FIELD = '/variations/alt'
REF_OBS = '/calls/RO'
ALT_OBS = '/calls/AO'
DP_FIELD = '/calls/DP'
AO_FIELD = '/calls/AO'
RO_FIELD = '/calls/RO'
CHROM_FIELD = '/variations/chrom'
POS_FIELD = '/variations/pos'


REQUIRED_FIELDS_FOR_STAT = {'calc_maf': [GT_FIELD],
                            'calc_allele_freq': [GT_FIELD, ALT_FIELD],
                            'calc_hwe_chi2_test': [GT_FIELD, ALT_FIELD],
                            'calc_called_gts_distrib_per_depth': [DP_FIELD,
                                                                  GT_FIELD],
                            'calc_missing_gt': [GT_FIELD],
                            'calc_obs_het': [GT_FIELD],
                            'calc_obs_het_by_sample': [GT_FIELD]}


def _calc_histogram(vector, n_bins, range_):
    vector = remove_nans(vector)
    return numpy.histogram(vector, bins=n_bins, range=range_)


def histogram(vector, n_bins=DEF_NUM_BINS, range_=None):
    return _calc_histogram(vector, n_bins, range_)


def calc_cum_distrib(distrib):
    if len(distrib.shape) == 1:
        return numpy.fliplr(numpy.cumsum(numpy.fliplr([distrib]), axis=1))[0]
    else:
        return numpy.fliplr(numpy.cumsum(numpy.fliplr(distrib), axis=1))


def _guess_stat_funct_called(calc_funct):
    if 'func' in dir(calc_funct):
        funct_name = calc_funct.func.__name__
    else:
        funct_name = calc_funct.__name__
    return funct_name


def _calc_stats_for_chunks(calc_funct, variations, chunk_size):
    funct_name = _guess_stat_funct_called(calc_funct)
    req_fields = REQUIRED_FIELDS_FOR_STAT[funct_name]
    vectors = (calc_funct(chunk)
               for chunk in variations.iterate_chunks(kept_fields=req_fields,
                                                      chunk_size=chunk_size))
    return vectors


def histogram_for_chunks(variations, calc_funct, n_bins=DEF_NUM_BINS,
                         range_=None, chunk_size=None):
    if range_ is None:
        range_ = _calc_range(_calc_stats_for_chunks(calc_funct, variations,
                                                    chunk_size))

    hist = None
    for stat in _calc_stats_for_chunks(calc_funct, variations, chunk_size):
        stat_hist, bins = histogram(stat, n_bins, range_)
        if hist is None:
            hist = stat_hist
        else:
            hist += stat_hist
    return hist, bins


def _calc_range(vectors):
    min_ = None
    max_ = None
    for stat in vectors:
        stat_max = numpy.max(stat)
        stat_min = numpy.min(stat)
        if min_ is None or min_ > stat_min:
            min_ = stat_min
        if max_ is None or max_ < stat_max:
            max_ = stat_max
    range_ = min_, max_
    return range_


def _calc_maf(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    gts = variations[GT_FIELD]
    gt_counts = counts_by_row(gts, missing_value=MISSING_VALUES[int])
    max_ = numpy.amax(gt_counts, axis=1)
    sum_ = numpy.sum(gt_counts, axis=1)

    # To avoid problems with NaNs
    with numpy.errstate(invalid='ignore'):
        mafs_gt = max_ / sum_
    return _mask_stats_with_few_samples(mafs_gt, variations, min_num_genotypes)


def _calc_maf_by_chunk(variations,
                       min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                       chunk_size=SNPS_PER_CHUNK):
    mafs = None
    for chunk in variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=chunk_size):
        chunk_maf = _calc_maf(chunk, min_num_genotypes=min_num_genotypes)
        if mafs is None:
            mafs = chunk_maf
        else:
            mafs = numpy.append(mafs, chunk_maf)
    return mafs


def calc_maf(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
             chunk_size=SNPS_PER_CHUNK):
    if chunk_size is None:
        return _calc_maf(variations, min_num_genotypes=min_num_genotypes)
    else:
        return _calc_maf_by_chunk(variations, min_num_genotypes, chunk_size)


def calc_depth(variations):
    mat = variations[DP_FIELD]
    shape = mat.shape
    if is_dataset(mat):
        mat = mat[:]
    return mat.reshape(shape[0] * shape[1])


def calc_genotype_qual(variations):
    mat = variations[GQ_FIELD]
    shape = mat.shape
    if is_dataset(mat):
        mat = mat[:]
    return mat.reshape(shape[0] * shape[1])


def _is_called(gts):
    return numpy.logical_not(is_missing(gts, axis=2))


def _is_ge_dp(variations, depth):
    dps = variations[DP_FIELD]
    if is_dataset(dps):
        dps = dps[:]
    return dps >= depth


def _calc_gt_mask_by_depth(variations, depth):
    is_called = _is_called(variations[GT_FIELD])
    is_eq_hi_dp = _is_ge_dp(variations, depth)
    is_called_eq_hi_than_dp = numpy.logical_and(is_called, is_eq_hi_dp)
    return is_called_eq_hi_than_dp


def calc_num_samples_called_distrib(variations, depth):
    n_samples = len(variations.samples)
    n_bins = n_samples
    range_ = 0, n_samples + 1

    is_called_higher_depth = _calc_gt_mask_by_depth(variations, depth)
    n_called_per_snp = numpy.sum(is_called_higher_depth, axis=1)
    return histogram(n_called_per_snp, range_=range_, n_bins=n_bins)


def _calc_called_gts_distrib_per_depth(variations, depths):
    distributions = None
    for depth in depths:
        distrib, bins = calc_num_samples_called_distrib(variations, depth)
        if distributions is None:
            distributions = distrib
        else:
            distributions = numpy.vstack((distributions, distrib))
    return distributions, bins


def _calc_called_gts_distrib_per_depth_by_chunk(variations, depths,
                                                chunk_size):
    distributions = None
    req_fields = REQUIRED_FIELDS_FOR_STAT['calc_called_gts_distrib_per_depth']
    for chunk in variations.iterate_chunks(kept_fields=req_fields,
                                           chunk_size=chunk_size):
        chunk_distribs = None
        for depth in depths:
            chunk_distrib, bins = calc_num_samples_called_distrib(chunk,
                                                                  depth)
            if chunk_distribs is None:
                chunk_distribs = chunk_distrib
            else:
                chunk_distribs = numpy.vstack((chunk_distribs, chunk_distrib))
        if distributions is None:
            distributions = chunk_distribs
        else:
            distributions = numpy.add(distributions, chunk_distribs)

    return distributions, bins


def calc_called_gts_distrib_per_depth(variations, depths,
                                      chunk_size=SNPS_PER_CHUNK):
    if chunk_size:
        return _calc_called_gts_distrib_per_depth_by_chunk(variations, depths,
                                                           chunk_size)
    else:
        return _calc_called_gts_distrib_per_depth(variations, depths)


def _calc_items_in_row(mat):
    num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
    return num_items_per_row


def calc_missing_gt(variations, rates=True, axis=1):
    gts = variations[GT_FIELD]
    if is_dataset(gts):
        gts = gts[:]
    missing = numpy.any(gts == MISSING_VALUES[int], axis=2).sum(axis=axis)
    if rates:
        num_items_per_row = gts.shape[axis]
        result = missing / num_items_per_row
    else:
        result = missing
    return result


def calc_called_gt(variations, rates=True, axis=1):
    missing = calc_missing_gt(variations, rates=rates, axis=axis)
    if rates:
        return 1 - missing
    else:
        total = variations[GT_FIELD].shape[axis]
        return total - missing


def call_is_het(gts):
    is_het = numpy.logical_not(call_is_hom(gts))
    missing_gts = is_missing(gts, axis=2)
    is_het[missing_gts] = False
    return is_het


def call_is_hom(gts):
    is_hom = numpy.full(gts.shape[:-1], True, dtype=numpy.bool)
    for idx in range(1, gts.shape[2]):
        is_hom = numpy.logical_and(gts[:, :, idx] == gts[:, :, idx - 1],
                                   is_hom)
    missing_gts = is_missing(gts, axis=2)
    is_hom[missing_gts] = False
    return is_hom


def call_is_hom_ref(gts):
    return numpy.logical_and(call_is_hom(gts), gts[:, :, 0] == 0)


def call_is_hom_alt(gts):
    return numpy.logical_and(call_is_hom(gts), gts[:, :, 0] != 0)


def _calc_obs_het_counts(variations, axis):
    gts = variations[GT_FIELD]
    if is_dataset(gts):
        gts = gts[:]
    is_het = call_is_het(gts)
    return numpy.sum(is_het, axis=axis)


def _mask_stats_with_few_samples(stats, variations, min_num_genotypes,
                                 num_called_gts=None):
    if min_num_genotypes:
        if num_called_gts is None:
            num_called_gts = calc_called_gt(variations, rates=False)
        stats[num_called_gts < min_num_genotypes] = numpy.NaN
    return stats


def calc_obs_het(variations,
                 min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    het = _calc_obs_het_counts(variations, axis=1)
    called_gts = calc_called_gt(variations, rates=False)
    # To avoid problems with NaNs
    with numpy.errstate(invalid='ignore'):
        het = het / called_gts
    return _mask_stats_with_few_samples(het, variations, min_num_genotypes,
                                        num_called_gts=called_gts)


def _calc_obs_het_by_sample(variations):
    return _calc_obs_het_counts(variations, axis=0)


def calc_obs_het_by_sample(variations, chunk_size=SNPS_PER_CHUNK):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=chunk_size)
    obs_het_by_sample = None
    called_gts = None
    for chunk in chunks:
        chunk_obs_het_by_sample = _calc_obs_het_by_sample(chunk)
        chunk_called_gts = calc_called_gt(chunk, rates=False,
                                          axis=0)
        if called_gts is None:
            obs_het_by_sample = chunk_obs_het_by_sample
            called_gts = chunk_called_gts
        else:
            obs_het_by_sample += chunk_obs_het_by_sample
            called_gts += chunk_called_gts
    with numpy.errstate(invalid='ignore'):
        obs_het_by_sample = obs_het_by_sample / called_gts
    return obs_het_by_sample


def _calc_gt_type_stats(variations):
    gts = variations[GT_FIELD]
    het = numpy.sum(call_is_het(gts), axis=0)
    missing = numpy.sum(is_missing(gts, axis=2), axis=0)
    gts_alt = numpy.copy(gts)
    gts_alt[gts_alt == 0] = -1
    alt_hom = numpy.sum(call_is_hom(gts_alt), axis=0)
    ref_hom = numpy.sum(call_is_hom(gts), axis=0) - alt_hom
    return numpy.array([ref_hom, het, alt_hom, missing])


def calc_gt_type_stats(variations, chunk_size=None):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks(kept_fields=[GT_FIELD],
                                           chunk_size=chunk_size)
    gt_type_stats = None
    for chunk in chunks:
        chunk_stats = _calc_gt_type_stats(chunk)
        if gt_type_stats is None:
            gt_type_stats = chunk_stats
        else:
            gt_type_stats += chunk_stats
    return gt_type_stats


def calc_snp_density(variations, window, chunk_size=SNPS_PER_CHUNK):
    dens = []
    dic_index = PosIndex(variations)
    n_snps = variations[CHROM_FIELD].shape[0]
    chromosomes = variations[CHROM_FIELD]
    positions = variations[POS_FIELD]
    for i in range(0, n_snps, chunk_size):
        chunk_chrom = chromosomes[i: i + chunk_size]
        chunk_pos = positions[i: i + chunk_size]
        for chunk_idx in range(chunk_chrom.shape[0]):
            chrom = chunk_chrom[chunk_idx]
            pos = chunk_pos[chunk_idx]
            pos_right = window + pos
            pos_left = pos - window
            index_right = dic_index.index_pos(chrom, pos_right)
            index_left = dic_index.index_pos(chrom, pos_left)
            dens.append(index_right - index_left)
    return numpy.array(dens)


def _calc_allele_counts(gts):
    return counts_by_row(gts, MISSING_VALUES[int])


def calc_allele_freq(variations,
                     min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]
    max_num_allele = variations[ALT_FIELD].shape[1] + 1
    allele_counts = _calc_allele_counts(gts)
    total_counts = numpy.sum(allele_counts, axis=1)
    allele_freq = allele_counts / total_counts[:, None]
    if allele_freq.shape[1] < max_num_allele:
        allele_freq = fill_array(allele_freq, max_num_allele, dim=1)
    _mask_stats_with_few_samples(allele_freq, variations, min_num_genotypes)
    return allele_freq


def _calc_expected_het(variations,
                       min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    allele_freq = calc_allele_freq(variations,
                                   min_num_genotypes=min_num_genotypes)
    ploidy = variations[GT_FIELD].shape[2]
    return 1 - numpy.sum(allele_freq ** ploidy, axis=1)


def calc_expected_het(variations,
                      min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                      chunk_size=SNPS_PER_CHUNK):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks(kept_fields=[GT_FIELD, ALT_FIELD],
                                           chunk_size=chunk_size)
    exp_het = None
    for chunk in chunks:
        chunk_exp_het = _calc_expected_het(chunk, min_num_genotypes)
        if exp_het is None:
            exp_het = chunk_exp_het
        else:
            exp_het = numpy.append(exp_het, chunk_exp_het)
    return exp_het


def _calc_inbreeding_coef(variations,
                          min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    obs_het = calc_obs_het(variations, min_num_genotypes=min_num_genotypes)
    exp_het = _calc_expected_het(variations,
                                 min_num_genotypes=min_num_genotypes)
    with numpy.errstate(invalid='ignore'):
        inbreed = 1 - (obs_het / exp_het)
    return inbreed


def calc_inbreeding_coef(variations, chunk_size=SNPS_PER_CHUNK,
                         min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks(kept_fields=[GT_FIELD, ALT_FIELD],
                                           chunk_size=chunk_size)
    inbreed_coef = None
    for chunk in chunks:
        chunk_inbreed_coef = _calc_inbreeding_coef(chunk,
                                                   min_num_genotypes)
        if inbreed_coef is None:
            inbreed_coef = chunk_inbreed_coef
        else:
            inbreed_coef = numpy.append(inbreed_coef, chunk_inbreed_coef)
    return inbreed_coef


def _calc_hwe_chi2_test(variations, num_allele,
                        min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    ploidy = variations.ploidy
    gts = variations[GT_FIELD]

    allele_freq = calc_allele_freq(variations,
                                   min_num_genotypes=min_num_genotypes)
    # Select vars with a certain number of alleles
    sel_vars = numpy.sum(allele_freq != 0, axis=1) == num_allele
    allele_freq = allele_freq[sel_vars]
    gts = gts[sel_vars, :, :]

    genotypes = list(combinations_with_replacement(range(num_allele),
                                                   ploidy))

    gts_counts = numpy.zeros((gts.shape[0], len(genotypes)))
    exp_gts_freq = numpy.ones((gts.shape[0], len(genotypes)))

    for i, genotype in enumerate(genotypes):

        permutated_gts = set(permutations(genotype))
        mask = None
        for gt in permutated_gts:
            permuted_mask = None
            for allele_idx in range(ploidy):
                if permuted_mask is None:
                    permuted_mask = gts[:, :, allele_idx] == gt[allele_idx]
                else:
                    new_mask = gts[:, :, allele_idx] == gt[allele_idx]
                    permuted_mask = numpy.logical_and(permuted_mask, new_mask)

            if mask is None:
                mask = permuted_mask
            else:
                mask = numpy.stack([mask, permuted_mask], axis=2)
                mask = numpy.any(mask, axis=2)

            allele = genotype[allele_idx]
            exp_gts_freq[:, i] *= allele_freq[:, allele]
        gts_counts[:, i] = numpy.sum(mask, axis=1)

    total_gt_counts = numpy.sum(gts_counts, axis=1)
    exp_gts_counts = (exp_gts_freq.T * total_gt_counts).T
    with numpy.errstate(invalid='ignore'):
        chi2, pvalue = chisquare(gts_counts, f_exp=exp_gts_counts, axis=1)
    return numpy.array([chi2, pvalue]).T


def calc_hwe_chi2_test(variations, num_allele=2,
                       min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                       chunk_size=SNPS_PER_CHUNK):
    if chunk_size is None:
        chunks = [variations]
    else:
        req_fields = REQUIRED_FIELDS_FOR_STAT['calc_hwe_chi2_test']
        chunks = variations.iterate_chunks(kept_fields=req_fields,
                                           chunk_size=chunk_size)
    hwe_test = None
    for chunk in chunks:
        chunk_hwe_test = _calc_hwe_chi2_test(chunk, num_allele,
                                             min_num_genotypes)
        if hwe_test is None:
            hwe_test = chunk_hwe_test
        else:
            hwe_test = numpy.append(hwe_test, chunk_hwe_test, axis=0)
    return hwe_test


def _get_allele_observations(variations, mask_func, weights_field=None,
                             mask_field=GT_FIELD):
    mat1 = variations[REF_OBS]
    alt_obs = variations[ALT_OBS]
    if is_dataset(alt_obs):
        alt_obs = alt_obs[:]
    mat2 = numpy.max(variations[ALT_OBS], axis=2)

    mask1 = is_missing(mat1, axis=None)
    mask2 = is_missing(mat2, axis=None)
    missing_mask = numpy.logical_not(numpy.logical_or(mask1, mask2))

    if mask_func is None:
        mask = missing_mask
    else:
        mask = mask_func(variations[mask_field])
        mask = numpy.logical_and(mask, missing_mask)
    mat1 = mat1[mask]
    mat2 = mat2[mask]
    if weights_field is None:
        weights = None
    else:
        weights = variations[weights_field][mask]
    return mat1, mat2, weights


def _hist2d_allele_observations(variations, n_bins=DEF_NUM_BINS, range_=None,
                                mask_func=None, weights_field=None,
                                mask_field=GT_FIELD):
    mat1, mat2, weights = _get_allele_observations(variations, mask_func,
                                                   weights_field=weights_field,
                                                   mask_field=mask_field)
    return numpy.histogram2d(mat1, mat2, bins=n_bins, range=range_,
                             weights=weights)


def _update_range(xrange, this_x_range):
    if xrange is None:
        xrange = list(this_x_range)
    if xrange[0] > this_x_range[0]:
        xrange[0] = this_x_range[0]
    if xrange[1] < this_x_range[1]:
        xrange[1] = this_x_range[1]
    return xrange


def _hist2d_allele_observations_by_chunk(variations, n_bins=DEF_NUM_BINS,
                                         range_=None, mask_func=None,
                                         weights_field=None,
                                         chunk_size=SNPS_PER_CHUNK,
                                         mask_field=GT_FIELD):
    fields = [REF_OBS, ALT_OBS, GT_FIELD]
    if range_ is None:
        xrange = None
        yrange = None
        for var_chunk in variations.iterate_chunks(kept_fields=fields,
                                                   chunk_size=chunk_size):
            mat1, mat2, _ = _get_allele_observations(var_chunk, mask_func,
                                                     mask_field=mask_field)
            this_x_range = calc_min_max(mat1)
            xrange = _update_range(xrange, this_x_range)
            this_y_range = calc_min_max(mat2)
            yrange = _update_range(yrange, this_y_range)
        range_ = [xrange, yrange]

    if weights_field is not None:
        fields.append(weights_field)
    hist = None
    for var_chunk in variations.iterate_chunks(kept_fields=fields,
                                               chunk_size=chunk_size):
        res = _hist2d_allele_observations(var_chunk, n_bins=n_bins,
                                          range_=range_, mask_func=mask_func,
                                          weights_field=weights_field)
        this_hist, xbins, y_bins = res
        if hist is None:
            hist = this_hist
        else:
            hist = numpy.add(hist, this_hist)
    return hist, xbins, y_bins


def hist2d_allele_observations(variations, n_bins=DEF_NUM_BINS, range_=None,
                               mask_func=None, chunk_size=SNPS_PER_CHUNK):
    if chunk_size:
        return _hist2d_allele_observations_by_chunk(variations,
                                                    n_bins=n_bins,
                                                    range_=range_,
                                                    mask_func=mask_func,
                                                    chunk_size=chunk_size)
    else:
        return _hist2d_allele_observations(variations, n_bins=n_bins,
                                           range_=range_, mask_func=mask_func)


def hist2d_gq_allele_observations(variations, n_bins=DEF_NUM_BINS, range_=None,
                                  mask_func=None, chunk_size=SNPS_PER_CHUNK,
                                  hist_counts=None):
    if hist_counts is None:
        res = hist2d_allele_observations(variations, n_bins=n_bins,
                                         range_=range_,
                                         mask_func=mask_func,
                                         chunk_size=chunk_size)
        hist_counts, _, _ = res

    if chunk_size:
        res = _hist2d_allele_observations_by_chunk(variations,
                                                   n_bins=n_bins,
                                                   range_=range_,
                                                   mask_func=mask_func,
                                                   weights_field=GQ_FIELD,
                                                   chunk_size=chunk_size)
    else:
        res = _hist2d_allele_observations(variations, n_bins=n_bins,
                                          range_=range_, mask_func=mask_func,
                                          weights_field=GQ_FIELD)
    hist, xbins, ybins = res
    with numpy.errstate(invalid='ignore'):
        hist = hist / hist_counts
        hist[numpy.isnan(hist)] = 0
    return hist, xbins, ybins


def calc_field_distribs_per_sample(variations, field, range_=None,
                                   n_bins=DEF_NUM_BINS,
                                   chunk_size=SNPS_PER_CHUNK,
                                   mask_func=None, mask_field=None):
    mat = variations[field]
    mask_mat = None
    if mask_field is not None:
        mask_mat = variations[mask_field]

    if range_ is None:
        min_, max_ = calc_min_max(mat)
        if issubclass(mat.dtype.type, numpy.integer) and min_ < 0:
            # we remove the missing data
            min_ = 0
        range_ = min_, max_ + 1

    if chunk_size:
        chunks = iterate_matrix_chunks(mat, chunk_size=chunk_size)
        if mask_mat is not None:
            mask_chunks = iterate_matrix_chunks(mask_mat,
                                                chunk_size=chunk_size)
            chunks = zip(chunks, mask_chunks)
    else:
        chunks = [mat]
        if mask_mat is not None:
            chunks = [(mat, mask_mat)]

    histograms = None

    for chunk in chunks:
        chunk_hists = None
        chunk_mask = None
        if mask_mat is not None and mask_func is not None:
            chunk, mask_chunk = chunk
            chunk_mask = mask_func(mask_chunk)
        for sample_idx in range(len(variations.samples)):
            dps = chunk[:, sample_idx]
            if chunk_mask is not None:
                dps = dps[chunk_mask[:, sample_idx]]
            chunk_hist, bins = histogram(dps, n_bins=n_bins, range_=range_)
            if chunk_hists is None:
                chunk_hists = chunk_hist
            else:
                chunk_hists = numpy.vstack((chunk_hists, chunk_hist))
        if histograms is None:
            histograms = chunk_hists
        else:
            histograms = numpy.add(histograms, chunk_hists)
    return histograms, bins


def _calc_maf_depth(variations, min_depth=DEF_MIN_DEPTH):
    ro = variations[RO_FIELD]
    ao = variations[AO_FIELD]
    if len(ro.shape) == len(ao.shape):
        ao = ao.reshape((ao.shape[0], ao.shape[1], 1))
    if is_dataset(ro):
        ro = ro[:]
    read_counts = numpy.append(ro.reshape((ao.shape[0], ao.shape[1],
                                           1)), ao, axis=2)
    read_counts[read_counts == MISSING_VALUES[int]] = 0
    total_counts = numpy.sum(read_counts, axis=2)
    with numpy.errstate(invalid='ignore'):
        depth_maf = numpy.max(read_counts, axis=2) / total_counts
    depth_maf[total_counts < min_depth] = numpy.nan
    return depth_maf


def calc_maf_depth_distribs_per_sample(variations, min_depth=DEF_MIN_DEPTH,
                                       n_bins=DEF_NUM_BINS * 2,
                                       chunk_size=SNPS_PER_CHUNK):
    range_ = 0, 1
    if chunk_size:
        chunks = variations.iterate_chunks(kept_fields=[AO_FIELD, RO_FIELD],
                                           chunk_size=chunk_size)
    else:
        chunks = [variations]

    histograms = None
    for chunk in chunks:
        chunk_hists = None
        maf_depth = _calc_maf_depth(chunk, min_depth)
        for sample_idx in range(len(variations.samples)):
            chunk_hist, bins = histogram(maf_depth[:, sample_idx],
                                         n_bins=n_bins, range_=range_)
            if chunk_hists is None:
                chunk_hists = chunk_hist
            else:
                chunk_hists = numpy.vstack((chunk_hists, chunk_hist))
        if histograms is None:
            histograms = chunk_hists
        else:
            histograms = numpy.add(histograms, chunk_hists)
    return histograms, bins


class PositionalStatsCalculator:
    def __init__(self, chrom, pos, stat, window_size=None, step=1,
                 take_windows=True):
        self.chrom = chrom
        self.pos = pos
        self.stat = stat
        self.chrom_names = numpy.unique(self.chrom)
        self.window_size = window_size
        self.step = step
        self.take_windows = take_windows

    def _calc_chrom_window_stat(self, pos, values):
        # TODO: take into account unknown positions in the genome fastafile?
        if self.window_size and self.take_windows:
            for i in range(pos[0], pos[-1], self.step):
                window = numpy.logical_and(pos >= i,
                                           pos < i + self.window_size)
                yield i, numpy.sum(values[window]) / self.window_size
        else:
            for x, y in zip(pos, values):
                yield x, y

    def calc_window_stat(self):
        w_chroms, w_pos, w_stat = [], [], []
        for chrom_name, pos, values in self._iterate_chroms():
            for chrom_pos, value in self._calc_chrom_window_stat(pos, values):
                w_chroms.append(chrom_name)
                w_pos.append(chrom_pos)
                w_stat.append(value)
        chrom = numpy.array(w_chroms)
        pos = numpy.array(w_pos)
        stat = numpy.array(w_stat)

        return PositionalStatsCalculator(chrom, pos, stat, self.window_size,
                                         self.step, False)

    def _iterate_chroms(self):
        for chrom_name in self.chrom_names:
            mask = numpy.logical_and(self.chrom == chrom_name,
                                     numpy.logical_not(numpy.isnan(self.stat)))
            values = self.stat[mask]
            pos = self.pos[mask]
            if values.shape != () and values.shape != (0,):
                yield chrom_name, pos, values

    def _get_track_definition(self, track_type, name, description, **kwargs):
        types = {'wig': 'wiggle_0', 'bedgraph': 'bedGraph'}
        track_line = 'track type={} name="{}" description="{}"'
        track_line = track_line.format(types[track_type], name, description)
        for key, value in kwargs:
            track_line += ' {}={}'.format(key, value)
        return track_line

    def to_wig(self):
        stat = self.stat
        span = self.window_size
        if (stat.shape[0] != self.pos.shape[0] or
                stat.shape[0] != self.chrom.shape[0]):
            raise ValueError('Stat does not have the same size as pos')
        for chrom_name, pos, values in self._iterate_chroms():
            try:
                chrom_name = chrom_name.decode()
            except AttributeError:
                pass
            chrom_line = 'variableStep chrom={}'.format(chrom_name)
            variable_step = True
            if span is not None:
                chrom_line = 'fixedStep chrom={} start={} span={} step={}'
                chrom_line = chrom_line.format(chrom_name, pos[0], span,
                                               self.step)
                variable_step = False
            yield chrom_line
            # When containing only one value, it is not iterable
            if values.shape != () and pos.shape != ():
                for pos, value in self._calc_chrom_window_stat(pos, values):
                    if variable_step:
                        yield '{} {}'.format(pos, value)
                    else:
                        yield str(value)
            else:
                if variable_step:
                    yield '{} {}'.format(pos, value)
                else:
                    yield str(value)

    def to_bedGraph(self):
        window_size = self.window_size
        stat = self.stat
        if window_size is None:
            window_size = 1
        if (stat.shape[0] != self.pos.shape[0] or
                stat.shape[0] != self.chrom.shape[0]):
            raise ValueError('Stat does not have the same size as pos')
        for chrom_name, pos, values in self._iterate_chroms():
            try:
                chrom_name = chrom_name.decode()
            except AttributeError:
                pass
            # When containing only one value, it is not iterable
            if values.shape != () and pos.shape != ():
                for pos, value in self._calc_chrom_window_stat(pos, values):
                    yield '{} {} {} {}'.format(chrom_name, pos,
                                               pos + window_size, value)
            else:
                yield '{} {} {} {}'.format(chrom_name, pos,
                                           pos + window_size, value)

    def write(self, fhand, track_name, track_description,
              buffer_size=1000, track_type='bedgraph', **kwargs):
        # TODO: Fix error. writes a line for each position
        get_lines = {'wig': self.to_wig, 'bedgraph': self.to_bedGraph}
        buffer = self._get_track_definition(track_type, track_name,
                                            track_description, **kwargs) + '\n'
        lines = 1
        for line in get_lines[track_type]():
            lines += 1
            buffer += line + '\n'
            if lines == buffer_size:
                fhand.write(buffer)
                buffer = ''
                lines = 0
        if lines != buffer_size:
            fhand.write(buffer)
        fhand.flush()
