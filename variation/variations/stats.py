import re
from functools import reduce
from itertools import combinations_with_replacement, permutations
import operator
import math
from functools import lru_cache

import numpy
from scipy.stats import chisquare

from allel.stats.ld import rogers_huff_r
from allel.model.ndarray import GenotypeArray

from variation import (
    MISSING_VALUES,
    SNPS_PER_CHUNK,
    DEF_MIN_DEPTH,
    MISSING_INT,
    GT_FIELD,
    ALT_FIELD,
    DP_FIELD,
    GQ_FIELD,
    CHROM_FIELD,
    POS_FIELD,
    RO_FIELD,
    AO_FIELD,
    MIN_NUM_GENOTYPES_FOR_POP_STAT,
    AD_FIELD,
)
from variation.matrix.stats import (
    counts_by_row,
    counts_and_allels_by_row,
    row_value_counter_fact,
)
from variation.matrix.methods import (
    is_missing,
    calc_min_max,
    is_dataset,
    iterate_matrix_chunks,
)
from variation.plot import _estimate_percentiles_from_distrib

DEF_NUM_BINS = 20

REQUIRED_FIELDS_FOR_STAT = {
    "calc_maf": [GT_FIELD],
    "calc_allele_freq": [GT_FIELD],
    "calc_hwe_chi2_test": [GT_FIELD, ALT_FIELD],
    "calc_called_gts_distrib_per_depth": [DP_FIELD, GT_FIELD],
    "calc_missing_gt": [GT_FIELD],
    "calc_obs_het": [GT_FIELD],
    "calc_obs_het_by_sample": [GT_FIELD],
}


def _calc_histogram(vector, n_bins, range_, weights=None):
    try:
        dtype = vector.dtype
    except AttributeError:
        dtype = type(vector[0])
    missing_value = MISSING_VALUES[dtype]

    if is_dataset(vector):
        vector = vector[:]

    if weights is None:
        if math.isnan(missing_value):
            not_nan = ~numpy.isnan(vector)
        else:
            not_nan = vector != missing_value

        vector = vector[not_nan]
    try:
        result = numpy.histogram(vector, bins=n_bins, range=range_, weights=weights)
    except ValueError as error:
        if "parameter must be finite" in str(error) or re.search(
            "autodetected range of .*finite", str(error)
        ):
            isfinite = ~numpy.isinf(vector)
            vector = vector[isfinite]
            if weights is not None:
                weights = weights[isfinite]
            result = numpy.histogram(vector, bins=n_bins, range=range_, weights=weights)
        else:
            raise
    return result


def histogram(vector, n_bins=DEF_NUM_BINS, range_=None, weights=None):
    return _calc_histogram(vector, n_bins, range_=range_, weights=weights)


def calc_cum_distrib(distrib):
    if len(distrib.shape) == 1:
        return numpy.fliplr(numpy.cumsum(numpy.fliplr([distrib]), axis=1))[0]
    else:
        return numpy.fliplr(numpy.cumsum(numpy.fliplr(distrib), axis=1))


def histograms_for_columns(matrix2d, n_bins=DEF_NUM_BINS, range_=None):
    if range_ is None:
        range_ = calc_min_max(matrix2d, chunk_size=None)

    hists = None
    edges = None
    n_cols = matrix2d.shape[1]
    for col_idx in range(n_cols):
        column = matrix2d[:, col_idx]
        counts, col_edges = _calc_histogram(column, n_bins, range_=range_)
        if hists is None:
            # hists = numpy.zeros((n_cols, counts.shape[0]))
            hists = numpy.zeros((counts.shape[0], n_cols))
            edges = col_edges
        assert numpy.allclose(edges, col_edges)
        # hists[col_idx, :] += counts
        hists[:, col_idx] += counts
    return hists, edges


def _guess_stat_funct_called(calc_funct):
    if "func" in dir(calc_funct):
        funct_name = calc_funct.func.__name__
    else:
        funct_name = calc_funct.__name__
    return funct_name


def _calc_stats_for_chunks(calc_funct, variations, chunk_size):
    funct_name = _guess_stat_funct_called(calc_funct)
    req_fields = REQUIRED_FIELDS_FOR_STAT[funct_name]
    vectors = (
        calc_funct(chunk)
        for chunk in variations.iterate_chunks(
            kept_fields=req_fields, chunk_size=chunk_size
        )
    )
    return vectors


def histogram_for_chunks(
    variations, calc_funct, n_bins=DEF_NUM_BINS, range_=None, chunk_size=None
):
    if range_ is None:
        range_ = _calc_range(_calc_stats_for_chunks(calc_funct, variations, chunk_size))

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


def _calc_mac(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gt_counts, alleles = counts_and_allels_by_row(variations[GT_FIELD])
    if gt_counts is None:
        return numpy.array([])

    if MISSING_INT in alleles:
        missing_allele_idx = alleles.index(MISSING_INT)
        num_missing = numpy.copy(gt_counts[:, missing_allele_idx])
        gt_counts[:, missing_allele_idx] = 0
    else:
        num_missing = 0
    max_ = numpy.amax(gt_counts, axis=1)
    num_samples = variations[GT_FIELD].shape[1]
    ploidy = variations[GT_FIELD].shape[2]
    num_chroms = num_samples * ploidy
    mac = num_samples - (num_chroms - num_missing - max_) / ploidy
    # we set the snps with no data to nan
    mac[max_ == 0] = numpy.nan
    return _mask_stats_with_few_samples(mac, variations, min_num_genotypes)


def _calc_maf(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]
    gt_counts = counts_by_row(gts, missing_value=MISSING_INT)
    if gt_counts is None:
        return numpy.array([])
    max_ = numpy.amax(gt_counts, axis=1)
    sum_ = numpy.sum(gt_counts, axis=1)

    # To avoid problems with NaNs
    with numpy.errstate(invalid="ignore"):
        mafs_gt = max_ / sum_
    return _mask_stats_with_few_samples(mafs_gt, variations, min_num_genotypes)


def _calc_allele_observation_based_maf(variations):
    allele_depths = variations[AD_FIELD]
    allele_depths[allele_depths == MISSING_INT] = 0

    if not allele_depths.size:
        return numpy.array([])

    allele_depths_allele_counts_for_snps = numpy.sum(allele_depths, axis=1)
    major_allele_count_by_snp = numpy.max(allele_depths_allele_counts_for_snps, axis=1)
    tot_allele_count_by_snp = numpy.sum(allele_depths_allele_counts_for_snps, axis=1)
    # To avoid problems with zero division
    with numpy.errstate(invalid="ignore"):
        maf_by_snp = major_allele_count_by_snp / tot_allele_count_by_snp
    return maf_by_snp


def _calc_allele_observation_based_maf_by_chunk(variations, chunk_size=SNPS_PER_CHUNK):
    mafs = None
    for chunk in variations.iterate_chunks(
        kept_fields=[AD_FIELD], chunk_size=chunk_size
    ):
        chunk_maf = _calc_allele_observation_based_maf(chunk)
        if mafs is None:
            mafs = chunk_maf
        else:
            mafs = numpy.append(mafs, chunk_maf)
    if mafs is None:
        return numpy.array([])
    return mafs


def _calc_mac_by_chunk(
    variations,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=SNPS_PER_CHUNK,
):
    macs = None
    for chunk in variations.iterate_chunks(
        kept_fields=[GT_FIELD], chunk_size=chunk_size
    ):
        chunk_maf = _calc_mac(chunk, min_num_genotypes=min_num_genotypes)
        if macs is None:
            macs = chunk_maf
        else:
            macs = numpy.append(macs, chunk_maf)
    if macs is None:
        return numpy.array([])
    return macs


def _calc_maf_by_chunk(
    variations,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=SNPS_PER_CHUNK,
):
    mafs = None
    for chunk in variations.iterate_chunks(
        kept_fields=[GT_FIELD], chunk_size=chunk_size
    ):
        chunk_maf = _calc_maf(chunk, min_num_genotypes=min_num_genotypes)
        if mafs is None:
            mafs = chunk_maf
        else:
            mafs = numpy.append(mafs, chunk_maf)
    if mafs is None:
        return numpy.array([])
    return mafs


def calc_mac(
    variations,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=SNPS_PER_CHUNK,
):
    if chunk_size is None:
        return _calc_mac(variations, min_num_genotypes=min_num_genotypes)
    else:
        return _calc_mac_by_chunk(variations, min_num_genotypes, chunk_size)


def calc_maf(
    variations,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=SNPS_PER_CHUNK,
):
    if chunk_size is None:
        return _calc_maf(variations, min_num_genotypes=min_num_genotypes)
    else:
        return _calc_maf_by_chunk(variations, min_num_genotypes, chunk_size)


def calc_allele_observation_based_maf(variations, chunk_size=SNPS_PER_CHUNK):
    if not chunk_size:
        return _calc_allele_observation_based_maf(variations)
    else:
        return _calc_allele_observation_based_maf_by_chunk(variations, chunk_size)


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


def _calc_called_gts_distrib_per_depth_by_chunk(variations, depths, chunk_size):
    distributions = None
    req_fields = REQUIRED_FIELDS_FOR_STAT["calc_called_gts_distrib_per_depth"]
    for chunk in variations.iterate_chunks(
        kept_fields=req_fields, chunk_size=chunk_size
    ):
        chunk_distribs = None
        for depth in depths:
            chunk_distrib, bins = calc_num_samples_called_distrib(chunk, depth)
            if chunk_distribs is None:
                chunk_distribs = chunk_distrib
            else:
                chunk_distribs = numpy.vstack((chunk_distribs, chunk_distrib))
        if distributions is None:
            distributions = chunk_distribs
        else:
            distributions = numpy.add(distributions, chunk_distribs)

    return distributions, bins


def calc_called_gts_distrib_per_depth(variations, depths, chunk_size=SNPS_PER_CHUNK):
    if chunk_size:
        return _calc_called_gts_distrib_per_depth_by_chunk(
            variations, depths, chunk_size
        )
    else:
        return _calc_called_gts_distrib_per_depth(variations, depths)


def _calc_items_in_row(mat):
    num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
    return num_items_per_row


def calc_missing_gt(variations, rates=True, axis=1):
    gts = variations[GT_FIELD]
    if gts.shape[0] == 0:
        return numpy.array([])
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
    if missing.shape[0] == 0:
        return numpy.array([])

    if rates:
        return 1 - missing
    else:
        total = variations[GT_FIELD].shape[axis]
        return total - missing


def _call_is_het(variations, min_call_dp, max_call_dp=None):
    is_hom, is_missing = _call_is_hom(variations, min_call_dp, max_call_dp=max_call_dp)
    if is_hom.shape[0] == 0:
        return is_hom, is_missing
    is_het = numpy.logical_not(is_hom)
    is_het[is_missing] = False
    return is_het, is_missing


def call_is_het(gts):
    return _call_is_het({GT_FIELD: gts}, min_call_dp=0)[0]


def _call_is_hom(variations, min_call_dp, max_call_dp=None):
    gts = variations[GT_FIELD]

    if gts.shape[0] == 0:
        return numpy.array([]), numpy.array([])

    if is_dataset(gts):
        gts = gts[:]

    is_hom = numpy.full(gts.shape[:-1], True, dtype=numpy.bool)
    for idx in range(1, gts.shape[2]):
        is_hom = numpy.logical_and(gts[:, :, idx] == gts[:, :, idx - 1], is_hom)

    missing_gts = is_missing(gts, axis=2)

    if min_call_dp or max_call_dp:
        dps = variations[DP_FIELD]
        if is_dataset(dps):
            dps = dps[:]
        if min_call_dp:
            low_dp = dps < min_call_dp
            missing_gts = numpy.logical_or(missing_gts, low_dp)
        if max_call_dp:
            high_dp = dps > max_call_dp
            missing_gts = numpy.logical_or(missing_gts, high_dp)

    is_hom[missing_gts] = False

    return is_hom, missing_gts


def call_is_hom(gts):
    return _call_is_hom({GT_FIELD: gts}, min_call_dp=0)[0]


def call_is_hom_ref(gts):
    return numpy.logical_and(call_is_hom(gts), gts[:, :, 0] == 0)


def call_is_hom_alt(gts):
    return numpy.logical_and(call_is_hom(gts), gts[:, :, 0] != 0)


def _calc_obs_het_counts(variations, axis, min_call_dp, max_call_dp=None):
    is_het, is_missing = _call_is_het(variations, min_call_dp, max_call_dp)
    if is_het.shape[0] == 0:
        return is_het, is_missing
    return (
        numpy.sum(is_het, axis=axis),
        numpy.sum(numpy.logical_not(is_missing), axis=axis),
    )


def _get_mask_for_masking_samples_with_few_gts(
    variations, min_num_genotypes, num_called_gts=None
):
    if num_called_gts is None:
        num_called_gts = calc_called_gt(variations, rates=False)
    mask = num_called_gts < min_num_genotypes
    return mask


def _mask_stats_with_few_samples(
    stats, variations, min_num_genotypes, num_called_gts=None, masking_value=numpy.NaN
):
    if min_num_genotypes:
        mask = _get_mask_for_masking_samples_with_few_gts(
            variations, min_num_genotypes, num_called_gts=num_called_gts
        )
        stats[mask] = masking_value
    return stats


def calc_obs_het(
    variations,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    min_call_dp=0,
    max_call_dp=None,
):
    het, called_gts = _calc_obs_het_counts(
        variations, axis=1, min_call_dp=min_call_dp, max_call_dp=max_call_dp
    )
    # To avoid problems with NaNs
    with numpy.errstate(invalid="ignore"):
        het = het / called_gts
    return _mask_stats_with_few_samples(
        het, variations, min_num_genotypes, num_called_gts=called_gts
    )


def _calc_obs_het_by_sample(variations, min_call_dp=0, max_call_dp=None):
    return _calc_obs_het_counts(
        variations, axis=0, min_call_dp=min_call_dp, max_call_dp=max_call_dp
    )


def calc_obs_het_by_sample(
    variations, chunk_size=SNPS_PER_CHUNK, min_call_dp=0, max_call_dp=None
):
    if not chunk_size:
        chunks = [variations]
    else:
        kept_fields = [GT_FIELD]
        if DP_FIELD in variations.keys():
            kept_fields.append(DP_FIELD)
        chunks = variations.iterate_chunks(
            kept_fields=kept_fields, chunk_size=chunk_size
        )
    obs_het_by_sample = None
    called_gts = None
    for chunk in chunks:
        chunk_obs_het_by_sample, missing = _calc_obs_het_by_sample(
            chunk, min_call_dp, max_call_dp
        )
        chunk_called_gts = missing
        if called_gts is None:
            obs_het_by_sample = chunk_obs_het_by_sample
            called_gts = chunk_called_gts
        else:
            obs_het_by_sample += chunk_obs_het_by_sample
            called_gts += chunk_called_gts
    with numpy.errstate(invalid="ignore"):
        obs_het_by_sample = obs_het_by_sample / called_gts
    return obs_het_by_sample


def calc_stats_by_sample(
    variations,
    chunk_size=SNPS_PER_CHUNK,
    min_call_dp=0,
    max_call_dp=None,
    dp_range=None,
    dp_n_bins=DEF_NUM_BINS,
):
    do_depth = True if DP_FIELD in variations.keys() else False

    if not chunk_size:
        chunks = [variations]
    else:
        kept_fields = [GT_FIELD]
        if do_depth:
            kept_fields.append(DP_FIELD)
        chunks = variations.iterate_chunks(
            kept_fields=kept_fields, chunk_size=chunk_size
        )

    if dp_range is None and do_depth:
        dp_range = calc_min_max(variations[DP_FIELD], chunk_size=chunk_size)
    if dp_range is not None and dp_range[0] < 0:
        dp_range = [0, dp_range[1]]

    het_counts = None
    call_gt_counts = None
    hom_counts = None
    dp_hist_cnts = None
    dp_hist_no_missing_cnts = None
    dp_het_hist_cnts = None
    for chunk in chunks:
        is_hom, is_missing = _call_is_hom(chunk, min_call_dp, max_call_dp)
        is_het = numpy.logical_not(is_hom)
        is_het[is_missing] = False

        chunk_het_counts = numpy.sum(is_het, axis=0)
        if het_counts is None:
            het_counts = chunk_het_counts
        else:
            het_counts += chunk_het_counts

        chunk_hom_ref_counts = numpy.sum(is_hom, axis=0)
        if hom_counts is None:
            hom_counts = chunk_hom_ref_counts
        else:
            hom_counts += chunk_hom_ref_counts

        chunk_called_gts = numpy.sum(numpy.logical_not(is_missing), axis=0)
        if call_gt_counts is None:
            call_gt_counts = chunk_called_gts
        else:
            call_gt_counts += chunk_called_gts

        if do_depth:
            dps = chunk[DP_FIELD]
            chunk_dp_hist = histograms_for_columns(
                dps, n_bins=dp_n_bins, range_=dp_range
            )
            if dp_hist_cnts is None:
                dp_hist_cnts = chunk_dp_hist[0]
                dp_bin_edges = chunk_dp_hist[1]
            else:
                dp_hist_cnts += chunk_dp_hist[0]

            dps_no_missing = numpy.copy(dps)
            dps_no_missing[is_missing] = MISSING_INT
            chunk_dp_hist_no_missing = histograms_for_columns(
                dps_no_missing, n_bins=dp_n_bins, range_=dp_range
            )
            if dp_hist_no_missing_cnts is None:
                dp_hist_no_missing_cnts = chunk_dp_hist_no_missing[0]
            else:
                dp_hist_no_missing_cnts += chunk_dp_hist_no_missing[0]

            dps_het = numpy.copy(dps)
            dps_het[numpy.logical_not(is_het)] = MISSING_INT
            chunk_dp_het_hist = histograms_for_columns(
                dps_het, n_bins=dp_n_bins, range_=dp_range
            )
            if dp_het_hist_cnts is None:
                dp_het_hist_cnts = chunk_dp_het_hist[0]
            else:
                dp_het_hist_cnts += chunk_dp_het_hist[0]
    if do_depth:
        dp_hom_hist_cnts = dp_hist_no_missing_cnts - dp_het_hist_cnts

    with numpy.errstate(invalid="ignore"):
        het_rate = het_counts / call_gt_counts

    with numpy.errstate(invalid="ignore"):
        hom_rate = hom_counts / call_gt_counts

    with numpy.errstate(invalid="ignore"):
        if isinstance(variations, dict):
            # just for the test
            num_vars = variations[GT_FIELD].shape[0]
        else:
            num_vars = variations.num_variations
        call_gt_rate = call_gt_counts / num_vars

    # for the test
    samples = {} if isinstance(variations, dict) else variations.samples[:]
    res = {
        "called_gt_rate": call_gt_rate,
        "obs_het": het_rate,
        "homozygosity": hom_rate,
        "samples": samples,
    }
    if do_depth:
        dp_hists = {
            "bin_edges": dp_bin_edges,
            "dp_counts": dp_hist_cnts,
            "dp_no_missing_counts": dp_hist_no_missing_cnts,
            "dp_het_counts": dp_het_hist_cnts,
            "dp_hom_counts": dp_hom_hist_cnts,
        }
        res["dp_hists"] = dp_hists
    return res


def write_stats_by_sample(stats, fhand):
    sep = "\t"
    fields = ["sample", "call_rate", "heterozygosity", "mean_dp"]
    fhand.write(sep.join(fields))
    fhand.write("\n")
    samples = stats["samples"]
    means = _estimate_percentiles_from_distrib(
        stats["dp_hists"]["dp_counts"],
        stats["dp_hists"]["bin_edges"],
        percentiles=(0.5,),
        samples_in_rows=False,
    )
    means = means[0, :]
    assert len(samples) == means.shape[0]
    call_rate = stats["called_gt_rate"]
    hets = stats["obs_het"]
    for values in zip(samples, call_rate, hets, means):
        fhand.write(sep.join(map(str, values)))
        fhand.write("\n")


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
        chunks = variations.iterate_chunks(
            kept_fields=[GT_FIELD], chunk_size=chunk_size
        )
    gt_type_stats = None
    for chunk in chunks:
        chunk_stats = _calc_gt_type_stats(chunk)
        if gt_type_stats is None:
            gt_type_stats = chunk_stats
        else:
            gt_type_stats += chunk_stats
    return gt_type_stats


def _snps_close(chrom1, pos1, chrom2, pos2, dist):
    if chrom1 != chrom2:
        return None
    if pos1 == pos2:
        return True
    elif abs(pos1 - pos2) <= dist:
        return True
    else:
        return False


class _ArrayWrapper:
    def __init__(self, array, cache_len=400):
        self.array = array
        self._h5_cache = None
        self._cache_len = cache_len
        self._cache_offset = None

    @lru_cache(maxsize=128)
    def __getitem__(self, idx):
        if self._h5_cache is None:
            self._fill_cache(idx)
        return self.read_from_cache(idx)

    def read_from_cache(self, idx):
        # is cache consumed?
        if idx - self._cache_offset >= len(self._h5_cache):
            self._fill_cache(idx)
        return self._h5_cache[idx - self._cache_offset]

    def _fill_cache(self, idx):
        self._h5_cache = self.array[idx : idx + self._cache_len]
        self._cache_offset = idx


def calc_snp_density(variations, window):
    half_win = (window - 1) / 2
    chroms = _ArrayWrapper(variations[CHROM_FIELD])
    poss = _ArrayWrapper(variations[POS_FIELD])

    last_idx = len(chroms.array) - 1
    left_win_idx = 0
    right_win_idx = 0
    for idx in range(chroms.array.shape[0]):
        chrom = chroms[idx]
        pos = poss[idx]
        # update left index
        while True:
            if not _snps_close(
                chrom, pos, chroms[left_win_idx], poss[left_win_idx], half_win
            ):
                left_win_idx += 1
            else:
                break
        # update right index
        while True:
            if right_win_idx + 1 > last_idx:
                break
            elif _snps_close(
                chrom, pos, chroms[right_win_idx + 1], poss[right_win_idx + 1], half_win
            ):
                right_win_idx += 1
            else:
                break
        snps_in_win = right_win_idx - left_win_idx + 1
        yield snps_in_win


def _calc_allele_counts(gts, alleles=None):
    return counts_by_row(gts, MISSING_VALUES[int], alleles=alleles)


def calc_allele_freq(
    variations, alleles=None, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT
):
    gts = variations[GT_FIELD]

    if gts.shape[0] == 0:
        return numpy.array([])

    allele_counts = _calc_allele_counts(gts, alleles=alleles)
    if allele_counts is None:
        raise ValueError("No alleles, everything is missing data")
    total_counts = numpy.sum(allele_counts, axis=1)
    with numpy.errstate(invalid="ignore"):
        allele_freq = allele_counts / total_counts[:, None]
    _mask_stats_with_few_samples(allele_freq, variations, min_num_genotypes)
    return allele_freq


def calc_allele_freq_by_depth(chunk):
    allele_counts = chunk[AD_FIELD]
    allele_counts[allele_counts == -1] = 0
    allele_counts = numpy.sum(allele_counts, axis=1)
    total_counts = numpy.sum(allele_counts, axis=1)
    allele_freq = allele_counts / total_counts[:, None]
    return allele_freq


def calc_expected_het(
    variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, ploidy=None
):
    try:
        allele_freq = calc_allele_freq(variations, min_num_genotypes=min_num_genotypes)
    except ValueError:
        exp_het = numpy.empty((variations.num_variations,))
        exp_het[:] = numpy.nan
        return exp_het
    if allele_freq.shape[0] == 0:
        return numpy.array([])
    gts = variations[GT_FIELD]

    if ploidy is None:
        ploidy = gts.shape[2]

    exp_het = 1 - numpy.sum(allele_freq**ploidy, axis=1)
    return exp_het


def calc_unbias_expected_het(
    variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, ploidy=None
):
    exp_het = calc_expected_het(
        variations, min_num_genotypes=min_num_genotypes, ploidy=ploidy
    )

    num_called_gts = calc_called_gt(variations, rates=False)
    num_samples = num_called_gts.astype(float)
    num_samples[num_samples < min_num_genotypes] = numpy.nan

    unbiased_exp_het = (2 * num_samples / (2 * num_samples - 1)) * exp_het
    return unbiased_exp_het


def calc_unbias_expected_het2(
    variations, ploidy=None, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT
):
    exp_het = calc_expected_het(
        variations, ploidy=ploidy, min_num_genotypes=min_num_genotypes
    )

    num_called_gts = calc_called_gt(variations, rates=False)
    num_samples = num_called_gts.astype(float)
    num_samples[num_samples < min_num_genotypes] = numpy.nan

    unbiased_exp_het = (num_samples / (num_samples - 1)) * exp_het
    return unbiased_exp_het


def _calc_inbreeding_coef(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    obs_het = calc_obs_het(variations, min_num_genotypes=min_num_genotypes)
    exp_het = calc_expected_het(variations, min_num_genotypes=min_num_genotypes)
    with numpy.errstate(invalid="ignore"):
        inbreed = 1 - (obs_het / exp_het)
    return inbreed


def calc_inbreeding_coef(
    variations,
    chunk_size=SNPS_PER_CHUNK,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    if chunk_size is None:
        chunks = [variations]
    else:
        chunks = variations.iterate_chunks(
            kept_fields=[GT_FIELD, ALT_FIELD], chunk_size=chunk_size
        )
    inbreed_coef = None
    for chunk in chunks:
        chunk_inbreed_coef = _calc_inbreeding_coef(chunk, min_num_genotypes)
        if inbreed_coef is None:
            inbreed_coef = chunk_inbreed_coef
        else:
            inbreed_coef = numpy.append(inbreed_coef, chunk_inbreed_coef)
    if inbreed_coef is None:
        return numpy.array([])
    return inbreed_coef


def _calc_hwe_chi2_test(
    variations, num_allele, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT
):
    ploidy = variations.ploidy
    gts = variations[GT_FIELD]
    if gts.shape[0] == 0:
        return numpy.array([])

    allele_freq = calc_allele_freq(variations, min_num_genotypes=min_num_genotypes)
    # Select vars with a certain number of alleles
    sel_vars = numpy.sum(allele_freq != 0, axis=1) == num_allele
    allele_freq = allele_freq[sel_vars]
    gts = gts[sel_vars, :, :]

    genotypes = list(combinations_with_replacement(range(num_allele), ploidy))

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
    with numpy.errstate(invalid="ignore"):
        chi2, pvalue = chisquare(gts_counts, f_exp=exp_gts_counts, axis=1)
    return numpy.array([chi2, pvalue]).T


def calc_hwe_chi2_test(
    variations,
    num_allele=2,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=SNPS_PER_CHUNK,
):
    if chunk_size is None:
        chunks = [variations]
    else:
        req_fields = REQUIRED_FIELDS_FOR_STAT["calc_hwe_chi2_test"]
        chunks = variations.iterate_chunks(
            kept_fields=req_fields, chunk_size=chunk_size
        )
    hwe_test = None
    for chunk in chunks:
        chunk_hwe_test = _calc_hwe_chi2_test(chunk, num_allele, min_num_genotypes)
        if hwe_test is None:
            hwe_test = chunk_hwe_test
        else:
            hwe_test = numpy.append(hwe_test, chunk_hwe_test, axis=0)
    return hwe_test


def _get_allele_observations(
    variations, mask_func, weights_field=None, mask_field=GT_FIELD
):
    mat1 = variations[RO_FIELD]
    alt_obs = variations[AO_FIELD]
    if is_dataset(alt_obs):
        alt_obs = alt_obs[:]
    mat2 = numpy.max(variations[AO_FIELD], axis=2)

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


def _hist2d_allele_observations(
    variations,
    n_bins=DEF_NUM_BINS,
    range_=None,
    mask_func=None,
    weights_field=None,
    mask_field=GT_FIELD,
):
    mat1, mat2, weights = _get_allele_observations(
        variations, mask_func, weights_field=weights_field, mask_field=mask_field
    )
    return numpy.histogram2d(mat1, mat2, bins=n_bins, range=range_, weights=weights)


def _update_range(xrange, this_x_range):
    if xrange is None:
        xrange = list(this_x_range)
    if xrange[0] > this_x_range[0]:
        xrange[0] = this_x_range[0]
    if xrange[1] < this_x_range[1]:
        xrange[1] = this_x_range[1]
    return xrange


def _hist2d_allele_observations_by_chunk(
    variations,
    n_bins=DEF_NUM_BINS,
    range_=None,
    mask_func=None,
    weights_field=None,
    chunk_size=SNPS_PER_CHUNK,
    mask_field=GT_FIELD,
):
    fields = [RO_FIELD, AO_FIELD, GT_FIELD]
    if range_ is None:
        xrange = None
        yrange = None
        for var_chunk in variations.iterate_chunks(
            kept_fields=fields, chunk_size=chunk_size
        ):
            mat1, mat2, _ = _get_allele_observations(
                var_chunk, mask_func, mask_field=mask_field
            )
            this_x_range = calc_min_max(mat1)
            xrange = _update_range(xrange, this_x_range)
            this_y_range = calc_min_max(mat2)
            yrange = _update_range(yrange, this_y_range)
        range_ = [xrange, yrange]

    if weights_field is not None:
        fields.append(weights_field)
    hist = None
    for var_chunk in variations.iterate_chunks(
        kept_fields=fields, chunk_size=chunk_size
    ):
        res = _hist2d_allele_observations(
            var_chunk,
            n_bins=n_bins,
            range_=range_,
            mask_func=mask_func,
            weights_field=weights_field,
        )
        this_hist, xbins, y_bins = res
        if hist is None:
            hist = this_hist
        else:
            hist = numpy.add(hist, this_hist)
    return hist, xbins, y_bins


def hist2d_allele_observations(
    variations,
    n_bins=DEF_NUM_BINS,
    range_=None,
    mask_func=None,
    chunk_size=SNPS_PER_CHUNK,
):
    if chunk_size:
        return _hist2d_allele_observations_by_chunk(
            variations,
            n_bins=n_bins,
            range_=range_,
            mask_func=mask_func,
            chunk_size=chunk_size,
        )
    else:
        return _hist2d_allele_observations(
            variations, n_bins=n_bins, range_=range_, mask_func=mask_func
        )


def hist2d_gq_allele_observations(
    variations,
    n_bins=DEF_NUM_BINS,
    range_=None,
    mask_func=None,
    chunk_size=SNPS_PER_CHUNK,
    hist_counts=None,
):
    if hist_counts is None:
        res = hist2d_allele_observations(
            variations,
            n_bins=n_bins,
            range_=range_,
            mask_func=mask_func,
            chunk_size=chunk_size,
        )
        hist_counts, _, _ = res

    if chunk_size:
        res = _hist2d_allele_observations_by_chunk(
            variations,
            n_bins=n_bins,
            range_=range_,
            mask_func=mask_func,
            weights_field=GQ_FIELD,
            chunk_size=chunk_size,
        )
    else:
        res = _hist2d_allele_observations(
            variations,
            n_bins=n_bins,
            range_=range_,
            mask_func=mask_func,
            weights_field=GQ_FIELD,
        )
    hist, xbins, ybins = res
    with numpy.errstate(invalid="ignore"):
        hist = hist / hist_counts
        hist[numpy.isnan(hist)] = 0
    return hist, xbins, ybins


def _packed_gt_to_tuple(gt, ploidy):
    gt_len = ploidy * 2
    gt_fmt = "%0" + str(gt_len) + "d"
    gt = gt_fmt % gt
    gt = tuple(sorted([int(gt[idx : idx + 2]) for idx in range(0, len(gt), 2)]))
    return gt


def _gt_is_homo(gt):
    return all(allele == gt[0] for allele in gt)


def _hist2d_het_allele_freq(
    variations,
    n_bins=DEF_NUM_BINS,
    allele_freq_range=None,
    het_range=None,
    min_call_dp_for_het=None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    if variations[GT_FIELD].shape[0] == 0:
        return numpy.array([]), numpy.array([]), numpy.array([])

    gts = variations[GT_FIELD]
    if is_dataset(gts):
        gts = gts[...]

    # get rid of genotypes with missing alleles
    missing_alleles = gts == MISSING_INT
    miss_gts = numpy.any(missing_alleles, axis=2)
    if min_call_dp_for_het:
        dps = variations[DP_FIELD]
        if is_dataset(dps):
            dps = dps[...]
        low_dp = dps < min_call_dp_for_het
        miss_gts = numpy.logical_or(miss_gts, low_dp)

    # We pack the genotype of a sample that is in third axes as two
    # integers as one integer: 1, 1 -> 11 0, 1 -> 01, 0, 0-> 0
    gts_per_haplo = [
        (gts[:, :, idx].astype(numpy.int16)) * (100**idx)
        for idx in range(gts.shape[2])
    ]
    packed_gts = None
    for gts_ in gts_per_haplo:
        if packed_gts is None:
            packed_gts = gts_
        else:
            packed_gts += gts_
    packed_gts[miss_gts] = MISSING_INT

    different_gts = numpy.unique(packed_gts)
    # Count genotypes, homo, het and alleles
    ploidy = gts.shape[2]
    allele_counts_by_snp = {}
    het_counts_by_snp = None
    homo_counts_by_snp = None
    miss_gt_counts_by_snp = None
    for gt in different_gts:
        count_gt_by_row = row_value_counter_fact(gt)
        gt_counts = count_gt_by_row(packed_gts)

        if gt == MISSING_INT:
            miss_gt_counts_by_snp = gt_counts
            continue

        unpacked_gt = _packed_gt_to_tuple(gt, ploidy)
        if _gt_is_homo(unpacked_gt):
            if homo_counts_by_snp is None:
                homo_counts_by_snp = gt_counts
            else:
                homo_counts_by_snp += gt_counts
        else:
            if het_counts_by_snp is None:
                het_counts_by_snp = gt_counts
            else:
                het_counts_by_snp += gt_counts

        for allele in unpacked_gt:
            if allele not in allele_counts_by_snp:
                allele_counts_by_snp[allele] = numpy.copy(gt_counts)
            else:
                allele_counts_by_snp[allele] += gt_counts
    if het_counts_by_snp is None:
        het_counts_by_snp = numpy.zeros(shape=homo_counts_by_snp.shape)
    het = het_counts_by_snp / (homo_counts_by_snp + het_counts_by_snp)

    allele_counts = numpy.array(list(allele_counts_by_snp.values()))
    max_allele = numpy.amax(allele_counts, axis=0)
    max_allele_freq = max_allele / numpy.sum(allele_counts, axis=0)
    if min_num_genotypes > 0 and miss_gt_counts_by_snp is not None:
        num_samples = gts.shape[1]
        num_calls = -miss_gt_counts_by_snp + num_samples
        enoug_calls = num_calls >= min_num_genotypes
        het = het[enoug_calls]
        max_allele_freq = max_allele_freq[enoug_calls]

    if allele_freq_range is None or het_range is None:
        range_ = None
    else:
        range_ = (allele_freq_range, het_range)

    hist, xedges, yedges = numpy.histogram2d(
        max_allele_freq, het, bins=n_bins, range=range_
    )
    hist = numpy.fliplr(hist)
    return hist, xedges, yedges


def _hist2d_het_allele_freq_by_chunk(
    variations,
    n_bins=DEF_NUM_BINS,
    allele_freq_range=None,
    het_range=None,
    min_call_dp_for_het=None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=None,
):
    fields = [GT_FIELD, ALT_FIELD]
    if min_call_dp_for_het is not None:
        fields.append(DP_FIELD)

    if allele_freq_range is None or het_range is None:
        for var_chunk in variations.iterate_chunks(
            kept_fields=fields, chunk_size=chunk_size
        ):
            res = _hist2d_het_allele_freq(
                var_chunk,
                n_bins=n_bins,
                min_call_dp_for_het=min_call_dp_for_het,
                min_num_genotypes=min_num_genotypes,
            )
            _, xbins, ybins = res
            this_het_range = calc_min_max(ybins)
            this_freq_range = calc_min_max(xbins)
            het_range = _update_range(het_range, this_het_range)
            allele_freq_range = _update_range(allele_freq_range, this_freq_range)

    hist = None
    for var_chunk in variations.iterate_chunks(
        kept_fields=fields, chunk_size=chunk_size
    ):
        res = _hist2d_het_allele_freq(
            var_chunk,
            n_bins=n_bins,
            allele_freq_range=allele_freq_range,
            het_range=het_range,
            min_call_dp_for_het=min_call_dp_for_het,
            min_num_genotypes=min_num_genotypes,
        )
        this_hist, xbins, y_bins = res
        if hist is None:
            hist = this_hist
        else:
            hist = numpy.add(hist, this_hist)
    return hist, xbins, y_bins


def hist2d_het_allele_freq(
    variations,
    n_bins=DEF_NUM_BINS,
    allele_freq_range=None,
    het_range=None,
    min_call_dp_for_het=None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    chunk_size=SNPS_PER_CHUNK,
):
    if chunk_size:
        res = _hist2d_het_allele_freq_by_chunk(
            variations,
            n_bins=n_bins,
            allele_freq_range=allele_freq_range,
            het_range=het_range,
            min_call_dp_for_het=min_call_dp_for_het,
            min_num_genotypes=min_num_genotypes,
            chunk_size=chunk_size,
        )
    else:
        res = _hist2d_het_allele_freq(
            variations,
            n_bins=n_bins,
            allele_freq_range=allele_freq_range,
            het_range=het_range,
            min_call_dp_for_het=min_call_dp_for_het,
            min_num_genotypes=min_num_genotypes,
        )
    hist, xedges, yedges = res
    return hist, xedges, yedges


def calc_field_distrib_for_a_sample(
    variations,
    field,
    sample,
    range_=None,
    n_bins=DEF_NUM_BINS,
    chunk_size=SNPS_PER_CHUNK,
):
    mat = variations[field]

    sample_idx = variations.samples.index(sample) if sample else None

    if range_ is None:
        min_, max_ = calc_min_max(mat, chunk_size=chunk_size, sample_idx=sample_idx)
        if issubclass(mat.dtype.type, numpy.integer) and min_ < 0:
            # we remove the missing data
            min_ = 0
        range_ = min_, max_ + 1

    if chunk_size:
        chunks = iterate_matrix_chunks(
            mat, chunk_size=chunk_size, sample_idx=sample_idx
        )
    else:
        chunks = [mat[:, sample_idx]]

    counts, edges = None, None
    for chunk in chunks:
        chunk_counts, chunk_edges = histogram(chunk, n_bins=n_bins, range_=range_)
        if counts is None:
            counts = chunk_counts
            edges = chunk_edges
        else:
            counts += chunk_counts
            assert numpy.allclose(edges, chunk_edges)
    return counts, edges


def calc_field_distribs_per_sample(
    variations,
    field,
    range_=None,
    n_bins=DEF_NUM_BINS,
    chunk_size=SNPS_PER_CHUNK,
    mask_func=None,
    mask_field=None,
):
    mat = variations[field]
    mask_mat = None
    if mask_field is not None:
        mask_mat = variations[mask_field]

    if range_ is None:
        min_, max_ = calc_min_max(mat, chunk_size=chunk_size)
        if issubclass(mat.dtype.type, numpy.integer) and min_ < 0:
            # we remove the missing data
            min_ = 0
        range_ = min_, max_ + 1

    if chunk_size:
        chunks = iterate_matrix_chunks(mat, chunk_size=chunk_size)
        if mask_mat is not None:
            mask_chunks = iterate_matrix_chunks(mask_mat, chunk_size=chunk_size)
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
    if ro.shape[0] == 0:
        return numpy.array([])
    if len(ro.shape) == len(ao.shape):
        ao = ao.reshape((ao.shape[0], ao.shape[1], 1))
    if is_dataset(ro):
        ro = ro[:]
    read_counts = numpy.append(ro.reshape((ao.shape[0], ao.shape[1], 1)), ao, axis=2)
    read_counts[read_counts == MISSING_VALUES[int]] = 0
    total_counts = numpy.sum(read_counts, axis=2)
    with numpy.errstate(invalid="ignore"):
        depth_maf = numpy.max(read_counts, axis=2) / total_counts
    depth_maf[total_counts < min_depth] = numpy.nan
    return depth_maf


def calc_maf_depth_distribs_per_sample(
    variations,
    min_depth=DEF_MIN_DEPTH,
    n_bins=DEF_NUM_BINS * 2,
    chunk_size=SNPS_PER_CHUNK,
):
    range_ = 0, 1
    if chunk_size:
        chunks = variations.iterate_chunks(
            kept_fields=[AO_FIELD, RO_FIELD], chunk_size=chunk_size
        )
    else:
        chunks = [variations]

    histograms = None
    bins = None
    for chunk in chunks:
        chunk_hists = None
        maf_depth = _calc_maf_depth(chunk, min_depth)
        for sample_idx in range(len(variations.samples)):
            chunk_hist, bins = histogram(
                maf_depth[:, sample_idx], n_bins=n_bins, range_=range_
            )
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
    def __init__(self, chrom, pos, stat, window_size=None, step=1, take_windows=True):
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
                window = numpy.logical_and(pos >= i, pos < i + self.window_size)
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

        return PositionalStatsCalculator(
            chrom, pos, stat, self.window_size, self.step, False
        )

    def _iterate_chroms(self):
        for chrom_name in self.chrom_names:
            mask = numpy.logical_and(
                self.chrom == chrom_name, numpy.logical_not(numpy.isnan(self.stat))
            )
            values = self.stat[mask]
            pos = self.pos[mask]
            if values.shape != () and values.shape != (0,):
                yield chrom_name, pos, values

    def _get_track_definition(self, track_type, name, description, **kwargs):
        types = {"wig": "wiggle_0", "bedgraph": "bedGraph"}
        track_line = 'track type={} name="{}" description="{}"'
        track_line = track_line.format(types[track_type], name, description)
        for key, value in kwargs:
            track_line += " {}={}".format(key, value)
        return track_line

    def to_wig(self):
        stat = self.stat
        span = self.window_size
        if stat.shape[0] != self.pos.shape[0] or stat.shape[0] != self.chrom.shape[0]:
            raise ValueError("Stat does not have the same size as pos")
        for chrom_name, pos, values in self._iterate_chroms():
            try:
                chrom_name = chrom_name.decode()
            except AttributeError:
                pass
            chrom_line = "variableStep chrom={}".format(chrom_name)
            variable_step = True
            if span is not None:
                chrom_line = "fixedStep chrom={} start={} span={} step={}"
                chrom_line = chrom_line.format(chrom_name, pos[0], span, self.step)
                variable_step = False
            yield chrom_line
            # When containing only one value, it is not iterable
            if values.shape != () and pos.shape != ():
                for pos, value in self._calc_chrom_window_stat(pos, values):
                    if variable_step:
                        yield "{} {}".format(pos, value)
                    else:
                        yield str(value)
            else:
                if variable_step:
                    yield "{} {}".format(pos, value)
                else:
                    yield str(value)

    def to_bedGraph(self):
        window_size = self.window_size
        stat = self.stat
        if window_size is None:
            window_size = 1
        if stat.shape[0] != self.pos.shape[0] or stat.shape[0] != self.chrom.shape[0]:
            raise ValueError("Stat does not have the same size as pos")
        for chrom_name, pos, values in self._iterate_chroms():
            try:
                chrom_name = chrom_name.decode()
            except AttributeError:
                pass
            # When containing only one value, it is not iterable
            if values.shape != () and pos.shape != ():
                for pos, value in self._calc_chrom_window_stat(pos, values):
                    yield "{} {} {} {}".format(
                        chrom_name, pos, pos + window_size, value
                    )
            else:
                yield "{} {} {} {}".format(chrom_name, pos, pos + window_size, value)

    def write(
        self,
        fhand,
        track_name,
        track_description,
        buffer_size=1000,
        track_type="bedgraph",
        **kwargs
    ):
        get_lines = {"wig": self.to_wig, "bedgraph": self.to_bedGraph}
        buffer = (
            self._get_track_definition(
                track_type, track_name, track_description, **kwargs
            )
            + "\n"
        )
        lines = 1
        for line in get_lines[track_type]():
            lines += 1
            buffer += line + "\n"
            if lines == buffer_size:
                fhand.write(buffer)
                buffer = ""
                lines = 0
        if lines != buffer_size:
            fhand.write(buffer)
        fhand.flush()


def _calc_r2(gts):
    gts = GenotypeArray(gts)
    gns = gts.to_n_alt(fill=MISSING_VALUES[int])
    return rogers_huff_r(gns) ** 2


def calc_r2_windows(variations, window_size, step=None):
    if step is None:
        step = window_size
    chrom = variations[CHROM_FIELD]
    if is_dataset(chrom):
        chrom = chrom[:]
    pos = variations[POS_FIELD]
    if is_dataset(pos):
        pos = pos[:]
    gts = variations[GT_FIELD]

    window_chrom = []
    window_pos = []
    window_r2 = []

    chrom_names = numpy.unique(chrom)

    for chrom_name in chrom_names:
        chrom_mask = chrom == chrom_name
        chrom_pos = pos[chrom_mask]
        if isinstance(chrom_pos, int) or len(chrom_pos.shape) == 0:
            continue
        for position in range(chrom_pos[0], chrom_pos[-1], step):
            start, stop = position, position + window_size
            window_mask = numpy.logical_and(pos > start, pos <= stop)
            chrom_window_mask = numpy.logical_and(window_mask, chrom_mask)

            if not numpy.any(chrom_window_mask):
                continue

            r2s = _calc_r2(gts[chrom_window_mask, :, :])
            r2 = numpy.mean(r2s) if r2s.size else float("nan")

            window_chrom.append(chrom_name)
            window_pos.append(start)
            window_r2.append(r2)
    return (numpy.array(window_chrom), numpy.array(window_pos), numpy.array(window_r2))


def _call_is_hom_for_sample(gts):
    if gts.shape[0] == 0:
        return numpy.array([]), numpy.array([])

    if is_dataset(gts):
        gts = gts[:]

    is_hom = numpy.full(gts.shape[:-1], True, dtype=numpy.bool)
    for idx in range(1, gts.shape[1]):
        is_hom = numpy.logical_and(gts[:, idx] == gts[:, idx - 1], is_hom)

    are_missing = numpy.sum(gts == MISSING_INT, axis=1) > 0
    is_hom[are_missing] = False
    return is_hom, are_missing


def calc_call_dp_distrib_for_a_sample(
    variations, sample, range_=None, n_bins=DEF_NUM_BINS, chunk_size=SNPS_PER_CHUNK
):
    dps = variations["/calls/DP"]
    gts = variations[GT_FIELD]

    sample_idx = variations.samples.index(sample) if sample else None

    if range_ is None:
        min_, max_ = calc_min_max(dps, chunk_size=chunk_size, sample_idx=sample_idx)
        if issubclass(dps.dtype.type, numpy.integer) and min_ < 0:
            # we remove the missing data
            min_ = 0
        range_ = min_, max_ + 1

    if chunk_size:
        dp_chunks = iterate_matrix_chunks(
            dps, chunk_size=chunk_size, sample_idx=sample_idx
        )
        gt_chunks = iterate_matrix_chunks(
            gts, chunk_size=chunk_size, sample_idx=sample_idx
        )
    else:
        dp_chunks = [dps[:, sample_idx]]
        gt_chunks = [gts[:, sample_idx, :]]

    hom_counts, edges = None, None
    het_counts = None
    for dp_chunk, gt_chunk in zip(dp_chunks, gt_chunks):
        are_hom, are_missing = _call_is_hom_for_sample(gt_chunk)
        are_het = numpy.logical_and(
            numpy.logical_not(are_hom), numpy.logical_not(are_missing)
        )
        are_hom = are_hom.astype(int)
        are_het = are_het.astype(int)

        hom_res = histogram(dp_chunk, n_bins=n_bins, range_=range_, weights=are_hom)
        het_res = histogram(dp_chunk, n_bins=n_bins, range_=range_, weights=are_het)
        chunk_hom_counts, chunk_hom_edges = hom_res
        chunk_het_counts, chunk_het_edges = het_res

        if hom_counts is None:
            hom_counts = chunk_hom_counts
            het_counts = chunk_het_counts
            edges = chunk_hom_edges
        else:
            hom_counts += chunk_hom_counts
            het_counts += chunk_het_counts
            assert numpy.allclose(edges, chunk_hom_edges)
        het_counts += chunk_het_counts
        assert numpy.allclose(edges, chunk_het_edges)

    return {"hom": hom_counts, "het": het_counts}, edges


def _calc_depth_mean_by_sample(variations):
    mat = variations[DP_FIELD]
    if is_dataset(mat):
        mat = mat[:]
    mat = mat.astype(numpy.float)
    mat[mat == 0] = numpy.nan

    smpl_mean = numpy.mean(mat, axis=0)
    return smpl_mean


def calc_depth_mean_by_sample(variations, chunk_size=SNPS_PER_CHUNK):
    dps = variations["/calls/DP"]
    if chunk_size:
        chunks = iterate_matrix_chunks(dps, chunk_size=chunk_size)
    else:
        chunks = [dps]

    sums = None
    for chunk in chunks:
        chunk_sums = numpy.sum(chunk, axis=0)
        if sums is None:
            sums = chunk_sums
        else:
            sums += chunk_sums
    means = sums / dps.shape[0]
    return means


@lru_cache(maxsize=128)
def _get_vector_with_1_div_i(vector_length, idx):
    value_i = 1 / idx
    value_ii = 1 / (idx**2)
    return (numpy.full(vector_length, value_i), numpy.full(vector_length, value_ii))


def _calc_a1(num_seqs_with_data):
    max_num_seqs_with_data = max(num_seqs_with_data)

    cum_i_result = numpy.ones_like(num_seqs_with_data, dtype=float)
    cum_ii_result = numpy.ones_like(num_seqs_with_data, dtype=float)
    for idx in range(2, max_num_seqs_with_data):
        res = _get_vector_with_1_div_i(cum_i_result.shape[0], idx)

        value_i_to_sum, value_ii_to_sum = res
        pos_with_lower_value_than_idx = num_seqs_with_data < idx + 1
        value_i_to_sum[pos_with_lower_value_than_idx] = 0
        value_ii_to_sum[pos_with_lower_value_than_idx] = 0

        cum_i_result += value_i_to_sum
        cum_ii_result += value_ii_to_sum

    cum_i_result[num_seqs_with_data == 0] = numpy.nan
    cum_i_result[num_seqs_with_data == 1] = 0
    cum_ii_result[num_seqs_with_data == 0] = numpy.nan
    cum_ii_result[num_seqs_with_data == 1] = 0

    return cum_i_result, cum_ii_result


def calc_tajima_d_and_pi(
    variations,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    ploidy=None,
    debug=False,
    min_num_segregating_variations=10,
):
    # Implemented following: Genome-wide DNA polymorphism analyses using VariScan
    # https://doi.org/10.1186/1471-2105-7-409

    gts = variations[GT_FIELD]
    if ploidy is None:
        ploidy = variations.ploidy

    if gts.shape[2] > 1:
        random_index_for_ploidy = numpy.random.randint(gts.shape[2], size=gts.shape[1])

        # choose one haplotype at random for every sample
        seqs_choice = [gts[:, :, idx] for idx in range(gts.shape[2])]
        seqs = numpy.choose(random_index_for_ploidy, seqs_choice)
    else:
        gt_shape = gts.shape
        seqs = gts.reshape((gt_shape[0], gt_shape[1]))
    # print(seqs)
    # print(seqs.shape)

    non_missing_gts = seqs != MISSING_INT
    num_seqs_with_data_per_snp = numpy.sum(non_missing_gts, axis=1)
    a1_per_snp, a2_per_snp = _calc_a1(num_seqs_with_data_per_snp)

    avg_a1 = numpy.mean(a1_per_snp)
    avg_a2 = numpy.mean(a2_per_snp)
    if debug:
        print("num_seqs_with_data_per_snp")
        print(num_seqs_with_data_per_snp)
        print("a1_per_snp")
        print(a1_per_snp)
        print("a2_per_snp")
        print(a2_per_snp)
        print("avg_a1", avg_a1)
        print("avg_a2", avg_a2)

    snp_has_enough_data = num_seqs_with_data_per_snp > min_num_genotypes
    num_snps_with_enough_data = numpy.sum(snp_has_enough_data)
    avg_num_seqs_with_enough_data = numpy.mean(num_seqs_with_data_per_snp)

    allele_counts, _ = counts_and_allels_by_row(seqs, missing_value=MISSING_INT)

    snp_is_segregating = numpy.sum(allele_counts > 0, axis=1) > 1

    num_snp_segregating = numpy.sum(
        numpy.logical_and(snp_is_segregating, num_snps_with_enough_data)
    )
    if num_snp_segregating < min_num_segregating_variations:
        raise ValueError("Not enough segregating variations with data")

    if debug:
        print("num snps with enough data", num_snps_with_enough_data)
        print("avg_num_seqs_with_enough_data", avg_num_seqs_with_enough_data)
        print("num_snp_segregating (S)", num_snp_segregating)

    tot_theta = num_snp_segregating / avg_a1

    exp_het_per_snp = calc_unbias_expected_het2(
        variations, ploidy=ploidy, min_num_genotypes=min_num_genotypes
    )
    tot_exp_het = numpy.sum(exp_het_per_snp)
    num_sites = seqs.shape[0]

    pi_per_site = tot_exp_het / num_sites

    theta_per_site = tot_theta / num_sites

    if debug:
        print("tot_theta", tot_theta)
        print("exp_het_per_snp", exp_het_per_snp)
        print("tot_exp_het", tot_exp_het)
        print("pi_per_site ", pi_per_site)
        print("theta_per_site", theta_per_site)

    n = avg_num_seqs_with_enough_data
    a1 = avg_a1
    a2 = avg_a2
    S = num_snp_segregating

    b1 = (n + 1) / (3 * (n - 1))
    b2 = (2 * (n * n + n + 3)) / (9 * n * (n - 1))
    e1 = (b1 - (1 / a1)) / a1
    e2 = (b2 - (n + 2) / (a1 * n) + (a2 / (a1 * a1))) / ((a1 * a1) + a2)

    numerator = tot_exp_het - tot_theta
    denominator2 = (e1 * S) + (e2 * S * (S - 1))
    if denominator2 < 0:
        raise ValueError("Error when calculating tajima, sqrt of a negative number")
    denominator = numpy.sqrt(denominator2)

    tajima_d = numerator / denominator

    return {"tajima_d": tajima_d, "pi": pi_per_site, "theta": theta_per_site}
