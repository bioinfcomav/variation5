from itertools import combinations

import numpy
from functools import reduce
import operator

from variation import MISSING_VALUES
from variation.matrix.stats import counts_by_row, row_value_counter_fact
from variation.matrix.methods import append_matrix, calc_min_max, fill_array
from variation.variations.index import PosIndex
import pandas

CHUNK_SIZE = 200
MIN_NUM_GENOTYPES_FOR_POP_STAT = 10


def _remove_nans(mat):
    return mat[~numpy.isnan(mat)]


def _remove_infs(mat):
    return mat[~numpy.isinf(mat)]


def _calc_items_in_row(mat):
    num_items_per_row = reduce(operator.mul, mat.shape[1:], 1)
    return num_items_per_row


# def plot_hist_mafs(var_mat, fhand=None, no_interactive_win=False):
#     mafs = calc_mafs(var_mat)
#     mafs = _remove_nans(mafs)
#     return plot_histogram(mafs, fhand=fhand,
#                           no_interactive_win=no_interactive_win)


def calc_stat_by_chunk(var_matrices, function, reduce_funct=None,
                       matrix_transform=None,
                       matrix_transform_axis=None):
    stats = None
    chunks = var_matrices.iterate_chunks(kept_fields=function.required_fields)
    for chunk in chunks:
        chunk_stats = function(chunk)
        if matrix_transform:
            try:
                chunk_stats = matrix_transform(chunk_stats,
                                               axis=matrix_transform_axis)
            except TypeError:
                chunk_stats = matrix_transform(chunk_stats)
        if stats is None:
            stats = numpy.copy(chunk_stats)
        elif reduce_funct is not None:
            stats = reduce_funct(stats, chunk_stats)
        else:
            append_matrix(stats, chunk_stats)
    return stats


def _calc_stat(var_matrices, function, reduce_funct=None, matrix_transform=None,
               matrix_transform_axis=None, by_chunk=True, mask=None):
    if by_chunk:
        return calc_stat_by_chunk(var_matrices=var_matrices, function=function,
                                  reduce_funct=reduce_funct,
                                  matrix_transform=matrix_transform,
                                  matrix_transform_axis=matrix_transform_axis)
    else:
        if matrix_transform is not None:
            return matrix_transform(function(var_matrices),
                                    matrix_transform_axis)
        else:
            return function(var_matrices)

class _MafCalculator:
    def __init__(self, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
        self.required_fields = ['/calls/GT']
        self.min_num_genotypes = min_num_genotypes

    def __call__(self, variations):
        gts = variations['/calls/GT']
        gt_counts = counts_by_row(gts, missing_value=MISSING_VALUES[int])
        max_ = numpy.amax(gt_counts, axis=1)
        sum_ = numpy.sum(gt_counts, axis=1)
        mafs_gt = max_ / sum_
        mafs_gt[sum_ < self.min_num_genotypes] = numpy.nan
        return mafs_gt


class _MafDepthCalculator:
    def __init__(self, min_num_genotypes=0):
        self.required_fields = ['/calls/AO', '/calls/RO']
        self.min_num_genotypes = min_num_genotypes

    def __call__(self, variations):
        ro = variations['/calls/RO']
        ao = variations['/calls/AO']
        read_counts = numpy.append(ro.reshape((ao.shape[0], ao.shape[1],
                                               1)), ao, axis=2)
        read_counts[read_counts == MISSING_VALUES[int]] = 0
        total_counts = numpy.sum(read_counts, axis=2)
        depth_maf = numpy.max(read_counts, axis=2) / total_counts
        depth_maf[total_counts < self.min_num_genotypes] = 0
        depth_maf[numpy.isnan(depth_maf)] = 0
        return depth_maf


class _MissingGTCalculator:
    def __init__(self, rate=True):
        self.required_fields = ['/calls/GT']
        self.rate = rate

    def __call__(self, chunk):
        gts = chunk['/calls/GT']
        missing = _missing_gt_counts(gts)
        if self.rate:
            num_items_per_row = _calc_items_in_row(gts)
            result = missing / num_items_per_row
        else:
            result = missing
        return result


def _missing_gt_counts(chunk):
    count_missing = row_value_counter_fact(MISSING_VALUES[int], ratio=False)
    return count_missing(chunk)


class _ObsHetCalculator:
    def __call__(self, variations):
        #TODO: min_num_genotypes
        gts = variations['/calls/GT']
        is_het = _is_het(gts)
        missing_gts = _is_missing(gts)
        called_gts = numpy.sum(missing_gts == 0, axis=self.axis)
        het = numpy.divide(numpy.sum(is_het, axis=self.axis), called_gts)
        return het


def _is_het(gts):
    is_het = gts[:, :, 0] != gts[:, :, 1]
    missing_gts = _is_missing(gts)
    is_het[missing_gts] = False
    return is_het


def _is_missing(gts):
    missing_gts = numpy.any(gts == -1, axis=2)
    return missing_gts


def _is_hom(gts):
    is_hom = gts[:, :, 0] == gts[:, :, 1]
    missing_gts = _is_missing(gts)
    is_hom[missing_gts] = False
    return is_hom


def _is_called(gts):
    return numpy.logical_not(_is_missing(gts))


def _is_eq_hi_dp(variations, depth):
    return variations['/calls/DP'] >= depth


class _CalledHigherDepthMasker:
    def __init__(self, depth):
        self.depth = depth
        self.required_fields = ['/calls/GT', '/calls/DP']

    def __call__(self, variations):
        is_called = _is_called(variations['/calls/GT'])
        is_eq_hi_dp = _is_eq_hi_dp(variations, self.depth)
        is_called_eq_hi_than_dp = numpy.logical_and(is_called, is_eq_hi_dp)
        return is_called_eq_hi_than_dp


class _CalledHigherDepthDistribCalculator:
    def __init__(self, depth, calc_distrib):
        self.depth = depth
        self.required_fields = ['/calls/GT', '/calls/DP']
        self.calc_distrib = calc_distrib

    def __call__(self, variations):
        is_called_higher_depth = _CalledHigherDepthMasker(self.depth)
        n_called_per_snp = numpy.sum(is_called_higher_depth(variations),
                                     axis=1)
        n_called_per_snp = n_called_per_snp.reshape((1, n_called_per_snp.shape[0]))
        return self.calc_distrib(n_called_per_snp)


class GQualityByDepthDistribCalculator:
    def __init__(self, depth, calc_distrib):
        self.depth = depth
        self.required_fields = ['/calls/GQ', '/calls/DP']
        self.calc_distrib = calc_distrib

    def __call__(self, variations):
        gqs = variations['/calls/GQ']
        gqs = gqs[variations['/calls/DP'] == self.depth]
        if gqs.shape[0] == 0:
            return numpy.zeros((1, self.calc_distrib.max_value+1))
        else:
            return self.calc_distrib(gqs.reshape((1, gqs.shape[0])))


class _ObsHetCalculatorBySnps(_ObsHetCalculator):
    def __init__(self, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
        self.axis = 1
        self.required_fields = ['/calls/GT']
        self.min_num_genotypes = min_num_genotypes


class _ObsHetCalculatorBySample(_ObsHetCalculator):
    def __init__(self,  min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
        self.axis = 0
        self.required_fields = ['/calls/GT']
        self.min_num_genotypes = min_num_genotypes


class _IntDistributionCalculator():
    def __init__(self, max_value, fields=None, fillna=None, per_sample=True,
                 mask_function=None, mask_field=None):
        self.required_fields = fields
        self.max_value = max_value
        self.fillna = fillna
        self.per_sample = per_sample
        self.mask_function = mask_function
        self.mask_field = mask_field

    def __call__(self, variations):
        if self.required_fields is not None:
            mat = variations[self.required_fields[0]]
        else:
            mat = variations
        counts_matrix = _calc_counts_matrix(mat, variations,
                                            max_value=self.max_value,
                                            mask_function=self.mask_function,
                                            per_sample=self.per_sample,
                                            fillna=self.fillna,
                                            mask_field=self.mask_field)
        return counts_matrix


def _calc_counts_matrix(mat, variations, max_value, mask_function=None,
                        per_sample=True, fillna=None, mask_field=None):
    mask = None
    if 'float' in str(mat.dtype):
            is_nan = numpy.isnan(mat)
            mat[is_nan] = MISSING_VALUES[int]
            mat = mat.astype(int)
    if mask_function is not None:
        if mask_field is None:
            raise ValueError('A field is required for mask function')
        mask_mat = variations[mask_field]
        mask = mask_function(mask_mat)
        if per_sample:
            mask = mask.transpose()
    if per_sample:
        mat = mat.transpose()
    counts_matrix = None
    for i, row_mat in enumerate(mat):
        if mask is not None:
            row_mat = row_mat[mask[i]]
        if fillna is None:
            is_missing = row_mat == MISSING_VALUES[int]
            row_mat = row_mat[is_missing == False]
        else:
            row_mat[row_mat == MISSING_VALUES[int]] = fillna
        row_mat = row_mat[row_mat <= max_value]
        row_mat_counts = numpy.bincount(row_mat)
        if row_mat_counts.shape[0] <= max_value:
            row_mat_counts = fill_array(row_mat_counts, max_value+1)
        if counts_matrix is None:
            counts_matrix = numpy.array([row_mat_counts])
        else:
            counts_matrix = numpy.append(counts_matrix,
                                         [row_mat_counts],
                                         axis=0)
    return counts_matrix


class _IntDistribution2DCalculator():
    def __init__(self, max_values, fields=None, fillna=0,
                 mask_function=None, mask_field=None,
                 transform_func=[None, None], weights_field=None):
        self.required_fields = fields
        self.max_values = max_values
        self.fillna = fillna
        self.mask_function = mask_function

        def copy_mat(x):
            return x.copy()
        self.transform_func = [copy_mat if f is None else f
                               for f in transform_func]
        self.weights_field = weights_field
        self.mask_field = mask_field

    def __call__(self, variations):
        if self.required_fields is not None:
            if len(self.required_fields) != 2:
                raise ValueError('2 fields are required for 2D distribution')
            mat1 = variations[self.required_fields[0]]
            mat2 = variations[self.required_fields[1]]
        else:
            mat1, mat2 = variations
        if self.weights_field is not None:
            weights = variations[self.weights_field]
        else:
            weights = None
        mat1[mat1 == MISSING_VALUES[int]] = self.fillna
        mat2[mat2 == MISSING_VALUES[int]] = self.fillna
        mat1 = self.transform_func[0](mat1)
        mat2 = self.transform_func[1](mat2)
        if self.mask_function is not None:
            if self.mask_field is None:
                raise ValueError('A field is required for mask function')
            mask = self.mask_function(variations[self.mask_field])
            mat1 = mat1[mask]
            mat2 = mat2[mask]
        else:
            mat1 = mat1.reshape((mat1.shape[0]*mat1.shape[1],))
            mat2 = mat2.reshape((mat2.shape[0]*mat2.shape[1],))
            if weights is not None:
                weights = weights.reshape((weights.shape[0]*weights.shape[1],))
        distrib, _, _ = numpy.histogram2d(mat1, mat2,
                                          [x+1 for x in self.max_values],
                                          [[0, self.max_values[0]],
                                           [0, self.max_values[1]]],
                                          weights=weights)
        if weights is not None:
            counts_matrix, _, _ = numpy.histogram2d(mat1, mat2,
                                                    [x+1 for x in self.max_values],
                                                    [[0, self.max_values[0]],
                                                     [0, self.max_values[1]]])
            distrib = distrib / counts_matrix
        return distrib


def calc_allele_obs_distrib_2D(variations, max_values=[None, None],
                               by_chunk=True, mask_function=None):
    required_fields = ['/calls/RO', '/calls/AO']
    for i, field in enumerate(required_fields):
        if max_values[i] is None:
            _, max_values[i] = calc_min_max(variations[field])
    transform_func = [None, lambda x: numpy.max(x, axis=2)]
    calc_distr = _IntDistribution2DCalculator(max_values=max_values,
                                              fields=required_fields,
                                              transform_func=transform_func,
                                              mask_function=mask_function)
    distrib = _calc_stat(variations, calc_distr, reduce_funct=numpy.add,
                         by_chunk=by_chunk)
    return distrib


def calc_allele_obs_gq_distrib_2D(variations, max_values=[None, None],
                                  by_chunk=True, mask_function=None):
    required_fields = ['/calls/RO', '/calls/AO']
    for i, field in enumerate(required_fields):
        if max_values[i] is None:
            _, max_values[i] = calc_min_max(variations[field])
    transform_func = [None, lambda x: numpy.max(x, axis=2)]
    calc_distr = _IntDistribution2DCalculator(max_values=max_values,
                                              fields=required_fields,
                                              transform_func=transform_func,
                                              mask_function=mask_function,
                                              weights_field='/calls/GQ')
    distrib = _calc_stat(variations, calc_distr, reduce_funct=numpy.add,
                         by_chunk=by_chunk)
    return distrib


def _calc_distribution(variations, fields, max_value, fillna=None,
                       by_chunk=True, per_sample=True, mask_function=None,
                       mask_field=None):

    calc_distr = _IntDistributionCalculator(max_value=max_value,
                                            fields=fields,
                                            fillna=fillna,
                                            per_sample=per_sample,
                                            mask_function=mask_function,
                                            mask_field=mask_field)
    return _calc_stat(variations, calc_distr, reduce_funct=numpy.add,
                      by_chunk=by_chunk)


def calc_depth_cumulative_distribution_per_sample(variations, max_depth=None,
                                                  by_chunk=True,
                                                  mask_function=None,
                                                  mask_field=None):
    if max_depth is None:
        _, max_depth = calc_min_max(variations['/calls/DP'])
    distributions = _calc_distribution(variations, fields=['/calls/DP',
                                                           '/calls/GT'],
                                       max_value=max_depth, fillna=0,
                                       by_chunk=by_chunk,
                                       mask_function=mask_function,
                                       mask_field=mask_field)
    dp_cumulative_distr = numpy.cumsum(distributions, axis=1)
    return distributions, dp_cumulative_distr


def calc_called_gts_distrib_per_depth(variations, depths, by_chunk=True):
    distributions = None
    max_called_per_snp = variations['/calls/GT'].shape[1]
    calc_distrib = _IntDistributionCalculator(max_value=max_called_per_snp,
                                              per_sample=False)
    for depth in depths:
        calc_call_hi_dp_distr = _CalledHigherDepthDistribCalculator(depth=depth,
                                                                    calc_distrib=calc_distrib)
        called_gt_per_snp_distrib = _calc_stat(variations,
                                               calc_call_hi_dp_distr,
                                               reduce_funct=numpy.add,
                                               by_chunk=by_chunk)
        if distributions is None:
            distributions = numpy.copy(called_gt_per_snp_distrib)
        else:
            append_matrix(distributions, called_gt_per_snp_distrib)
    dp_cumulative_distr = numpy.cumsum(distributions, axis=1)
    return distributions, dp_cumulative_distr


def calc_quality_by_depth(variations, depths, by_chunk=True):
    distributions, gq_cumulative_distrs = None, None
    _, max_value = calc_min_max(_remove_nans(variations['/calls/GQ']))
    calculate_distribution = _IntDistributionCalculator(max_value=max_value,
                                                        per_sample=False)
    for depth in depths:
        calc_gq_by_dp_distrib = GQualityByDepthDistribCalculator(depth=depth,
                                         calc_distrib=calculate_distribution)
        distribution = _calc_stat(variations, calc_gq_by_dp_distrib,
                                  reduce_funct=numpy.add,
                                  by_chunk=by_chunk)
        gq_cumulative_distr = numpy.cumsum(distribution, axis=1)
        if distributions is None:
            distributions = numpy.copy(distribution)
            gq_cumulative_distrs = numpy.copy(gq_cumulative_distr)
        else:
            append_matrix(distributions, distribution)
            append_matrix(gq_cumulative_distrs, gq_cumulative_distr)
    distributions = distributions.reshape((len(depths), max_value+1))
    gq_cumulative_distrs = gq_cumulative_distrs.reshape((len(depths),
                                                         max_value+1))
    return distributions, gq_cumulative_distrs


def calc_gq_cumulative_distribution_per_sample(variations, by_chunk=True,
                                               mask_function=None,
                                               mask_field=None):
    _, max_gq = calc_min_max(_remove_nans(variations['/calls/GQ']))
    distributions = _calc_distribution(variations, fields=['/calls/GQ',
                                                           '/calls/GT'],
                                       max_value=max_gq, by_chunk=by_chunk,
                                       mask_function=mask_function,
                                       mask_field=mask_field)
    gq_cumulative_distr = numpy.cumsum(distributions, axis=1)
    return distributions, gq_cumulative_distr


def calc_hq_cumulative_distribution_per_sample(variations, by_chunk=True,
                                               mask_function=None):
    try:
        _, max_hq = calc_min_max(_remove_nans(variations['/calls/HQ']))
    except KeyError:
        return None
    distributions = _calc_distribution(variations, fields=['/calls/HQ',
                                                           '/calls/GT'],
                                       max_value=max_hq,
                                       by_chunk=by_chunk,
                                       mask_function=mask_function)
    hq_cumulative_distr = numpy.cumsum(distributions, axis=1)
    return distributions, hq_cumulative_distr


def calc_snv_density_distribution(variations, window):
    density = calc_snp_density(variations, window)
    distribution = numpy.bincount(density)
    return distribution


def calculate_maf_depth_distribution(variations, by_chunk=True):
    maf_depths = _calc_stat(variations, _MafDepthCalculator(),
                            by_chunk=by_chunk)
    calc_distribution = _IntDistributionCalculator(max_value=100)
    maf_depths_distrib = calc_distribution(maf_depths*100)
    return maf_depths_distrib


def calculate_maf_distribution(variations, by_chunk=True):
    maf = _calc_stat(variations, _MafCalculator(), by_chunk=by_chunk)
    calc_distribution = _IntDistributionCalculator(max_value=100)
    maf = calc_distribution((maf * 100).reshape(maf.shape[0], 1))
    return maf


class GenotypeStatsCalculator:
    def __init__(self):
        self.required_fields = ['/calls/GT']

    def __call__(self, variations):
        gts = variations['/calls/GT']
        het = numpy.sum(_is_het(gts), axis=0)
        missing = numpy.sum(_is_missing(gts), axis=0)
        gts_alt = numpy.copy(gts)
        gts_alt[gts_alt == 0] = -1
        alt_hom = numpy.sum(_is_hom(gts_alt), axis=0)
        ref_hom = numpy.sum(_is_hom(gts), axis=0) - alt_hom
        return numpy.array([ref_hom, het, alt_hom, missing])


class _CalledGTCalculator:
    def __init__(self, rate=True):
        self.required_fields = ['/calls/GT']
        self.rate = rate

    def __call__(self, chunk):
        gts = chunk['/calls/GT']
        gt_counts = _called_gt_counts(gts)
        gt = numpy.sum(gt_counts)
        if self.rate:
            gt_result = numpy.divide(gt_counts, gt)
        else:
            gt_result = gt_counts
        return gt_result


def _called_gt_counts(gts):
    return _calc_items_in_row(gts) - _missing_gt_counts(gts)


def calc_snp_density(variations, window):
    dens = []
    index_right = 0
    dic_index = PosIndex(variations)
    n_snps = variations['/variations/chrom'].shape
    for i in range(n_snps[0]):
        chrom = variations['/variations/chrom'][i]
        pos = variations['/variations/pos'][i]
        pos_right = pos - window
        pos_left = window + pos
        if pos_right > variations['/variations/pos'][0]:
            index_right = dic_index.index_pos(chrom, pos_right)
        index_left = dic_index.index_pos(chrom, pos_left)
        dens.append(index_left - index_right + 1)
    return numpy.array(dens)


class _AlleleFreqCalculator:
    def __init__(self, max_num_allele):
        self.required_fields = ['/calls/GT']
        self.max_num_allele = max_num_allele

    def __call__(self, chunk):
        gts = chunk['/calls/GT']
        allele_counts = counts_by_row(gts, MISSING_VALUES[int])
        total_counts = numpy.sum(allele_counts, axis=1)
        allele_freq = allele_counts/total_counts[:, None]
        if allele_freq.shape[1] < self.max_num_allele:
            allele_freq = fill_array(allele_freq, self.max_num_allele, dim=1)
        return allele_freq


def calc_expected_het(alleles_freq):
    exp_het = None
    for index_1, index_2 in combinations(range(alleles_freq.shape[1]), 2):
        if exp_het is None:
            exp_het = 2 * alleles_freq[:, index_1] * alleles_freq[:, index_2]
        else:
            exp_het += 2 * alleles_freq[:, index_1] * alleles_freq[:, index_2]
    return exp_het


def calc_inbreeding_coeficient(obs_het, exp_het):
    return 1 - (obs_het / exp_het)
