

import numpy
from variation import GT_FIELD, MISSING_INT
from variation.matrix.stats import counts_and_allels_by_row
from variation.variations.filters import SampleFilter, FLT_VARS


def _filter_samples_for_stats(variations, samples=None):
    if samples is None:
        vars_for_stat = variations
    else:
        filter_samples = SampleFilter(samples)
        vars_for_stat = filter_samples(variations)[FLT_VARS]
    return vars_for_stat


def is_variable(variations, samples):
    sample_variation = _filter_samples_for_stats(variations, samples=samples)
    gts = sample_variation[GT_FIELD]
    counts, _ = counts_and_allels_by_row(gts, missing_value=MISSING_INT)
    all_missing_gts = numpy.sum(counts, axis=1) == 0
    is_variable_ = (numpy.sum(counts > 0, axis=1) > 1).astype(int)
    is_variable_[all_missing_gts] = MISSING_INT
    return is_variable_
