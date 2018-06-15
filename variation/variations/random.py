
import numpy

from variation import GT_FIELD, MISSING_INT
from variation.variations.vars_matrices import VariationsArrays


def set_gts_to_missing_old(variations, percent_of_non_missing_gts_to_missing):
    gts = variations[GT_FIELD]

    non_missing_gt_mask = numpy.all(gts != MISSING_INT, axis=2)

    index_for_non_missing_gts = numpy.where(non_missing_gt_mask)

    num_of_non_missing_gts = index_for_non_missing_gts[0].shape[0]
    num_gts_to_keep = round(num_of_non_missing_gts * percent_of_non_missing_gts_to_missing / 100)

    gts_to_keep = numpy.random.choice(num_of_non_missing_gts,
                                      num_gts_to_keep,
                                      replace=False)
    assert gts_to_keep.shape[0] == num_gts_to_keep

    gts = variations[GT_FIELD]
    gts_to_set_missing = [index[gts_to_keep] for index in index_for_non_missing_gts]
    gts[gts_to_set_missing] = MISSING_INT


def _set_gts_to_missing(chunk, gt_rate_to_missing):
    gts = chunk[GT_FIELD]
    non_missing_gt_mask = numpy.all(gts != MISSING_INT, axis=2)

    index_for_non_missing_gts = numpy.where(non_missing_gt_mask)

    num_of_non_missing_gts = index_for_non_missing_gts[0].shape[0]
    num_gts_to_keep = round(num_of_non_missing_gts * gt_rate_to_missing)

    gts_to_keep = numpy.random.choice(num_of_non_missing_gts,
                                      num_gts_to_keep,
                                      replace=False)
    assert gts_to_keep.shape[0] == num_gts_to_keep

    gts_to_set_missing = [index[gts_to_keep] for index in index_for_non_missing_gts]
    gts[gts_to_set_missing] = MISSING_INT
    return chunk


def copy_setting_gts_to_missing(in_vars, gt_rate_to_missing,
                                out_vars=None, chunk_size=None):
    if out_vars is None:
        out_vars = VariationsArrays()

    chunks = in_vars.iterate_chunks(chunk_size=chunk_size)
    chunks = (_set_gts_to_missing(chunk, gt_rate_to_missing) for chunk in chunks)
    out_vars.put_chunks(chunks)
    return out_vars


def sample_variations(in_vars, out_vars, sample_rate, chunk_size=None):
    chunks = in_vars.iterate_chunks(chunk_size=chunk_size,
                                    random_sample_rate=sample_rate)
    out_vars.put_chunks(chunks)
