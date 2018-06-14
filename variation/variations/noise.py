
import numpy

from variation import GT_FIELD, MISSING_INT


def set_gts_to_missing(variations, percent_of_non_missing_gts_to_missing):
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
