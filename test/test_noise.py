# TODO sample

# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

import numpy

from variation.variations.vars_matrices import VariationsArrays, VariationsH5
from test.test_utils import TEST_DATA_DIR
from variation.variations.random import (copy_setting_gts_to_missing,
                                         sample_variations)
from variation import GT_FIELD, MISSING_INT
from variation.gt_parsers.vcf import VCFParser


class SetToMissingTest(unittest.TestCase):

    def test_set_to_missing(self):
        orig_vars = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')

        vars = copy_setting_gts_to_missing(orig_vars, gt_rate_to_missing=0.9)

        orig_gts = orig_vars[GT_FIELD][...]
        noise_gts = vars[GT_FIELD]
        assert orig_gts.shape == noise_gts.shape
        mask_different_gts = orig_gts != noise_gts
        expected_num_gts_set_to_missing = int(round(numpy.sum(orig_gts != MISSING_INT) * 0.9))
        assert expected_num_gts_set_to_missing == mask_different_gts.sum()
        assert not numpy.sum(orig_gts[mask_different_gts] == MISSING_INT)

        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VariationsArrays(ignore_undefined_fields=True)
        snps.put_vars(vcf)
        vcf_fhand.close()
        numpy.random.seed(1)
        gts = numpy.array([[[0, 1], [1, 0], [-1, 1]],
                           [[0, 0], [0, 1], [-1, 0]],
                           [[-1, 2], [2, 1], [-1, 2]],
                           [[0, 0], [-1, 0], [1, 0]],
                           [[0, 1], [-1, 2], [1, 1]]])

        expected_gts = numpy.array([[[0, 1], [1, 0], [-1, 1]],
                                    [[-1, -1], [0, 1], [-1, 0]],
                                    [[-1, 2], [2, 1], [-1, 2]],
                                    [[0, 0], [-1, 0], [-1, -1]],
                                    [[-1, -1], [-1, 2], [-1, -1]]])
        del snps[GT_FIELD]
        snps[GT_FIELD] = gts
        vars = copy_setting_gts_to_missing(snps,
                                           gt_rate_to_missing=0.5)

        noise_gts = vars[GT_FIELD]
        assert numpy.all(noise_gts == expected_gts)

        # TDO Not enough not missing


class VariationSamplingTest(unittest.TestCase):

    def test_sampling(self):
        vars_in = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        numpy.random.seed(1)
        vars_out = sample_variations(vars_in, sample_rate=0.1)
        assert vars_out.num_variations == 94


if __name__ == "__main__":
    # import sys; sys.argv = ['', 'VarMatsTests.test_iterate_chroms']
    unittest.main()
