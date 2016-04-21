
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

from test.test_utils import TEST_DATA_DIR
from variation.variations.vars_matrices import VariationsH5
from variation import DP_FIELD
from variation.variations.util import add_mock_depth


class VarUtilTest(unittest.TestCase):
    def test_add_depth(self):
        snps = VariationsH5(join(TEST_DATA_DIR, 'expected_merged3.h5'), 'r')
        snps2 = snps.get_chunk(slice(None, None))
        add_mock_depth(snps2, 30)

        assert snps2[DP_FIELD].shape == (snps2.num_variations,
                                         len(snps2.samples))
        assert snps2[DP_FIELD][0, 0] == 30


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'MergeTest.test_merge_complex_var']
    unittest.main()
