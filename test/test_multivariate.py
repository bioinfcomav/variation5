
import unittest
from os.path import join

import numpy

from test.test_utils import TEST_DATA_DIR
from variation.variations.multivariate import (non_param_multi_dim_scaling,
                                               do_pcoa, do_pca)
from variation.variations import VariationsH5

# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111


class MultivariateTest(unittest.TestCase):
    def test_mds(self):
        dists = [10., 22.3, 14.1]
        res = non_param_multi_dim_scaling(dists)
        assert res['stress'] < 0.1
        assert res['projections'].shape == (3, 3)

        # print(do_pcoa(dists))

    def test_pca(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        do_pca(hdf5)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PlotTest.test_plot_boxplot_series']
    unittest.main()
