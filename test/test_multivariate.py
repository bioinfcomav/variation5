
import unittest
from os.path import join

import numpy

from test.test_utils import TEST_DATA_DIR
from variation.variations.multivariate import (non_param_multi_dim_scaling,
                                               do_pcoa, do_pca)
from variation.variations.vars_matrices import VariationsArrays, VariationsH5
from variation import GT_FIELD

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

    def test_pca(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        do_pca(hdf5)

        varis = VariationsArrays()
        gts = [[[0, 0], [0, 0], [1, 1]],
               [[0, 0], [0, 0], [1, 1]],
               [[0, 0], [0, 0], [1, 1]],
               [[0, 0], [0, 0], [1, 1]],
               ]
        gts = numpy.array(gts)
        varis[GT_FIELD] = gts
        varis.samples = ['a', 'b', 'c']
        res = do_pca(varis)
        projs = res['projections']
        assert projs.shape[0] == gts.shape[1]
        assert numpy.allclose(projs[0], projs[1])
        assert not numpy.allclose(projs[0], projs[2])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PlotTest.test_plot_boxplot_series']
    unittest.main()
