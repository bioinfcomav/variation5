from os.path import join
import unittest

import numpy

from variation.variations import VariationsH5
from variation.variations.naive_coalescence import get_mean_alleles_per_window
from test.test_utils import TEST_DATA_DIR


class NaiveCoalescenceTest(unittest.TestCase):

    def test_mean_alleles_per_window(self):
        intermediate_list = []
        means_per_locus_window = numpy.array([])
        h5 = VariationsH5(join(TEST_DATA_DIR, '20SNPs.h5'), mode='r')
        # you must create a known size array to use a generator

        for mean in get_mean_alleles_per_window(h5, win_size=100):
            intermediate_list.append(mean)
        means_per_locus_window = numpy.array(intermediate_list)
        means_per_locus_window = means_per_locus_window[numpy.logical_not(numpy.isnan(means_per_locus_window))]
        numpy.testing.assert_array_equal(means_per_locus_window,
                                         [1.75, 2.5, 2.25, 2., 2., 3., 2., 2.,
                                          1., 1.])


if __name__ == "__main__":
    unittest.main()
