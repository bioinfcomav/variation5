# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest

import numpy

from variation.variations.vars_matrices import VariationsArrays
from variation import GT_FIELD
from variation.variations.pop_stats import (calc_number_of_alleles,
                                            calc_number_of_private_alleles,
                                            calc_pop_stats)


class PopStatsTest(unittest.TestCase):

    def _check_function(self, stat_funct, varis, pops, expected):
        results1 = stat_funct(varis, populations=pops)

        pop_stats = calc_pop_stats(varis, populations=pops, chunk_size=2,
                                   pop_stat_functions=[stat_funct])
        results2 = pop_stats[stat_funct.__name__]

        for pop_id in pops:
            assert numpy.all(results1[pop_id] == expected[pop_id])
            assert numpy.all(results2[pop_id] == expected[pop_id])

    def test_num_alleles(self):
        stat_funct = calc_number_of_alleles

        gts = numpy.array([[[0], [0], [0], [0], [-1]],
                           [[0], [0], [1], [1], [-1]],
                           [[0], [0], [0], [1], [-1]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [1, 1, 1, 0], 2: [1, 1, 2, 0]}

        self._check_function(stat_funct, varis, pops, expected)

        # a population empty
        gts = numpy.array([[[-1], [-1], [0], [0], [-1]],
                           [[-1], [-1], [1], [1], [-1]],
                           [[-1], [-1], [0], [1], [-1]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0, 0, 0, 0], 2: [1, 1, 2, 0]}
        self._check_function(stat_funct, varis, pops, expected)

        # only one pop
        gts = numpy.array([[[1], [-1], [0], [0], [-1]],
                           [[-1], [-1], [1], [1], [-1]],
                           [[-1], [-1], [0], [1], [-1]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2]}
        expected = {1: [1, 0, 0, 0]}
        self._check_function(stat_funct, varis, pops, expected)

    def test_num_private_alleles(self):
        stat_funct = calc_number_of_private_alleles

        gts = numpy.array([[[0], [0], [0], [0], [-1]],
                           [[0], [0], [1], [1], [-1]],
                           [[0], [2], [0], [1], [-1]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0, 1, 1, 0], 2: [0, 1, 1, 0]}

        self._check_function(stat_funct, varis, pops, expected)

        # No missing alleles
        gts = numpy.array([[[0], [0], [0], [0], [1]],
                           [[0], [0], [1], [1], [1]],
                           [[0], [2], [0], [1], [1]],
                           [[1], [1], [0], [0], [2]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0, 1, 1, 1], 2: [1, 1, 1, 2]}
        self._check_function(stat_funct, varis, pops, expected)

        # all missing
        gts = numpy.array([[[0], [0], [0], [-1], [-1]],
                           [[0], [0], [1], [-1], [-1]],
                           [[0], [2], [-1], [-1], [-1]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0, 1, 2, 0], 2: [0, 1, 0, 0]}
        self._check_function(stat_funct, varis, pops, expected)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'SampleStatsTest.test_stasts_per_sample']
    unittest.main()
