# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import math
from functools import partial
import tempfile
import os

import numpy

from variation.variations.vars_matrices import VariationsArrays
from variation import GT_FIELD, DP_FIELD
from variation.variations.pop_stats import (calc_number_of_alleles,
                                            calc_number_of_private_alleles,
                                            calc_pop_stats,
                                            calc_major_allele_freq,
                                            calc_obs_het, calc_exp_het,
                                            _get_original_function_name,
                                            create_pop_stats_report,
                                            STAT_FUNCTION_METADATA)


class PopStatsTest(unittest.TestCase):

    def _check_function(self, stat_funct, varis, pops, expected):
        results1 = stat_funct(varis, populations=pops)

        pop_stats = calc_pop_stats(varis, populations=pops, chunk_size=2,
                                   pop_stat_functions=[stat_funct])
        funct_name = _get_original_function_name(stat_funct)
        stat_name = STAT_FUNCTION_METADATA[funct_name]['stat_name']
        results2 = pop_stats[stat_name]

        for pop_id in pops:
            assert numpy.allclose(results1[pop_id], expected[pop_id],
                                  equal_nan=True)
            assert numpy.allclose(results2[pop_id], expected[pop_id],
                                  equal_nan=True)
            # assert numpy.all(results1[pop_id] == expected[pop_id])
            # assert numpy.all(results2[pop_id] == expected[pop_id])

    def test_num_alleles(self):
        stat_funct = partial(calc_number_of_alleles, min_num_genotypes=0)

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

        # min num genotypes
        stat_funct = partial(calc_number_of_alleles, min_num_genotypes=3)
        gts = numpy.array([[[1], [-1], [0], [0], [-1]],
                           [[-1], [-1], [1], [1], [-1]],
                           [[-1], [-1], [0], [1], [-1]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2, 3, 4, 5]}
        expected = {1: [2, 0, 0, 0]}
        self._check_function(stat_funct, varis, pops, expected)

    def test_num_private_alleles(self):
        stat_funct = partial(calc_number_of_private_alleles,
                             min_num_genotypes=0)

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

        # min_num_genotypes
        stat_funct = partial(calc_number_of_private_alleles,
                             min_num_genotypes=2)
        gts = numpy.array([[[0], [0], [0], [0], [1]],
                           [[0], [0], [1], [1], [1]],
                           [[0], [2], [0], [1], [1]],
                           [[1], [-1], [0], [0], [2]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0, 1, 1, 0], 2: [1, 1, 1, 0]}
        self._check_function(stat_funct, varis, pops, expected)

    def test_calc_major_allele_freq(self):
        stat_funct = calc_major_allele_freq

        gts = numpy.array([[[0], [0], [0], [0], [-1]],
                           [[0], [0], [1], [1], [-1]],
                           [[0], [2], [0], [1], [2]],
                           [[-1], [-1], [-1], [-1], [-1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [1., 1., 0.5, math.nan], 2: [1., 1., 1 / 3, math.nan]}
        stat_funct = partial(stat_funct, min_num_genotypes=1)
        self._check_function(stat_funct, varis, pops, expected)

    def test_calc_obs_het(self):
        stat_funct = calc_obs_het

        gts = numpy.array([[[0, 0], [0, 1], [0, 0], [0, 0], [0, -1]],
                           [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        dps = numpy.array([[20, 15, 20, 20, 20],
                           [20, 20, 20, 20, 20],
                           [20, 20, 20, 20, 20]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[DP_FIELD] = dps
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0.5, 0, math.nan], 2: [0, 1., math.nan]}
        partial_stat_funct = partial(stat_funct, min_num_genotypes=1,
                                     min_call_dp=0)
        self._check_function(partial_stat_funct, varis, pops, expected)

        # now setting a depth_threshold
        expected = {1: [0, 0, math.nan], 2: [0, 1., math.nan]}
        partial_stat_funct = partial(stat_funct, min_call_dp=20,
                                     min_num_genotypes=1)
        self._check_function(partial_stat_funct, varis, pops, expected)

    def test_calc_exp_het(self):
        stat_funct = calc_exp_het

        gts = numpy.array([[[0, 0], [0, 1], [0, 0], [0, 0], [0, -1]],
                           [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}
        expected = {1: [0.5, 0, math.nan], 2: [0, 0.6, math.nan]}
        partial_stat_funct = partial(stat_funct, min_num_genotypes=1)
        self._check_function(partial_stat_funct, varis, pops, expected)

    def test_report(self):
        gts = numpy.array([[[0, 0], [0, 1], [0, 0], [0, 0], [0, -1]],
                           [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis.samples = [1, 2, 3, 4, 5]
        pops = {1: [1, 2], 2: [3, 4, 5]}

        out_dir = tempfile.TemporaryDirectory()
        create_pop_stats_report(varis, pops, out_dir.name,
                                min_num_genotypes=1,
                                min_call_dp_for_obs_het=0,
                                violin_ylimits={'observed_heterozigosity': {'bottom':0, 'top':0.5}})
        stats_csv_fpath = os.path.join(out_dir.name,
                                       'pop_stats.csv')
        assert os.path.exists(stats_csv_fpath)
        stats_csv_fpath = os.path.join(out_dir.name,
                                       'pop_stats_violin_plots.svg')
        # input(out_dir.name)
        out_dir.cleanup()

# maf small/tot high/tot


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'PopStatsTest.test_num_alleles']
    unittest.main()
