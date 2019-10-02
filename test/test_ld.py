# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

import numpy

from test.test_utils import TEST_DATA_DIR
from variation.variations.vars_matrices import VariationsH5
from variation.variations.ld import (calc_ld_along_genome, _bivmom, _get_r,
                                     _calc_rogers_huff_r, calc_rogers_huff_r,
                                     calc_ld_random_pairs_from_different_chroms)
from variation.variations.stats import calc_maf


class TestLD(unittest.TestCase):

    def test_ld_along_genome(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        ld = calc_ld_along_genome(hdf5, max_dist=100000000, chunk_size=3,
                                  min_num_gts=1, max_maf=1.1)
        assert list(ld)[0:1] == [(0.0, 2960.0, (b'20', 14370, b'20', 17330))]

    def test_ld_calculation(self):
        Y = [2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1]
        Z = [2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1]

        Y = [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1,
                 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0]

        Z = [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1,
                 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2,
                 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1]
        yz_r = _get_r(Y, Z)
        yy_r = _get_r(Y, Y)
        zz_r = _get_r(Z, Z)
        # print('reference', yz_r, yy_r, zz_r)

        gt = numpy.array([Y, Z, Z])
        r = _calc_rogers_huff_r(gt)
        assert numpy.allclose(r, [yz_r, yz_r, zz_r])

        gts1 = numpy.array([Y, Z, Z])
        gts2 = numpy.array([Z, Y])
        r = calc_rogers_huff_r(gts1, gts2)
        assert numpy.allclose(r, [[yz_r, yy_r], [zz_r, yz_r], [zz_r, yz_r]])

        Y = [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1,
                 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0]

        Z = [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1,
                 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2,
                 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1]
        gts1 = numpy.array([Y, Z, Z])
        gts2 = numpy.array([Z, Y])
        r = calc_rogers_huff_r(gts1, gts2, debug=False)
        assert numpy.allclose(r, [[yz_r, yy_r], [zz_r, yz_r], [zz_r, yz_r]])

        Y = [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1,
                 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0, -1]

        Z = [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1,
                 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2,
                 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 0]
        gts1 = numpy.array([Y, Z, Z])
        gts2 = numpy.array([Z, Y])
        r = calc_rogers_huff_r(gts1, gts2, debug=True, min_num_gts=50)
        expected = [[yz_r, yy_r], [zz_r, yz_r], [zz_r, yz_r]]

        assert numpy.allclose(r, expected, atol=1e-3)

        r = calc_rogers_huff_r(gts1, gts2, debug=False, min_num_gts=51)
        expected = [[numpy.nan, numpy.nan], [zz_r, numpy.nan], [zz_r, numpy.nan]]
        assert numpy.allclose(r, expected, atol=1e-3, equal_nan=True)

    def test_ld_random_pairs_from_different_chroms(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.h5'),
                            mode='r')
        variations = hdf5.get_chunk(slice(5000, 15000))
        mafs = calc_maf(variations, min_num_genotypes=10, chunk_size=None)
        mafs[numpy.isnan(mafs)] = 1
        variations = variations.get_chunk(mafs < 0.95)
        lds = calc_ld_random_pairs_from_different_chroms(variations, 100)
        lds = list(lds)
        assert len(lds) == 100


if __name__ == "__main__":
    unittest.main()
