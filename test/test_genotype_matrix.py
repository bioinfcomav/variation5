# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join
from test.test_utils import TEST_DATA_DIR
from variation.genotypes_matrix import (GenotypesMatrixParser,
                                        IUPAC_CODING, STANDARD_GT)
import os
from variation.variations.vars_matrices import VariationsH5, put_vars_from_csv
import numpy


class GtMatrixTest(unittest.TestCase):
    def test_gt_matrix_parser(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.txt')
        parser = GenotypesMatrixParser(open(fpath), IUPAC_CODING,
                                       2, sep='\t',
                                       snp_fieldnames=['chrom', 'pos'])
        expected = [{'gts': [[1, 1], [0, 0], [-1, -1]], 'alt': ['G'],
                     'pos': '331954', 'ref': 'T', 'chrom': 'SL2.40ch02'},
                    {'gts': [[0, 0], [0, 0], [-1, -1]], 'alt': [''],
                     'pos': '681961', 'ref': 'C', 'chrom': 'SL2.40ch02'},
                    {'gts': [[0, 0], [0, 0], [1, 0]], 'alt': ['A'],
                     'pos': '1511764', 'ref': 'T', 'chrom': 'SL2.40ch02'}]
        for x, y in zip(parser, expected):
            for key in x.keys():
                assert x[key] == y[key]

        fpath = join(TEST_DATA_DIR, 'csv', 'standard_ex.tsv')
        parser = GenotypesMatrixParser(open(fpath), STANDARD_GT,
                                       2, sep='\t',
                                       snp_fieldnames=['SNP_ID'])
        expected = [{'gts': [[1, 1], [1, 1], [1, 1], [1, 2], [0, 0]],
                     'SNP_ID': 'solcap_snp_sl_15058', 'ref': 'G',
                     'alt': ['A', 'C']},
                    {'gts': [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                     'SNP_ID': 'solcap_snp_sl_60635', 'ref': 'G',
                     'alt': ['']},
                    {'gts': [[1, 1], [-1, -1], [0, 1], [1, 1], [1, 1]],
                     'SNP_ID': 'solcap_snp_sl_60604', 'ref': 'T',
                     'alt': ['C']}]
        for x, y in zip(parser, expected):
            for key in x.keys():
                assert x[key] == y[key]

        # With the SNP data in a different file
        fpath = join(TEST_DATA_DIR, 'csv', 'standard_ex.tsv')
        meta_fpath = join(TEST_DATA_DIR, 'csv', 'meta.tsv')
        parser = GenotypesMatrixParser(open(fpath), STANDARD_GT,
                                       2, sep='\t', id_fieldnames=['SNP_ID',
                                                                   'SNP_ID'],
                                       metadata_fhand=open(meta_fpath),
                                       snp_fieldnames=['chrom', 'pos',
                                                       'SNP_ID', 'ref'])
        expected = [{'gts': [[0, 0], [0, 0], [0, 0], [0, 1], [2, 2]],
                     'SNP_ID': 'solcap_snp_sl_15058', 'ref': 'A',
                     'alt': ['C', 'G'], 'chrom': '1', 'pos': '12432'},
                    {'gts': [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                     'SNP_ID': 'solcap_snp_sl_60635', 'ref': 'G',
                     'alt': [''], 'chrom': '2', 'pos': '43534'},
                    {'gts': [[1, 1], [-1, -1], [0, 1], [1, 1], [1, 1]],
                     'SNP_ID': 'solcap_snp_sl_60604', 'ref': 'T',
                     'alt': ['C'],  'chrom': 'sol.23', 'pos': '2345'}]
        for x, y in zip(parser, expected):
            for key in x.keys():
                assert x[key] == y[key]

    def test_put_vars_from_csv(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.txt')
        parser = GenotypesMatrixParser(open(fpath), IUPAC_CODING,
                                       2, sep='\t',
                                       snp_fieldnames=['chrom', 'pos'])
        out_fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        try:
            os.remove(out_fpath + '.hdf5')
        except FileNotFoundError:
            pass
        h5 = VariationsH5(out_fpath + '.hdf5', mode='w')
        put_vars_from_csv(parser, h5, 100)
        exp = [b'SL2.40ch02',b'SL2.40ch02', b'SL2.40ch02']
        assert list(h5['/variations/chrom'][:]) == exp
        assert list(h5['/variations/ref'][:]) == [b'T', b'C', b'T']
        assert list(h5['/variations/pos'][:]) == [b'331954', b'681961',
                                                  b'1511764']
        exp = numpy.array([[[1, 1], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [1, 0]]])
        assert numpy.all(h5['/calls/GT'][:] == exp)


if __name__ == "__main__":
#     import sys;sys.argv = ['', 'GtMatrixTest.test_put_vars_from_csv']
    unittest.main()
