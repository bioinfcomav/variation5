# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join
from test.test_utils import TEST_DATA_DIR, BIN_DIR
from variation.genotypes_matrix import (GenotypesMatrixParser,
                                        IUPAC_CODING, STANDARD_GT,
                                        change_gts_chain, decode_gts,
                                        collapse_gts, encode_gts,
                                        merge_sorted_variations, merge_snps,
    merge_alleles, merge_variations)
import os
from variations.vars_matrices import VariationsH5, put_vars_from_csv
import numpy
from variation.genotypes_matrix import count_compatible_snps_in_chains
from variation.vcf import VCFParser
from variation.variations.vars_matrices import _put_vars_from_vcf
from variation.variations.stats import _remove_nans
import sys
from subprocess import check_output


class GtMatrixTest(unittest.TestCase):
    def test_gt_matrix_parser(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.txt')
        parser = GenotypesMatrixParser(open(fpath), IUPAC_CODING,
                                       2, sep='\t',
                                       snp_fieldnames=['chrom', 'pos'])
        expected = [{'gts': [[1, 1], [0, 0], [-1, -1]], 'alt': ['G'],
                     'pos': 331954, 'ref': 'T', 'chrom': 'SL2.40ch02'},
                    {'gts': [[0, 0], [0, 0], [-1, -1]], 'alt': [''],
                     'pos': 681961, 'ref': 'C', 'chrom': 'SL2.40ch02'},
                    {'gts': [[0, 0], [0, 0], [1, 0]], 'alt': ['A'],
                     'pos': 1511764, 'ref': 'T', 'chrom': 'SL2.40ch02'}]
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
                     'alt': ['C', 'G'], 'chrom': '1', 'pos': 12432},
                    {'gts': [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1]],
                     'SNP_ID': 'solcap_snp_sl_60635', 'ref': 'G',
                     'alt': [''], 'chrom': '2', 'pos': 43534},
                    {'gts': [[1, 1], [-1, -1], [0, 1], [1, 1], [1, 1]],
                     'SNP_ID': 'solcap_snp_sl_60604', 'ref': 'T',
                     'alt': ['C'],  'chrom': 'sol.23', 'pos': 2345}]
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
            os.remove(out_fpath)
        except FileNotFoundError:
            pass
        h5 = VariationsH5(out_fpath, mode='w')
        put_vars_from_csv(parser, h5, 100)
        exp = [b'SL2.40ch02',b'SL2.40ch02', b'SL2.40ch02']
        assert list(h5['/variations/chrom'][:]) == exp
        assert list(h5['/variations/ref'][:]) == [b'T', b'C', b'T']
        assert list(h5['/variations/pos'][:]) == [331954, 681961,
                                                  1511764]
        exp = numpy.array([[[1, 1], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [1, 0]]])
        assert numpy.all(h5['/calls/GT'][:] == exp)

        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex2.txt')
        parser = GenotypesMatrixParser(open(fpath), IUPAC_CODING,
                                       2, sep='\t',
                                       snp_fieldnames=['chrom', 'pos'])
        out_fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex2.h5')
        try:
            os.remove(out_fpath)
        except FileNotFoundError:
            pass
        h5 = VariationsH5(out_fpath, mode='w')
        put_vars_from_csv(parser, h5, 100)

    def test_count_compatible_snsp_in_chains(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        h5 = VariationsH5(fpath, "r")
        chains_matrix = numpy.array([[[b'G', b'T']],
                                     [[b'G', b'T']],
                                     [[b'G', b'T']]])
        result = count_compatible_snps_in_chains(h5, chains_matrix)
        assert result == 1

    def test_change_gts_chain(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        h5 = VariationsH5(fpath, "r")
        mask = numpy.array([True, False, True])
        alleles = change_gts_chain(h5, mask)
        assert numpy.all(alleles == numpy.array([[b'A', b'C', b''],
                                                 [b'C', b'', b''],
                                                 [b'A', b'T', b'']]))

    def test_decode_gts(self):
        gts = numpy.array([[[0, 0], [0, 1]], [[1, 1], [1, 0]]])
        alleles = numpy.array([['A', 'T'], ['C', 'T']])
        expected = numpy.array([[['A', 'A'], ['A', 'T']],
                                [['T', 'T'], ['T', 'C']]])
        assert numpy.all(decode_gts(alleles, gts) == expected)

    def test_encode_gts(self):
        gts = numpy.array([[['A', 'A'], ['A', 'T']],
                           [['T', 'T'], ['T', 'C']]])
        alleles = numpy.array([['A', 'T'], ['C', 'T']])
        expected = numpy.array([[[0, 0], [0, 1]], [[1, 1], [1, 0]]])
        assert numpy.all(encode_gts(gts, alleles) == expected)
        gts = numpy.array([[['AA', 'AA'], ['G', 'T']]])
        alleles = numpy.array([['AA', 'T', 'G']])
        expected = numpy.array([[[0, 0], [2, 1]]])
        assert numpy.all(encode_gts(gts, alleles) == expected)

    def test_collapse_gts(self):
        gts = numpy.array([[['A', 'A'], ['A', 'T']],
                           [['T', 'T'], ['T', 'C']]])
        base_allele = b'ATCAC'
        relative_positions = [1, 4]
        expected = numpy.array([[['AACAT', 'AACAT'],
                                 ['AACAT', 'ATCAC']]])
        result = collapse_gts(base_allele, gts, relative_positions)
        assert numpy.all(result == expected)

    def test_merge_h5(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        h5_1 = VariationsH5(fpath, "r")
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex2.h5')
        h5_2 = VariationsH5(fpath, "r")
        expected = [[681961, 681961], [1511764, 1511764],
                    [None, 15164], [None, 15184]]
        for snp, exp in zip(merge_sorted_variations(h5_1, h5_2), expected):
            for x, y in zip(snp, exp):
                try:
                    assert x['/variations/pos'][0] == y
                except TypeError:
                    assert x is None

        expected = [[681961, 681961], [1511764, 1511764],
                    [15164, None], [15184, None]]
        for snp, exp in zip(merge_sorted_variations(h5_2, h5_1), expected):
            for x, y in zip(snp, exp):
                try:
                    assert x['/variations/pos'][0] == y
                except TypeError:
                    assert x is None

    def test_merge_alleles(self):
        alleles1 = numpy.array(['AT', 'TT'])
        alleles2 = numpy.array(['A', 'TT'])
        exp = numpy.array(['AT', 'TT', 'A'])
        assert numpy.all(merge_alleles(alleles1, alleles2) == exp)

    def test_merge_snps(self):
        class Variation(dict):
            def set_samples(self, samples):
                self.samples = samples

        merged = {'/variations/chrom': numpy.array([['']], dtype='S10'),
                  '/variations/pos': numpy.array([[-1]]),
                  '/variations/ref': numpy.array([['']], dtype='S10'),
                  '/variations/alt': numpy.array([['', '', '']], dtype='S10'),
                  '/variations/qual': numpy.array([[-1, ]]),
                  '/variations/info/AF': numpy.array([[-1, ]]),
                  '/variations/info/DP': numpy.array([[-1,-1]]),
                  '/calls/GT': numpy.array([[[-1, -1], [-1, -1], [-1, -1],
                                             [-1, -1], [-1, -1]]]),
                  '/calls/DP': numpy.array([[-1, -1, -1, -1, -1]])}
        snp1 = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                '/variations/pos': numpy.array([[10]]),
                '/variations/ref': numpy.array([['AT']], dtype='S10'),
                '/variations/alt': numpy.array([['TT', 'G']], dtype='S10'),
                '/variations/qual': numpy.array([[120]]),
                '/variations/info/AF': numpy.array([[15]]),
                '/variations/info/DP': numpy.array([[200]]),
                '/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 2]]]),
                '/calls/DP': numpy.array([[20, 30, 10]])}
        snp1 = Variation(snp1)
        snp1.set_samples(['s1', 's2', 's3'])
        snp2 = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                '/variations/pos': numpy.array([[11]]),
                '/variations/ref': numpy.array([['A']], dtype='S10'),
                '/variations/alt': numpy.array([['T']], dtype='S10'),
                '/variations/info/AF': numpy.array([[17]]),
                '/variations/info/DP': numpy.array([[210]]),
                '/calls/GT': numpy.array([[[1, 1], [0, 1]]])}
        snp2 = Variation(snp2)
        snp2.set_samples(['s4', 's5'])
        merge_snps(snp1, snp2, 0, merged,
                   fields_funct={'/variations/info/AF': min})
        expected = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                    '/variations/pos': numpy.array([[10]]),
                    '/variations/ref': numpy.array([['AT']], dtype='S10'),
                    '/variations/alt': numpy.array([['TT', 'G', 'AA']], dtype='S10'),
                    '/variations/qual': numpy.array([[120]]),
                    '/variations/info/AF': numpy.array([[15]]),
                    '/variations/info/DP': numpy.array([[200, 210]]),
                    '/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 2],
                                               [0, 0], [3, 0]]]),
                    '/calls/DP': numpy.array([[20, 30, 10, -1, -1]])}
        for key in expected.keys():
            assert numpy.all(merged[key] == expected[key])

        merged = {'/variations/chrom': numpy.array([['']], dtype='S10'),
                  '/variations/pos': numpy.array([[-1]]),
                  '/variations/ref': numpy.array([['']], dtype='S10'),
                  '/variations/alt': numpy.array([['', '']], dtype='S10'),
                  '/variations/qual': numpy.array([[-1, ]]),
                  '/variations/info/AF': numpy.array([[-1, ]]),
                  '/variations/info/DP': numpy.array([[-1,-1]]),
                  '/calls/GT': numpy.array([[[-1, -1], [-1, -1], [-1, -1],
                                             [-1, -1], [-1, -1]]]),
                  '/calls/DP': numpy.array([[-1, -1, -1, -1, -1]])}
        merge_snps(snp1, None, 0, merged,
                   fields_funct={'/variations/info/AF': min})
        expected = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                    '/variations/pos': numpy.array([[10]]),
                    '/variations/ref': numpy.array([['AT']], dtype='S10'),
                    '/variations/alt': numpy.array([['TT', 'G']], dtype='S10'),
                    '/variations/qual': numpy.array([[120]]),
                    '/variations/info/AF': numpy.array([[15]]),
                    '/variations/info/DP': numpy.array([[200, -1]]),
                    '/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 2],
                                               [-1, -1], [-1, -1]]]),
                    '/calls/DP': numpy.array([[20, 30, 10, -1, -1]])}
        for key in expected.keys():
            assert numpy.all(merged[key] == expected[key])

        merged = {'/variations/chrom': numpy.array([['']], dtype='S10'),
                  '/variations/pos': numpy.array([[-1]]),
                  '/variations/ref': numpy.array([['']], dtype='S10'),
                  '/variations/alt': numpy.array([['', '']], dtype='S10'),
                  '/variations/qual': numpy.array([[-1, ]]),
                  '/variations/info/AF': numpy.array([[-1, ]]),
                  '/variations/info/DP': numpy.array([[-1, -1]]),
                  '/calls/GT': numpy.array([[[-1, -1], [-1, -1], [-1, -1],
                                             [-1, -1], [-1, -1]]]),
                  '/calls/DP': numpy.array([[-1, -1, -1, -1, -1]])}
        merge_snps(None, snp2, 0, merged,
                   fields_funct={'/variations/info/AF': min})
        expected = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                    '/variations/pos': numpy.array([[11]]),
                    '/variations/ref': numpy.array([['A']], dtype='S10'),
                    '/variations/alt': numpy.array([['T', '']], dtype='S10'),
                    '/variations/info/AF': numpy.array([[17]]),
                    '/variations/info/DP': numpy.array([[-1, 210]]),
                    '/calls/GT': numpy.array([[[-1, -1], [-1, -1], [-1, -1],
                                               [1, 1], [0, 1]]]),
                    '/calls/DP': numpy.array([[-1, -1, -1, -1, -1]])}
        for key in expected.keys():
            assert numpy.all(merged[key] == expected[key])

    def test_merge_variations(self):
        merged_fpath = join(TEST_DATA_DIR, 'csv', 'merged.h5')
        format_array_h5 = VariationsH5(join(TEST_DATA_DIR, 'csv',
                                            'format.h5'), "r")
        format_h5 = VariationsH5(join(TEST_DATA_DIR, 'format_def.h5'), "r")
        try:
            os.remove(merged_fpath)
        except FileNotFoundError:
            pass
        try:
            merge_variations(format_h5, format_array_h5, merged_fpath)
            self.fail()
        except ValueError:
            pass
        os.remove(merged_fpath)
        merged_variations = merge_variations(format_h5, format_array_h5,
                                             merged_fpath, ignore_overlaps=True,
                                             ignore_2_or_more_overlaps=True)
        expected_h5 = VariationsH5(join(TEST_DATA_DIR, 'csv',
                                        'expected_merged.h5'), 'r')
        for key in merged_variations.keys():
            if 'float' in str(merged_variations[key][:].dtype):
                assert numpy.all(_remove_nans(expected_h5[key][:]) ==
                                 _remove_nans(merged_variations[key][:]))
            else:
                assert numpy.all(expected_h5[key][:] == merged_variations[key]
                                 [:])

        os.remove(merged_fpath)
        # Change the order
        merged_variations = merge_variations(format_array_h5, format_h5,
                                             merged_fpath, ignore_overlaps=True,
                                             ignore_2_or_more_overlaps=True)
        expected_h5 = VariationsH5(join(TEST_DATA_DIR, 'csv',
                                        'expected_merged2.h5'), 'r')
        for key in merged_variations.keys():
            if 'float' in str(merged_variations[key][:].dtype):
                assert numpy.all(_remove_nans(expected_h5[key][:]) ==
                                 _remove_nans(merged_variations[key][:]))
            else:
                assert numpy.all(expected_h5[key][:] == merged_variations[key]
                                 [:])

    def test_merge_hdf5_bin(self):
        merged_fpath = join(TEST_DATA_DIR, 'csv', 'merged2.h5')
        h5_1 = join(TEST_DATA_DIR, 'csv', 'format.h5')
        h5_2 = join(TEST_DATA_DIR, 'format_def.h5')
        try:
            os.remove(merged_fpath)
        except FileNotFoundError:
            pass
        bin_ = join(BIN_DIR, 'merge_hdf5.py')
        cmd = [sys.executable, bin_, h5_1, h5_2, '-o', merged_fpath,
               '-i', '-di']
        check_output(cmd)

if __name__ == "__main__":
#     import sys;sys.argv = ['', 'GtMatrixTest.test_merge_hdf5_bin']
    unittest.main()
