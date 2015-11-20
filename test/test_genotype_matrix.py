# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import os
import unittest
import sys
from os.path import join
from subprocess import check_output, CalledProcessError
from tempfile import NamedTemporaryFile

import numpy

from variation.genotypes_matrix import (CSVParser,
                                        def_gt_allele_splitter,
                                        create_iupac_allele_splitter,
                                        change_gts_chain,
                                        collapse_alleles,
                                        merge_sorted_variations, merge_snps,
                                        merge_alleles, merge_variations,
                                        transform_gts_to_merge)
from variation.variations.vars_matrices import VariationsH5, VariationsArrays
from variation.genotypes_matrix import count_compatible_snps_in_chains
from variation.variations.stats import _remove_nans
from test.test_utils import TEST_DATA_DIR, BIN_DIR


class CSVParserTest(unittest.TestCase):
    def test_def_gt_allele_splitter(self):
        assert def_gt_allele_splitter(b'-') == None
        assert def_gt_allele_splitter(b'--') == None
        assert def_gt_allele_splitter(b'') == None
        assert def_gt_allele_splitter(b'A-') == (ord('A'), None)
        assert def_gt_allele_splitter(b'AA') == (ord('A'), ord('A'))

    def test_create_iupac_splitter(self):
        spliter = create_iupac_allele_splitter()
        assert spliter(b'A') == (ord('A'), ord('A'))
        assert spliter(b'-') == None
        assert spliter(b'') == None

    def _var_gt_to_letter(self, variation):
        gts = variation['gts']
        allele_coding = {0: variation['ref']}
        del variation['ref']
        if 'alt' in variation:
            allele_coding.update({idx+1: allele for idx, allele in
                                                enumerate(variation['alt'])})
            del variation['alt']
        alleles = set(allele_coding.values())
        variation['alleles'] = alleles

        allele_coding[None] = None
        new_gts = []
        for gt in gts:
            if gt is None:
                new_gts.append(None)
                continue
            new_gts.append(tuple([allele_coding[allele] for allele in gt]))
        variation['gts'] = new_gts

    def test_csv_parser(self):
        # standad file
        fhand = open(join(TEST_DATA_DIR, 'csv', 'standard_ex.tsv'), 'rb')
        var_info = {b'solcap_snp_sl_15058': {'chrom': b'chrom1', 'pos': 345},
                    b'solcap_snp_sl_60635': {'chrom': b'chrom1', 'pos': 346},
                    b'solcap_snp_sl_60604': {'chrom': b'chrom1', 'pos': 347}}
        parser = CSVParser(fhand, var_info, first_sample_column=1,
                           sep='\t')
        expected = [(b'chrom1', 345, b'solcap_snp_sl_15058', b'A',
                     [b'C', b'G'], None, None, None, [(b'GT', [(0, 0), (0, 0),
                                                               (0, 0), (0, 1),
                                                               (2, 2)])]),
                    (b'chrom1', 346, b'solcap_snp_sl_60635', b'G', None, None,
                     None, None, [(b'GT', [(0, 0), (0, 0), (0, 0), (0, 0),
                                           None])]),
                    (b'chrom1', 347, b'solcap_snp_sl_60604', b'C', [b'T'],
                     None, None, None, [(b'GT', [(0, 0), None, (1, 0), (0, 0),
                                                 (0, 0)])])]

        assert list(parser.variations) == expected
        fhand.close()
        assert parser.samples == [b'SR-9', b'SR-12', b'SR-13', b'SR-15',
                                  b'SR-18']
        assert parser.max_field_lens['alt'] == 2
        assert parser.ploidy == 2

        # IUPAC file
        fhand = open(join(TEST_DATA_DIR, 'csv', 'iupac_ex3.txt'), 'rb')
        var_info = {b'1': {'chrom': b'SL2.40ch02', 'pos': 331954},
                    b'2': {'chrom': b'SL2.40ch02', 'pos': 681961},
                    b'3': {'chrom': b'SL2.40ch02', 'pos': 1511764}}

        parser = CSVParser(fhand, var_info, first_sample_column=3, sep='\t',
                           gt_splitter=create_iupac_allele_splitter())

        expected = [(b'SL2.40ch02', 331954, b'1', b'T', [b'G'], None, None,
                     None, [(b'GT', [(1, 1), (0, 0), None])]),
                    (b'SL2.40ch02', 681961, b'2', b'C', None, None, None,
                     None, [(b'GT', [(0, 0), (0, 0), None])]),
                    (b'SL2.40ch02', 1511764, b'3', b'A', [b'T'], None, None,
                     None, [(b'GT', [(1, 1), (1, 1), (0, 1)])]),
                    (b'SL2.40ch02', 331954, b'1', b'T', [b'G'], None, None,
                     None, [(b'GT', [(1, 1), (0, 0), None])])]
        for var, expect in zip(parser.variations, expected):
            assert var == expect

        fhand.close()
        assert parser.samples == [b'TS-1', b'TS-11', b'TS-21']
        assert parser.max_field_lens['alt'] == 1
        assert parser.ploidy == 2

    def test_put_vars_from_csv(self):
        fhand_ex = open(join(TEST_DATA_DIR, 'csv', 'iupac_ex3.txt'), 'rb')
        var_info = {b'1': {'chrom': b'SL2.40ch02', 'pos': 331954},
                    b'2': {'chrom': b'SL2.40ch02', 'pos': 681961},
                    b'3': {'chrom': b'SL2.40ch02', 'pos': 1511764}}
        parser = CSVParser(fhand_ex, var_info, first_sample_column=3, sep='\t',
                           gt_splitter=create_iupac_allele_splitter(),
                           max_field_lens={'alt': 1},
                           max_field_str_lens={'alt': 1, 'chrom':20})

        with NamedTemporaryFile(suffix='.h5') as fhand:
            os.remove(fhand.name)
            h5 = VariationsH5(fhand.name, mode='w')
            h5.put_vars(parser)
            exp = [b'SL2.40ch02', b'SL2.40ch02', b'SL2.40ch02']
            assert list(h5['/variations/chrom'][:]) == exp
            alleles = list(zip(h5['/variations/ref'],
                           [alts[0] for alts in h5['/variations/alt']]))
            exp = [(b'G', b'T'), (b'C', b''), (b'A', b'T')]
            for als, aexp in zip(alleles, exp):
                assert set(als) == set(aexp)
            assert list(h5['/variations/pos'][:]) == [331954, 681961,
                                                      1511764]
            exp1 = numpy.array([[[1, 1], [0, 0], [-1, -1]],
                                [[0, 0], [0, 0], [-1, -1]],
                                [[0, 0], [0, 0], [1, 0]]])
            exp2 = numpy.array([[[0, 0], [1, 1], [-1, -1]],
                                [[0, 0], [0, 0], [-1, -1]],
                                [[1, 1], [1, 1], [0, 1]]])
            for gts, exp_gts1, exp_gts2 in zip(h5['/calls/GT'][:], exp1, exp2):
                for gt, ex1, ex2 in zip(gts, exp_gts1, exp_gts2):
                    assert set(gt) == set(ex1) or set(gt) == set(ex2)

        if os.path.exists(fhand.name):
            os.remove(fhand.name)
        fhand_ex.close()

        fhand_ex = open(join(TEST_DATA_DIR, 'csv',
                             'two_letter_coding_ex3.txt'), 'rb')
        var_info = {b'1': {'chrom': b'SL2.40ch02', 'pos': 331954},
                    b'2': {'chrom': b'SL2.40ch02', 'pos': 681961},
                    b'3': {'chrom': b'SL2.40ch02', 'pos': 1511764}}
        parser = CSVParser(fhand_ex, var_info, first_sample_column=3, sep='\t',
                           max_field_lens={'alt': 1},
                           max_field_str_lens={'alt': 1, 'chrom':20})

        h5 = VariationsArrays()
        h5.put_vars(parser)
        exp = [b'SL2.40ch02', b'SL2.40ch02', b'SL2.40ch02']
        assert list(h5['/variations/chrom'][:]) == exp
        alleles = list(zip(h5['/variations/ref'],
                       [alts[0] for alts in h5['/variations/alt']]))
        exp = [(b'G', b'T'), (b'C', b''), (b'A', b'T')]
        for als, aexp in zip(alleles, exp):
            assert set(als) == set(aexp)
        assert list(h5['/variations/pos'][:]) == [331954, 681961,
                                                  1511764]
        exp1 = numpy.array([[[1, 1], [0, 0], [-1, -1]],
                            [[0, 0], [0, 0], [-1, -1]],
                            [[0, 0], [0, 0], [1, 0]]])
        exp2 = numpy.array([[[0, 0], [1, 1], [-1, -1]],
                            [[0, 0], [0, 0], [-1, -1]],
                            [[1, 1], [1, 1], [0, 1]]])
        for gts, exp_gts1, exp_gts2 in zip(h5['/calls/GT'][:], exp1, exp2):
            for gt, ex1, ex2 in zip(gts, exp_gts1, exp_gts2):
                assert set(gt) == set(ex1) or set(gt) == set(ex2)
        fhand_ex.close()

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

    def test_collapse_alleles(self):
        alleles = numpy.array([[b'A', b'T', b'C']])
        base_allele = b'ATCAC'
        relative_position = 1
        expected = numpy.array([[b'AACAC', b'ATCAC', b'ACCAC']])
        result = collapse_alleles(base_allele, alleles, relative_position)
        assert numpy.all(result == expected)

    def test_transform_gts_to_merge(self):
        alleles = numpy.array([b'AACAC', b'A'])
        collapsed_alleles = numpy.array([b'ACCAC', b'ATCAC', b'AACAC'])
        gts = numpy.array([[[0, 0], [1, 2]],
                           [[2, 2], [0, 1]]])
        expected = numpy.array([[[2, 2], [3, 0]],
                                [[0, 0], [2, 3]]])
        expected_alleles = numpy.array([b'AACAC', b'A', b'ACCAC', b'ATCAC'])
        result = transform_gts_to_merge(alleles, collapsed_alleles, gts)
        merged_alleles, gts = result
        assert numpy.all(gts == expected)
        assert numpy.all(merged_alleles == expected_alleles)

    def test_merge_h5(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        h5_1 = VariationsH5(fpath, "r")
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex2.h5')
        h5_2 = VariationsH5(fpath, "r")
        expected = [[681961, 681961], [1511764, 1511764],
                    [None, 15164], [None, 15184]]
        for snp, exp in zip(merge_sorted_variations(h5_1, h5_2, True), expected):
            for x, y in zip(snp, exp):
                try:
                    assert x['/variations/pos'][0] == y
                except TypeError:
                    assert x is None

        expected = [[681961, 681961], [1511764, 1511764],
                    [15164, None], [15184, None]]
        for snp, exp in zip(merge_sorted_variations(h5_2, h5_1, True), expected):
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
        class FakeVariation(dict):
            def set_samples(self, samples):
                self.samples = samples

        merged = {'/variations/chrom': numpy.array([['']], dtype='S10'),
                  '/variations/pos': numpy.array([[-1]]),
                  '/variations/ref': numpy.array([['']], dtype='S10'),
                  '/variations/alt': numpy.array([['', '', '']], dtype='S10'),
                  '/variations/qual': numpy.array([[-1, ]]),
                  '/variations/info/AF': numpy.array([[-1, ]]),
                  '/variations/info/DP': numpy.array([[-1, -1]]),
                  '/calls/GT': numpy.array([[[-1, -1], [-1, -1], [-1, -1],
                                             [-1, -1], [-1, -1]]]),
                  '/calls/DP': numpy.array([[-1, -1, -1, -1, -1]])}
        snp1 = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                '/variations/pos': numpy.array([10]),
                '/variations/ref': numpy.array([['AT']], dtype='S10'),
                '/variations/alt': numpy.array([['TT', 'G']], dtype='S10'),
                '/variations/qual': numpy.array([[120]]),
                '/variations/info/AF': numpy.array([[15]]),
                '/variations/info/DP': numpy.array([[200]]),
                '/calls/GT': numpy.array([[[0, 0], [0, 1], [0, 2]]]),
                '/calls/DP': numpy.array([[20, 30, 10]])}
        snp1 = FakeVariation(snp1)
        snp1.set_samples(['s1', 's2', 's3'])
        snp2 = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                '/variations/pos': numpy.array([11]),
                '/variations/ref': numpy.array([['A']], dtype='S10'),
                '/variations/alt': numpy.array([['T']], dtype='S10'),
                '/variations/info/AF': numpy.array([[17]]),
                '/variations/info/DP': numpy.array([[210]]),
                '/calls/GT': numpy.array([[[1, 1], [0, 1]]])}
        snp2 = FakeVariation(snp2)
        snp2.set_samples(['s4', 's5'])
        merge_snps(snp1, snp2, 0, merged,
                   fields_funct={'/variations/info/AF': min})
        expected = {'/variations/chrom': numpy.array([['chr1']], dtype='S10'),
                    '/variations/pos': numpy.array([[10]]),
                    '/variations/ref': numpy.array([['AT']], dtype='S10'),
                    '/variations/alt': numpy.array([['TT', 'G', 'AA']],
                                                   dtype='S10'),
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
                  '/variations/info/DP': numpy.array([[-1, -1]]),
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
        merged_fhand = NamedTemporaryFile()
        merged_fpath = merged_fhand.name
        merged_fhand.close()

        format_array_h5 = VariationsH5(join(TEST_DATA_DIR, 'csv',
                                            'format.h5'), "r")
        format_h5 = VariationsH5(join(TEST_DATA_DIR, 'format_def.h5'), "r")
        try:
            merge_variations(format_h5, format_array_h5, merged_fpath)
            self.fail()
        except ValueError:
            pass
        os.remove(merged_fpath)

        merged_variations, log = merge_variations(format_h5, format_array_h5,
                                                  merged_fpath,
                                                  ignore_overlaps=True,
                                                  ignore_2_or_more_overlaps=True)

        expected_h5 = VariationsH5(join(TEST_DATA_DIR, 'csv',
                                        'expected_merged.h5'), 'r')
        expected_log = {'added_new_snps': 5, 'total_merged_snps': 6,
                        'ignored_overlap_snps': 3, 'modified_merged_snps': 1,
                        'ignored_ref_snps': 0}
        assert log == expected_log
        # Dirty hack to remove tmp_path
        try:
            for key in merged_variations.keys():
                if 'float' in str(merged_variations[key][:].dtype):
                    assert numpy.all(_remove_nans(expected_h5[key][:]) ==
                                     _remove_nans(merged_variations[key][:]))
                else:
                    result = merged_variations[key][:]
                    assert numpy.all(expected_h5[key][:] == result)
            os.remove(merged_fpath)
        except Exception:
            os.remove(merged_fpath)

        # Change the order
        merged_variations = merge_variations(format_array_h5, format_h5,
                                             merged_fpath,
                                             ignore_overlaps=True,
                                             ignore_2_or_more_overlaps=True)
        expected_h5 = VariationsH5(join(TEST_DATA_DIR, 'csv',
                                        'expected_merged2.h5'), 'r')
        try:
            for key in merged_variations.keys():
                if 'float' in str(merged_variations[key][:].dtype):
                    assert numpy.all(_remove_nans(expected_h5[key][:]) ==
                                     _remove_nans(merged_variations[key][:]))
                else:
                    result = merged_variations[key][:]
                    assert numpy.all(expected_h5[key][:] == result)
            os.remove(merged_fpath)
        except Exception:
            os.remove(merged_fpath)

    def test_merge_hdf5_bin(self):
        merged_fhand = NamedTemporaryFile()
        merged_fpath = merged_fhand.name
        merged_fhand.close()
        h5_1 = join(TEST_DATA_DIR, 'csv', 'format.h5')
        h5_2 = join(TEST_DATA_DIR, 'format_def.h5')
        bin_ = join(BIN_DIR, 'merge_hdf5.py')
        cmd = [sys.executable, bin_, h5_1, h5_2, '-o', merged_fpath, '-i',
               '-di']
        try:
            check_output(cmd)
            os.remove(merged_fpath)
            os.remove(merged_fpath + '.log')
        except CalledProcessError:
            os.remove(merged_fpath)
            os.remove(merged_fpath + '.log')

if __name__ == "__main__":
    unittest.main()
