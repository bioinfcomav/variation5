import os
import unittest
from os.path import join
from tempfile import NamedTemporaryFile

import numpy

from variation.gt_parsers.csv import (CSVParser,
                                      def_gt_allele_splitter,
                                      create_iupac_allele_splitter)
from variation.variations.vars_matrices import VariationsArrays, VariationsH5

from test.test_utils import TEST_DATA_DIR


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
                           sep=b'\t')
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

        parser = CSVParser(fhand, var_info, first_sample_column=3,
                           first_gt_column=3, sep=b'\t',
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

        # pandas csv
        fhand = open(join(TEST_DATA_DIR, 'csv', 'pandas.csv'), 'rb')
        var_info = {b'solcap_snp_sl_15058': {'chrom': b'chrom1', 'pos': 345},
                    b'solcap_snp_sl_60635': {'chrom': b'chrom1', 'pos': 346},
                    b'solcap_snp_sl_60604': {'chrom': b'chrom1', 'pos': 347}}

        parser = CSVParser(fhand, var_info, first_sample_column=0, sep=b'\t')
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
        for var, expect in zip(parser.variations, expected):
            assert var == expect

        fhand.close()
        assert parser.samples == [b'SR-9', b'SR-12', b'SR-13', b'SR-15',
                                  b'SR-18']
        assert parser.max_field_lens['alt'] == 2
        assert parser.ploidy == 2

    def test_put_vars_from_csv(self):
        fhand_ex = open(join(TEST_DATA_DIR, 'csv', 'iupac_ex3.txt'), 'rb')
        var_info = {b'1': {'chrom': b'SL2.40ch02', 'pos': 331954},
                    b'2': {'chrom': b'SL2.40ch02', 'pos': 681961},
                    b'3': {'chrom': b'SL2.40ch02', 'pos': 1511764}}
        parser = CSVParser(fhand_ex, var_info, first_sample_column=3,
                           first_gt_column=3, sep=b'\t',
                           gt_splitter=create_iupac_allele_splitter(),
                           max_field_lens={'alt': 1},
                           max_field_str_lens={'alt': 1, 'chrom': 20,
                                               'ref': 1})
        print('aa', parser.max_field_lens)
        print('str', parser.max_field_str_lens)
        with NamedTemporaryFile(suffix='.h5') as fhand:
            os.remove(fhand.name)
            h5 = VariationsH5(fhand.name, mode='w', ignore_overflows=True,
                              ignore_undefined_fields=True)
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
        parser = CSVParser(fhand_ex, var_info, first_sample_column=3,
                           first_gt_column=3, sep=b'\t',
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

if __name__ == "__main__":
    unittest.main()
