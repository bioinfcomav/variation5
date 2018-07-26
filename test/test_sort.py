
import unittest
from os.path import join

import numpy

from variation.variations.vars_matrices import VariationsArrays
from variation.gt_parsers.csv import CSVParser
from variation.variations.sort import sort_variations
from test.test_utils import TEST_DATA_DIR


class SortVariationsTest(unittest.TestCase):
    def test_sort_variations(self):
        fhand = open(join(TEST_DATA_DIR, 'csv', 'standard_ex.tsv'), 'rb')
        var_info = {b'solcap_snp_sl_15058': {'chrom': b'chrom2', 'pos': 345},
                    b'solcap_snp_sl_60635': {'chrom': b'chrom1', 'pos': 346},
                    b'solcap_snp_sl_60604': {'chrom': b'chrom1', 'pos': 325}}
        parser = CSVParser(fhand, var_info, first_sample_column=1, sep=b'\t')
        variations = VariationsArrays(ignore_undefined_fields=True)
        variations.put_vars(parser)
        sorted_vars = VariationsArrays()
        sort_variations(variations, sorted_vars)
        exp_chrom = [b'chrom1', b'chrom1', b'chrom2']
        exp_pos = [325, 346, 345]
        assert numpy.all(sorted_vars['/variations/chrom'] == exp_chrom)
        assert numpy.all(sorted_vars['/variations/pos'] == exp_pos)
        fhand.close()

if __name__ == "__main__":
    unittest.main()
