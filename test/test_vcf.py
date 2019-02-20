# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

from variation.variations.vars_matrices import VariationsArrays
from variation.gt_parsers.vcf import VCFParser
from test.test_utils import TEST_DATA_DIR


class VcfTest(unittest.TestCase):

    def test_vcf_detect_fields(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf_fhand2 = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand,
                        kept_fields=['/variations/qual'])
        vcf2 = VCFParser(vcf_fhand2,
                         ignored_fields=['/variations/qual'])
        snps = VariationsArrays(ignore_undefined_fields=True)
        snps.put_vars(vcf)
        metadata = snps.metadata
        snps2 = VariationsArrays(ignore_undefined_fields=True)
        snps2.put_vars(vcf2)
        metadata2 = snps2.metadata
        assert '/calls/HQ' in metadata.keys()
        assert '/variations/qual' not in metadata2.keys()
        vcf_fhand.close()
        vcf_fhand2.close()

    def test_parser_vcf_filters(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def_without_info.vcf'),
                         'rb')
        vcf = VCFParser(vcf_fhand)
        filters = []
        for var in vcf.variations:
            filters.append(var[6])
        assert filters == [[], [b'q10'], [], [], []]
        vcf_fhand.close()
        # No filters
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def_without_filter.vcf'),
                         'rb')
        vcf = VCFParser(vcf_fhand)
        filters = []
        for var in vcf.variations:
            filters.append(var[6])
        assert filters == [None, None, None, None, None]
        vcf_fhand.close()


if __name__ == "__main__":
    # import sys; sys.argv = ['', 'VcfTest.test_parser_vcf_filters']
    unittest.main()
