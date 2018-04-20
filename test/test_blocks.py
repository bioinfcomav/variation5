
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import io
import math

import numpy

from variation.variations.vars_matrices import VariationsArrays
from variation.gt_parsers.vcf import VCFParser
from variation.variations.blocks import (BlocksVariationGrouper,
                                         ALIGNED_ALLELES_FIELD_NAME)
from variation import (GT_FIELD, REF_FIELD, ALT_FIELD, CHROM_FIELD, POS_FIELD)

VCF1 = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
20	10	.	G	A	29	.	.	GT	0|0	0|0	1/1
20	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
20	30	.	A	G,T	67	.	.	GT	1|2	2|1	2/2
20	40	.	T	A	47	.	.	GT	0|0	1|1	0/0
20	50	.	GT	G,.	50	.	.	GT	0/1	0/2	1/1
'''


class GroupVarsPerBlockTest(unittest.TestCase):

    def _get_var_array(self, vcf_fhand):
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        variations = VariationsArrays(ignore_undefined_fields=True)
        variations.put_vars(vcf)
        vcf_fhand.close()
        return variations

    def test_group_vars_per_block_test(self):
        vcf_fhand = io.BytesIO(VCF1)
        variations = self._get_var_array(vcf_fhand)

        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]

        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             pre_read_max_size=math.inf)

        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert grouped_vars[REF_FIELD] == [b'GT']
        assert numpy.all(grouped_vars[ALT_FIELD] == [[b'GC', b'AT']])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [2, 2]]])
        assert grouped_vars[CHROM_FIELD] == [b'20']
        assert grouped_vars[POS_FIELD] == [10]
        assert grouped_vars['/variations/info/SN'] == [2]
        assert list(grouped_vars['/variations/info/AA'][0]) == [b'GT', b'GC',
                                                                b'AT']

        # het
        blocks = [{'chrom': b'20', 'start': 30, 'stop': 41}]

        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             pre_read_max_size=math.inf)
        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert grouped_vars[REF_FIELD] == [b'T']
        assert numpy.all(grouped_vars[ALT_FIELD] == [[b'A']])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [0, 0]]])
        assert grouped_vars[CHROM_FIELD] == [b'20']
        assert grouped_vars[POS_FIELD] == [40]
        assert grouped_vars['/variations/info/SN'] == [1]
        assert list(grouped_vars['/variations/info/AA'][0]) == [b'T', b'A', ]

        # missing
        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]
        vcf = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
20	10	.	G	A	29	.	.	GT	0|0	0|0	.
20	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
'''
        variations = self._get_var_array(io.BytesIO(vcf))
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             pre_read_max_size=math.inf)
        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert grouped_vars[REF_FIELD] == [b'T']
        assert numpy.all(grouped_vars[ALT_FIELD] == [[b'C']])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [0, 0]]])
        assert grouped_vars[CHROM_FIELD] == [b'20']
        assert grouped_vars[POS_FIELD] == [20]
        assert grouped_vars['/variations/info/SN'] == [1]
        assert list(grouped_vars['/variations/info/AA'][0]) == [b'T', b'C', ]

        # insertion
        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]
        vcf = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
20	10	.	G	AT	29	.	.	GT	0|0	0|0	1/1
20	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
'''
        variations = self._get_var_array(io.BytesIO(vcf))
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             pre_read_max_size=math.inf)
        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert grouped_vars[REF_FIELD] == [b'GT']
        assert numpy.all(grouped_vars[ALT_FIELD] == [[b'GC', b'ATT']])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [2, 2]]])
        assert grouped_vars[CHROM_FIELD] == [b'20']
        assert grouped_vars[POS_FIELD] == [10]
        assert grouped_vars['/variations/info/SN'] == [2]
        assert list(grouped_vars['/variations/info/AA'][0]) == [b'G-T', b'G-C',
                                                                b'ATT']

    def test_group_vars_per_block_deletion_test(self):
        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]
        vcf = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
20	10	.	GA	G	29	.	.	GT	0|0	0|0	1/1
20	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
'''
        variations = self._get_var_array(io.BytesIO(vcf))
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             pre_read_max_size=math.inf)
        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert grouped_vars[REF_FIELD] == [b'GAT']
        assert numpy.all(grouped_vars[ALT_FIELD] == [[b'GAC', b'GT']])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [2, 2]]])
        assert grouped_vars[CHROM_FIELD] == [b'20']
        assert grouped_vars[POS_FIELD] == [10]
        assert grouped_vars['/variations/info/SN'] == [2]
        assert list(grouped_vars['/variations/info/AA'][0]) == [b'GAT', b'GAC',
                                                                b'G-T']

    def test_slices(self):
        blocks = [{'chrom': b'1', 'start': 10, 'stop': 21},
                  {'chrom': b'1', 'start': 30, 'stop': 40},
                  {'chrom': b'2', 'start': 10, 'stop': 20},
                  {'chrom': b'2', 'start': 20, 'stop': 41},
                  ]
        vcf = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
1	10	.	G	A	2	.	.	GT	0|0	0|0	1/1
1	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
1	30	.	G	A	2	.	.	GT	1|1	0|0	1/1
1	40	.	T	C	3	.	.	GT	0|0	1|1	0/0
2	10	.	G	A	2	.	.	GT	0|0	1|1	1/1
2	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
2	30	.	G	A	2	.	.	GT	1|1	0|0	1/1
2	40	.	T	C	3	.	.	GT	1|1	1|1	0/0
'''
        variations = self._get_var_array(io.BytesIO(vcf))
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             pre_read_max_size=math.inf)
        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert list(grouped_vars[REF_FIELD]) == [b'GT', b'G', b'G', b'TGT']
        assert numpy.all(grouped_vars[ALT_FIELD] == [[b'GC', b'AT', b''],
                                                     [b'A', b'', b''],
                                                     [b'A', b'', b''],
                                                     [b'TAC', b'CGC', b'TAT'], ])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [2, 2]],
                                                    [[1, 1], [0, 0], [1, 1]],
                                                    [[0, 0], [1, 1], [1, 1]],
                                                    [[1, 1], [2, 2], [3, 3]]])
        assert list(grouped_vars[CHROM_FIELD]) == [b'1', b'1', b'2', b'2']
        assert list(grouped_vars[POS_FIELD]) == [10, 30, 10, 20]
        assert list(grouped_vars['/variations/info/SN']) == [2, 1, 1, 3]

    def test_field_lens(self):
        vcf_fhand = io.BytesIO(VCF1)
        variations = self._get_var_array(vcf_fhand)

        # 20    10    .    G    A    29    .    .    GT    0|0    0|0    1/1
        # 20    20    .    T    C    3    .    .    GT    0|0    1|1    0/0
        # 20    40    .    T    A    47    .    .    GT    0|0    1|1    0/0

        blocks = [{'chrom': b'20', 'start': 10, 'stop': 100}]
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True)

        list(var_grouper.variations)
        field_lens = {'alt': 2, 'INFO': {ALIGNED_ALLELES_FIELD_NAME: 3}}
        field_str_lens = {'alt': 3,
                          'chrom': 2,
                          'ref': 3,
                          'INFO': {b'AA': 3},
                          }
        assert var_grouper.max_field_lens == field_lens
        assert var_grouper.max_field_str_lens == field_str_lens


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'GroupVarsPerBlockTest.test_group_vars_per_block_deletion_test']
    unittest.main()
