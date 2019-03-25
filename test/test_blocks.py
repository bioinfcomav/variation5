
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
import io
import pickle
from functools import partial

import numpy

from variation.variations.vars_matrices import VariationsArrays
from variation.gt_parsers.vcf import VCFParser
from variation.variations.blocks import (BlocksVariationGrouper,
                                         generate_blocks,
                                         _calc_if_snps_are_highly_correlated012,
                                         _snp_has_enough_data, find_last,
                                         find_first, find_first_in_matrices)
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
        vcf = VCFParser(vcf_fhand)
        variations = VariationsArrays(ignore_undefined_fields=True)
        variations.put_vars(vcf)
        vcf_fhand.close()
        return variations

    def test_group_vars_per_block_test(self):
        vcf_fhand = io.BytesIO(VCF1)
        variations = self._get_var_array(vcf_fhand)

        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]

        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True)

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

        blocks = [{'chrom': b'20', 'start_idx': 0, 'stop_idx': 2}]

        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True)

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
                                             remove_snps_with_hets_or_missing=True)
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
                                             remove_snps_with_hets_or_missing=True)
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
                                             remove_snps_with_hets_or_missing=True)
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

        # Don't remove missing
        vcf = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
20	10	.	G	A	29	.	.	GT	0|0	0|0	1/1
20	20	.	T	C	3	.	.	GT	0|0	1|1	0/1
'''
        vcf_fhand = io.BytesIO(vcf)
        variations = self._get_var_array(vcf_fhand)
        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]

        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=False,
                                             variations_are_phased=True)
        grouped_vars = VariationsArrays(ignore_undefined_fields=True)
        grouped_vars.put_vars(var_grouper)
        assert grouped_vars[REF_FIELD] == [b'GT']
        assert numpy.all(grouped_vars[ALT_FIELD][0] == [b'GC', b'AT', b'AC'])
        assert numpy.all(grouped_vars[GT_FIELD] == [[[0, 0], [1, 1], [2, 3]]])
        assert grouped_vars[CHROM_FIELD] == [b'20']
        assert grouped_vars[POS_FIELD] == [10]
        assert grouped_vars['/variations/info/SN'] == [2]
        assert list(grouped_vars['/variations/info/AA'][0]) == [b'GT', b'GC',
                                                                b'AT', b'AC']

    def test_group_vars_per_block_deletion_test(self):
        blocks = [{'chrom': b'20', 'start': 10, 'stop': 21}]
        vcf = b'''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA1	NA2	NA3
20	10	.	GA	G	29	.	.	GT	0|0	0|0	1/1
20	20	.	T	C	3	.	.	GT	0|0	1|1	0/0
'''
        variations = self._get_var_array(io.BytesIO(vcf))
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True)
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
                                             remove_snps_with_hets_or_missing=True)
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

    def test_alleles_json(self):
        vcf_fhand = io.BytesIO(VCF1)
        variations = self._get_var_array(vcf_fhand)

        # 20    10    .    G    A    29    .    .    GT    0|0    0|0    1/1
        # 20    20    .    T    C    3    .    .    GT    0|0    1|1    0/0
        # 20    40    .    T    A    47    .    .    GT    0|0    1|1    0/0

        blocks = [{'chrom': b'20', 'start': 10, 'stop': 100}]
        pickle_fhand = io.BytesIO()
        var_grouper = BlocksVariationGrouper(variations, blocks,
                                             remove_snps_with_hets_or_missing=True,
                                             out_alleles_pickle_fhand=pickle_fhand)

        list(var_grouper.variations)
        pickle_fhand.seek(0)
        alleles = pickle.load(pickle_fhand)
        assert alleles == {(b'20', 10): [b'GTT', b'GCA', b'ATT']}


class BlockGenerationTest(unittest.TestCase):

    def test_snp_has_data(self):
        gts = numpy.array([[[0, 0], [-1, 0], [0, 0], [0, 0]], # 0  1 11
                           [[0, 0], [0, 0], [0, 0], [-1, 0]], # 1  1 12
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 2  1 13
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 3  1 14____
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 4  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 5  1 17
                           [[0, 0], [0, 0], [2, -1], [2, 2]], # 6  1 18____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 7  1 20
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 8  1 21____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 9  2 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10 2 11
                           ])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        result = _snp_has_enough_data(varis, max_missing_rate_in_ref_snp=0.1)
        expected = [False, False, True, True, True, True, False, True, True,
                    True, True]
        assert numpy.all(result == expected)

        result = _snp_has_enough_data(varis, max_missing_rate_in_ref_snp=0.9)
        assert numpy.all(result)

    def test_find_first(self):
        mat = numpy.arange(20)

        def lt(mat, value):
            return mat > value

        lt2 = partial(lt, value=2)
        assert find_first(mat, lt2, chunk_size=2) == 3

        lt19 = partial(lt, value=19)
        try:
            find_first(mat, lt19, chunk_size=2)
            self.fail('ValueError expected')
        except ValueError:
            pass

        lt2 = partial(lt, value=2)
        assert find_first(mat, lt2, start=4, chunk_size=2) == 4

        assert find_first(mat, lt2, chunk_size=None) == 3

        lt2 = partial(lt, value=2)
        assert find_first(mat, lt2, start=4, chunk_size=None) == 4

        try:
            find_first(mat, lt19, chunk_size=None)
            self.fail('ValueError expected')
        except ValueError:
            pass

    def test_find_last(self):
        mat = numpy.arange(20)

        def gt(mat, value):
            return mat < value

        gt2 = partial(gt, value=2)
        assert find_last(mat, gt2) == 1

    def test_find_first_in_matrices(self):

        def get_every_n(array, size=2):
            for idx in range(array.shape[0] // size):
                yield array[size * idx:2 * (idx + 1)]

        mat = numpy.arange(20)
        mats = list(get_every_n(mat, size=2))

        def lt(mat, value):
            return mat > value

        lt2 = partial(lt, value=2)
        assert find_first_in_matrices(mats, lt2) == 3

        lt19 = partial(lt, value=19)
        try:
            find_first_in_matrices(mats, lt19)
            self.fail('ValueError expected')
        except ValueError:
            pass

        lt2 = partial(lt, value=2)
        assert find_first_in_matrices(mats, lt2, offset=4) == 7

    def test_snp_correlation_with_one_snp(self):
        # poner un límite mínimo al rate de datos faltantes del snp prueba
        # Si hay datos faltantes en el snp prueba
        # ordenar por datos faltantes
        # buscar el snp con menos datos faltantes que correlaciona con el primero
        # si no se encuentra ninguno el primero no cambia de bloque, pasar al segundo
        # cuando se encuentre usar el snp elegido para ver las correlaciones en el bloque

        mat012 = [[0, 0, 2, 0, 0],
                  [0, 0, 2, 0, 0],
                  [2, 2, 0, 2, 1],
                  [2, 2, 0, 0, 0],
                  [2, 2, 0, 0, -1],
                  [2, 2, 2, 1, 0],
                  [0, 1, 1, -1, 0],
                 ]
        snp = numpy.array([0, 0, 2, 0, -1])
        result = _calc_if_snps_are_highly_correlated012(snp,
                                                        numpy.array(mat012),
                                                        min_num_gts_compared=10)
        expected = [True, True, True, False, False, False, False]
        assert numpy.all(result['snp_is_highly_correlated'] == expected)
        expected = [False, False, False, False, False, False, False]
        assert numpy.all(result['enough_information'] == expected)

        result = _calc_if_snps_are_highly_correlated012(snp,
                                                        numpy.array(mat012),
                                                        min_num_gts_compared=1)
        assert numpy.all(result['enough_information'])

        result = _calc_if_snps_are_highly_correlated012(snp,
                                                        numpy.array(mat012),
                                                        difference_rate_allowed=0.6)
        expected = [True, True, True, True, True, True, False]
        assert numpy.all(result['snp_is_highly_correlated'] == expected)

        result = _calc_if_snps_are_highly_correlated012(snp,
                                                        numpy.array(mat012),
                                                        difference_rate_allowed=0.9)
        assert numpy.all(result['snp_is_highly_correlated'])

    def test_generate_blocks(self):
        gts = numpy.array([[[0, 0], [0, 0], [0, 0], [0, 0]], # 0  1 11
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 1  1 12
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 2  1 13
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 3  1 14____
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 4  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 5  1 17
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 6  1 18____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 7  1 20
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 8  1 21____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 9  2 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10 2 11
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        pos = numpy.array([11, 12, 13, 14, 16, 17, 18, 20, 21, 10, 11])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=1, min_num_gts_compared=4)
        from pprint import pprint
        # pprint(list(blocks))

        expected = [{'chrom': 1, 'start': 11, 'start_idx': 0, 'stop': 15, 'stop_idx': 4},
                    {'chrom': 1, 'start': 16, 'start_idx': 4, 'stop': 19, 'stop_idx': 7},
                    {'chrom': 1, 'start': 20, 'start_idx': 7, 'stop': 22, 'stop_idx': 9},
                    {'chrom': 2, 'start': 10, 'start_idx': 9, 'stop': 12, 'stop_idx': 11}]
        assert list(blocks) == expected

        blocks = generate_blocks(varis, chunk_size=100, min_num_gts_compared=4)
        assert list(blocks) == expected

        blocks = generate_blocks(varis, chunk_size=2, min_num_gts_compared=4)
        assert list(blocks) == expected

    def test_generate_blocks_with_missing(self):
        for chunk_size in [1, 2, 3, 4, 5, 6, 10, 20]:
            self._generate_blocks_with_missing_test(chunk_size)

    def _generate_blocks_with_missing_test(self, chunk_size):

        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 0  1 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 1  1 11
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 2  1 12
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 3  1 13
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 4  1 14
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 5  1 15____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 6  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 7  1 18
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 8  1 19
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 9  1 20____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10  1 21
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 11  1 22____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 2  2 9
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 13  2 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 14 2 11
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        pos = numpy.array([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 9, 10, 11])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint; blocks = list(blocks); pprint(blocks)

        # 1  6
        # 7  10
        # 10 12
        # 13 15

        expected = [{'chrom': 1, 'start': 11, 'start_idx': 1, 'stop': 16, 'stop_idx': 6},
                    {'chrom': 1, 'start': 18, 'start_idx': 7, 'stop': 21, 'stop_idx': 10},
                    {'chrom': 1, 'start': 21, 'start_idx': 10, 'stop': 23, 'stop_idx': 12},
                    {'chrom': 2, 'start': 10, 'start_idx': 13, 'stop': 12, 'stop_idx': 15}]
        assert list(blocks) == expected

        # Last has no info
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 0  1 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 1  1 11
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 2  1 12
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 3  1 13
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 4  1 14
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 5  1 15____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 6  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 7  1 18
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 8  1 19
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 9  1 20____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10  1 21
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 11  1 22____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 12  2 9
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 13  2 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 14 2 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 15 2 15
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        pos = numpy.array([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 9, 10, 11, 15])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint
        # pprint(list(blocks))
        # 1  6
        # 7  10
        # 10 12
        # 13 15

        expected = [{'chrom': 1, 'start': 11, 'start_idx': 1, 'stop': 16, 'stop_idx': 6},
                    {'chrom': 1, 'start': 18, 'start_idx': 7, 'stop': 21, 'stop_idx': 10},
                    {'chrom': 1, 'start': 21, 'start_idx': 10, 'stop': 23, 'stop_idx': 12},
                    {'chrom': 2, 'start': 10, 'start_idx': 13, 'stop': 12, 'stop_idx': 15}]
        assert list(blocks) == expected

        # no info in chrom change
        # Last has no info
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 0  1 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 1  1 11
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 2  1 12
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 3  1 13
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 4  1 14
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 5  1 15____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 6  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 7  1 18
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 8  1 19
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 9  1 20____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10  1 21
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 11  1 22____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 12  2 9
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 13  2 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 14 2 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 15 2 15
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        pos = numpy.array([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 9, 10, 11, 15])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint
        # pprint(list(blocks))
        # 1  6
        # 7  10
        # 10 12
        # 14 15

        expected = [{'chrom': 1, 'start': 11, 'start_idx': 1, 'stop': 16, 'stop_idx': 6},
                    {'chrom': 1, 'start': 18, 'start_idx': 7, 'stop': 21, 'stop_idx': 10},
                    {'chrom': 1, 'start': 21, 'start_idx': 10, 'stop': 23, 'stop_idx': 12},
                    {'chrom': 2, 'start': 11, 'start_idx': 14, 'stop': 12, 'stop_idx': 15}]
        assert list(blocks) == expected

        # Last has no info
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 0  1 10
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 1  1 11
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 2  1 12
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 3  1 13
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 4  1 14
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 5  1 15____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 6  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 7  1 18
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 8  1 19
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 9  1 20____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10  1 21
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 11  1 22____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 12  2 9
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 13  2 10
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 14 2 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 15 2 15
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        pos = numpy.array([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 9, 10, 11, 15])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint
        # pprint(list(blocks))
        # 1  6
        # 7  10
        # 10 12

        expected = [{'chrom': 1, 'start': 11, 'start_idx': 1, 'stop': 16, 'stop_idx': 6},
                    {'chrom': 1, 'start': 18, 'start_idx': 7, 'stop': 21, 'stop_idx': 10},
                    {'chrom': 1, 'start': 21, 'start_idx': 10, 'stop': 23, 'stop_idx': 12}]
        assert list(blocks) == expected

        # Last has no info
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 0  1 10
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 1  1 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 2  1 12
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 3  1 13
                           [[1, 1], [1, 1], [1, 1], [1, 1]], # 4  1 14
                           [[2, 2], [2, 2], [2, 2], [2, 2]], # 5  1 15____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 6  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 7  1 18
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 8  1 19
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 9  1 20____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10  1 21
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 11  1 22____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 12  2 9
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 13  2 10
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 14 2 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 15 2 15
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        pos = numpy.array([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 9, 10, 11, 15])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint
        # pprint(list(blocks))
        # 1  6
        # 7  10
        # 10 12

        expected = [{'chrom': 1, 'start': 14, 'start_idx': 4, 'stop': 16, 'stop_idx': 6},
                    {'chrom': 1, 'start': 18, 'start_idx': 7, 'stop': 21, 'stop_idx': 10},
                    {'chrom': 1, 'start': 21, 'start_idx': 10, 'stop': 23, 'stop_idx': 12}]
        assert list(blocks) == expected

        # Last has no info
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 0  1 10
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 1  1 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 2  1 12
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 3  1 13
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 4  1 14
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 5  1 15____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 6  1 16
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 7  1 18
                           [[2, 2], [2, 2], [0, 0], [0, 0]], # 8  1 19
                           [[0, 0], [0, 0], [2, 2], [2, 2]], # 9  1 20____
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 10  1 21
                           [[0, 0], [0, 0], [0, 0], [0, 0]], # 11  1 22____
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 12  2 9
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 13  2 10
                           [[-1, 0], [-1, 0], [-1, 0], [-1, 0]], # 14 2 11
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]], # 15 2 15
                           ])
        chroms = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        pos = numpy.array([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 9, 10, 11, 15])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint
        # pprint(list(blocks))
        # 1  6
        # 7  10
        # 10 12

        expected = [{'chrom': 1, 'start': 18, 'start_idx': 7, 'stop': 21, 'stop_idx': 10},
                    {'chrom': 1, 'start': 21, 'start_idx': 10, 'stop': 23, 'stop_idx': 12}]
        assert list(blocks) == expected

        # No blocks
        gts = numpy.array([[[-1, -1], [-1, -1], [-1, 0], [-1, 0]],
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]],
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]],
                           [[-1, -1], [-1, -1], [-1, 0], [-1, 0]],
                           ])
        chroms = numpy.array([1, 1, 2, 3])
        pos = numpy.array([10, 11, 12, 13])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        varis[CHROM_FIELD] = chroms
        varis[POS_FIELD] = pos

        blocks = generate_blocks(varis, chunk_size=chunk_size,
                                 min_num_gts_compared=4)
        # from pprint import pprint
        # pprint(list(blocks))

        assert not list(blocks)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'GroupVarsPerBlockTest.test_group_vars_per_block_test']
    unittest.main()

