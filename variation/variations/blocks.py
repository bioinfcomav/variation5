
import copy
import pickle
import itertools
from functools import partial
import operator

import numpy

from variation import (REF_FIELD, ALT_FIELD, POS_FIELD, GT_FIELD, DEF_METADATA,
                       MISSING_INT, CHROM_FIELD, SNPS_PER_CHUNK)
from variation.variations import stats
from variation.variations.filters import NoMissingGTsOrHetFilter, FLT_VARS
from variation.matrix.methods import is_missing

ALIGNED_ALLELES_FIELD_NAME = b'AA'
NUMBER_OF_SNPS_FIELD_NAME = b'SN'


def _create_new_alleles_and_genotypes(variations, variations_are_phased):

    gts = variations[GT_FIELD]
    ref_allele = tuple([0] * variations.num_variations)
    alleles = {ref_allele: 0}
    samples_alleles = []
    ploidy = gts.shape[2]
    for sample_idx in range(len(variations.samples)):
        sample_gtss = [tuple(gts[:, sample_idx, allele_idx]) for allele_idx in range(ploidy)]
        sample_gtss = gts[:, sample_idx, :]
        sample_alleles = []
        for allele_idx in range(ploidy):
            sample_gts = tuple(sample_gtss[:, allele_idx])
            try:
                sample_allele = alleles[sample_gts]
            except KeyError:
                new_allele = len(alleles)
                alleles[sample_gts] = new_allele
                sample_allele = new_allele
            sample_alleles.append(sample_allele)
        samples_alleles.append(sample_alleles)

    refs_for_each_snp = variations[REF_FIELD]
    alt_for_each_snp = variations[ALT_FIELD]

    allele_lengths_for_snps = []
    for snp_idx in range(variations.num_variations):
        ref_len = len(refs_for_each_snp[snp_idx])
        alt_lens = [len(alt_allele) for alt_allele in alt_for_each_snp[snp_idx]]
        allele_len = max([ref_len] + alt_lens)
        allele_lengths_for_snps.append(allele_len)

    snp_alleles_with_letters = {}
    for haplotype, allele_number in alleles.items():
        letter_haplotype = []
        for snp_idx, snp_allele in enumerate(haplotype):
            if snp_allele == 0:
                letter_snp_allele = refs_for_each_snp[snp_idx]
            else:
                letter_snp_allele = alt_for_each_snp[snp_idx][snp_allele - 1]
            letter_snp_allele = letter_snp_allele.ljust(allele_lengths_for_snps[snp_idx], b'-')
            letter_haplotype.append(letter_snp_allele)
        letter_haplotype = b''.join(letter_haplotype)
        snp_alleles_with_letters[allele_number] = letter_haplotype

    info_alleles = snp_alleles_with_letters
    snp_alleles_with_letters = {key: allele.replace(b'-', b'') for key, allele in snp_alleles_with_letters.items()}

    ref_allele = snp_alleles_with_letters[0]
    alt_allele_numbers = list(set(alleles.values()).difference([0]))

    alt_alleles = [snp_alleles_with_letters[alt_allele_number] for alt_allele_number in range(1, len(alt_allele_numbers) + 1)]

    gts = numpy.array(samples_alleles)
    return {'ref_allele': ref_allele,
            'alt_alleles': alt_alleles,
            'gts': gts,
            'all_alleles_aligned': info_alleles}


class BlocksVariationGrouper():

    def __init__(self, variations, blocks, min_num_vars_in_block=1,
                 remove_snps_with_hets_or_missing=False,
                 pre_read_max_size=None, max_field_lens=None,
                 max_field_str_lens=None, max_n_vars=None,
                 n_threads=None, out_alleles_pickle_fhand=None,
                 variations_are_phased=False):

        self.variations_are_phased = variations_are_phased

        if n_threads is not None:
            raise NotImplemented('Parallelization is not implemented')
        self.n_threads = n_threads

        self.pre_read_max_size = pre_read_max_size

        if max_field_lens is None:
            self._max_field_lens = {'alt': 0,
                                    'INFO': {ALIGNED_ALLELES_FIELD_NAME: 0}}
        else:
            self._max_field_lens = max_field_lens
        if max_field_str_lens is None:
            self._max_field_str_lens = {'alt': 0,
                                        'chrom': 1,
                                        'ref': 1,
                                        'id': 0,
                                        'INFO': {ALIGNED_ALLELES_FIELD_NAME: 1}}
        else:
            self._max_field_str_lens = max_field_str_lens

        metadata = copy.deepcopy(DEF_METADATA)
        metadata['INFO'] = {NUMBER_OF_SNPS_FIELD_NAME:
                                   {'Description': 'SNP Number',
                                    'dtype': 'int',
                                    'Number': '1'},
                            ALIGNED_ALLELES_FIELD_NAME:
                                   {'Description': 'Aligned Alleles',
                                    'dtype': 'str',
                                    'Number': 'R'}}
        self.metadata = metadata
        self.out_alleles_pickle_fhand = out_alleles_pickle_fhand
        if self.out_alleles_pickle_fhand is not None:
            del metadata['INFO'][ALIGNED_ALLELES_FIELD_NAME]

        self._variations_to_group = variations
        self.blocks = iter(blocks)
        self.min_num_vars_in_block = min_num_vars_in_block
        self.remove_snps_with_hets_or_missing = remove_snps_with_hets_or_missing

    @property
    def samples(self):
        return [sample.encode() for sample in self._variations_to_group.samples]

    @property
    def ploidy(self):
        return self._variations_to_group.ploidy

    @property
    def variations(self):
        variations = self._variations_to_group
        index = variations.pos_index

        if self.remove_snps_with_hets_or_missing:
            flt = NoMissingGTsOrHetFilter()

        alleles_pickle_fhand = self.out_alleles_pickle_fhand
        aligned_alleles_for_snps = {}

        for block in self.blocks:
            chrom = block['chrom']
            try:
                start_idx = block['start_idx']
            except KeyError:
                start_idx = index.index_pos(chrom, block['start'])
            try:
                stop_idx = block['stop_idx']
            except KeyError:
                stop_idx = index.index_pos(chrom, block['stop'])

            block_chunk = variations.get_chunk(slice(start_idx, stop_idx))
            if block_chunk.num_variations < self.min_num_vars_in_block:
                continue

            if self.remove_snps_with_hets_or_missing:
                block_chunk = flt(block_chunk)[FLT_VARS]
            else:
                gt_is_het, gt_is_missing = stats._call_is_het(block_chunk, min_call_dp=0)
                if self.variations_are_phased:
                    gt_is_not_missing = numpy.logical_not(gt_is_missing)
                    all_gts_are_not_missing = numpy.all(gt_is_not_missing)
                    if not all_gts_are_not_missing:
                        raise ValueError('Missing genotypes found, you could use remove_snps_with_hets_or_missing=True')
                else:
                    het_or_missing = numpy.logical_or(gt_is_het, gt_is_missing)
                    gt_is_hom_and_not_missing = numpy.logical_not(het_or_missing)
                    all_gts_are_hom_and_not_missing = numpy.all(gt_is_hom_and_not_missing)
                    if not all_gts_are_hom_and_not_missing:
                        raise ValueError('Missing or het genotypes found, you could use remove_snps_with_hets_or_missing=True')

            if block_chunk.num_variations < self.min_num_vars_in_block:
                continue

            snp_id = None
            start = block_chunk[POS_FIELD][0]

            alleles_and_gts = _create_new_alleles_and_genotypes(block_chunk,
                                                                self.variations_are_phased)
            ref_allele = alleles_and_gts['ref_allele']
            gts = alleles_and_gts['gts']
            alt_alleles = alleles_and_gts['alt_alleles']

            aligned_alleles = alleles_and_gts['all_alleles_aligned']
            aligned_alleles = [aligned_alleles[idx] for idx in range(len(aligned_alleles))]

            info = {NUMBER_OF_SNPS_FIELD_NAME: block_chunk.num_variations}

            if alleles_pickle_fhand is None:
                info[ALIGNED_ALLELES_FIELD_NAME] = aligned_alleles
            else:
                aligned_alleles_for_snps[(chrom, start)] = aligned_alleles
                ref_allele = None
                alt_alleles = None

            yield (chrom, start, snp_id, ref_allele, alt_alleles,
                   None, None, info, [(b'GT', gts)])

        if alleles_pickle_fhand is not None:
            pickle.dump(aligned_alleles_for_snps,
                        alleles_pickle_fhand)
            alleles_pickle_fhand.flush()


def _calc_if_snps_are_highly_correlated012(reference_snp_gt, mat012,
                                           difference_rate_allowed=0.05,
                                           min_num_gts_compared=10):

    reference_snp_gt = reference_snp_gt.copy()

    assert 3 not in reference_snp_gt
    reference_snp_gt[reference_snp_gt == MISSING_INT] = 3

    complementary_ref = reference_snp_gt.copy()
    complementary_ref[reference_snp_gt == 0] = 2
    complementary_ref[reference_snp_gt == 2] = 0

    assert reference_snp_gt.shape[0] == mat012.shape[1]

    same_gt_as_ref = mat012 == reference_snp_gt
    same_gt_as_comp = mat012 == complementary_ref

    num_same_gt_as_ref = numpy.sum(same_gt_as_ref, axis=1)
    num_same_gt_as_comp = numpy.sum(same_gt_as_comp, axis=1)
    num_gt_similar_to_ref = numpy.maximum(num_same_gt_as_ref,
                                          num_same_gt_as_comp)

    non_missing_in_gt_mat = mat012 != MISSING_INT

    non_missing_in_gt_mat[..., reference_snp_gt == 3] = False
    num_gts_evaluated = numpy.sum(non_missing_in_gt_mat, axis=1)

    enough_information = num_gts_evaluated >= min_num_gts_compared

    gt_matching_rate = num_gt_similar_to_ref / num_gts_evaluated
    snp_is_highly_correlated = gt_matching_rate > 1 - difference_rate_allowed
    return {'snp_is_highly_correlated': snp_is_highly_correlated,
            'enough_information': enough_information}


def _calc_if_snps_are_highly_correlated_and_same_chrom(chunk, reference_snp_gt,
                                                       reference_snp_chrom,
                                                       difference_rate_allowed=0.05,
                                                       min_num_gts_compared=10):
    result = _calc_if_snps_are_highly_correlated012(reference_snp_gt,
                                                    chunk.gts_as_mat012,
                                                    difference_rate_allowed=difference_rate_allowed,
                                                    min_num_gts_compared=min_num_gts_compared)
    same_chrom = chunk[CHROM_FIELD] == reference_snp_chrom
    # result['highly_correlated_and_same_chrom'] = numpy.logical_and(result['snp_is_highly_correlated'],
    #                                                               same_chrom)
    # del result['snp_is_highly_correlated']
    highly_correlated_or_not_enough_info = numpy.logical_or(result['snp_is_highly_correlated'],
                                                            numpy.logical_not(result['enough_information']))

    same_chrom_and_highly_correlated_or_not_enough_info = numpy.logical_and(same_chrom,
                                                                            highly_correlated_or_not_enough_info)
    return {'same_chrom_and_highly_correlated_or_not_enough_info':
            same_chrom_and_highly_correlated_or_not_enough_info,
            'enough_information': result['enough_information']}


def _find_first_in_array(array, funct_condition, start=0):
    if start:
        array = array[start:]
    result = funct_condition(array)
    nonzero = numpy.flatnonzero(result)
    if nonzero.size:
        return nonzero[0] + start

    raise ValueError('Condition not satisfied')


def find_first(array, funct_condition, chunk_size=1024, start=0):
    if not chunk_size:
        return _find_first_in_array(array, funct_condition, start=start)

    chunk_start = start
    chunk_stops = itertools.chain(range(chunk_start + chunk_size,
                                        array.size, chunk_size),
                                  [None])

    for chunk_stop in chunk_stops:
        chunk = array[chunk_start:chunk_stop]
        result = funct_condition(chunk)
        nonzero = numpy.flatnonzero(result)
        if nonzero.size:
            return nonzero[0] + chunk_start
        chunk_start = chunk_stop

    raise ValueError('Condition not satisfied')


def find_last(array, funct_condition=None):

    if funct_condition is None:
        result = array
    else:
        result = funct_condition(array)

    nonzero = numpy.flatnonzero(result)
    if nonzero.size:
        return nonzero[-1]

    raise ValueError('Condition not satisfied')


def find_first_in_matrices(arrays, funct_condition=None, offset=0,
                           array_len_funct=None):
    if array_len_funct is None:
        array_len_funct = lambda array: array.shape[0]

    for array in arrays:
        if funct_condition is None:
            result = array
        else:
            result = funct_condition(array)
        nonzero = numpy.flatnonzero(result)
        if nonzero.size:
            return nonzero[0] + offset
        offset += array_len_funct(array)

    raise ValueError('Condition not satisfied')


def _snp_has_enough_data(variations, max_missing_rate_in_ref_snp=0.1):
    gts = variations[GT_FIELD]
    missing_rate = numpy.sum(is_missing(gts, axis=2), axis=1) / gts.shape[1]
    return missing_rate < max_missing_rate_in_ref_snp


def _min(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return min((a, b))


def generate_blocks(variations, difference_rate_allowed=0.05,
                    min_num_gts_compared=10, chunk_size=100,
                    max_missing_rate_in_ref_snp=0.9,
                    debug=False):

    # an example
    # SNP correlated and in the same chromosome -> 1
    # SNP with no info X
    # SNP not correlated or in different chromome -> 0

    # Case 1:
    #         1 1 1 X 1 0
    # offset  ^
    # ref     ^
    # block   <------->
    # Next offset       ^
    # Next ref          ^

    # Case 2:
    #         X 1 1 X 0 0
    # offset  ^
    # ref       ^
    # block     <->
    # next offset   ^
    # next ref        ^

    # algorithm
    # choose as reference SNP the first SNP after the offset that has enough information
    # this reference SNP will start the block
    # look for first SNP that is in a different chromosome or is not correlated
    # look for last SNP with info previous to the first non correlated SNP

    enough_data_funct = partial(_snp_has_enough_data,
                                max_missing_rate_in_ref_snp=max_missing_rate_in_ref_snp)
    kept_fields = [GT_FIELD, CHROM_FIELD, POS_FIELD]
    chroms = variations[CHROM_FIELD]
    pos = variations[POS_FIELD]
    num_vars = variations.num_variations

    if debug:
        print('DEBUG' * 30)

    offset_idx = 0
    while True:
        if debug:
            print('offset_idx', offset_idx)
        chunks = variations.iterate_chunks(start=offset_idx,
                                           chunk_size=chunk_size,
                                           kept_fields=kept_fields)
        try:
            ref_snp_idx_rel_to_offset = find_first_in_matrices(chunks,
                                                               enough_data_funct,
                                                               array_len_funct=lambda chunk: chunk.num_variations)
        except ValueError:
            # No SNP with enough info to serve as reference
            break

        ref_snp_idx = ref_snp_idx_rel_to_offset + offset_idx
        if debug:
            print('ref_snp_idx:', ref_snp_idx)

        first_snp = variations.get_chunk(slice(ref_snp_idx, ref_snp_idx + 1),
                                         kept_fields=kept_fields)
        ref_snp_012_gt = first_snp.gts_as_mat012[0]
        reference_snp_chrom = first_snp[CHROM_FIELD][0]

        calc_if_snps_are_highly_correlated_and_same_chrom = partial(_calc_if_snps_are_highly_correlated_and_same_chrom,
                                                                    reference_snp_gt=ref_snp_012_gt,
                                                                    reference_snp_chrom=reference_snp_chrom,
                                                                    difference_rate_allowed=difference_rate_allowed,
                                                                    min_num_gts_compared=min_num_gts_compared)

        highly_correlated_offset2 = ref_snp_idx + 1
        chunks = variations.iterate_chunks(start=highly_correlated_offset2,
                                           chunk_size=chunk_size,
                                           kept_fields=kept_fields)
        highly_correlated_result = map(calc_if_snps_are_highly_correlated_and_same_chrom,
                                       chunks)
        highly_correlated1, highly_correlated2 = itertools.tee(highly_correlated_result)
        highly_correlated = map(operator.itemgetter('same_chrom_and_highly_correlated_or_not_enough_info'),
                                highly_correlated1)
        try:
            first_not_correlated_idx_rel_to_offset2 = find_first_in_matrices(highly_correlated,
                                                                             numpy.logical_not)
        except ValueError:
            first_not_correlated_idx_rel_to_offset2 = None
        if first_not_correlated_idx_rel_to_offset2 is not None:
            first_not_correlated_idx = first_not_correlated_idx_rel_to_offset2 + highly_correlated_offset2
            stop_idx_due_to_lack_of_correlation = first_not_correlated_idx
        else:
            first_not_correlated_idx = None
            stop_idx_due_to_lack_of_correlation = num_vars
        if debug:
            print('first_not_correlated_idx_rel_to_offset2: ', first_not_correlated_idx_rel_to_offset2)
            print('first_not_correlated_idx:', first_not_correlated_idx)
            print('stop_idx_due_to_lack_of_correlation:', stop_idx_due_to_lack_of_correlation)

        not_enough_info = numpy.concatenate([numpy.logical_not(array_chunk['enough_information']) for array_chunk in highly_correlated2])

        try:
            last_with_no_info_before_first_non_correlated_idx_rel_to_offset2 = find_last(not_enough_info[:first_not_correlated_idx])
        except ValueError:
            last_with_no_info_before_first_non_correlated_idx_rel_to_offset2 = None
        if debug:
            print('last_with_no_info_before_first_non_correlated_idx_rel_to_offset2:', last_with_no_info_before_first_non_correlated_idx_rel_to_offset2)

        if last_with_no_info_before_first_non_correlated_idx_rel_to_offset2 is not None:
            enough_info = numpy.logical_not(not_enough_info[:last_with_no_info_before_first_non_correlated_idx_rel_to_offset2])
            try:
                last_with_info_before_first_non_correlated_idx_rel_to_offset2 = find_last(enough_info)
            except ValueError:
                last_with_info_before_first_non_correlated_idx_rel_to_offset2 = None
            if debug:
                print('last_with_info_before_first_non_correlated_idx_rel_to_offset2:', last_with_info_before_first_non_correlated_idx_rel_to_offset2)

            if last_with_info_before_first_non_correlated_idx_rel_to_offset2 is None:
                stop_idx_due_to_lack_of_info = ref_snp_idx + 1
            else:
                stop_idx_due_to_lack_of_info = last_with_info_before_first_non_correlated_idx_rel_to_offset2 + highly_correlated_offset2 + 1
            if debug:
                print('stop_idx_due_to_lack_of_info: ', stop_idx_due_to_lack_of_info)
        else:
            stop_idx_due_to_lack_of_info = None

        stop_idx = _min(stop_idx_due_to_lack_of_correlation, stop_idx_due_to_lack_of_info)

        if debug:
            print('stop_idx:', stop_idx)

        block_start_idx = ref_snp_idx
        block_stop_idx = stop_idx
        block = {'chrom': chroms[block_start_idx],
                 'start': pos[block_start_idx],
                 'stop': pos[block_stop_idx - 1] + 1,
                 'start_idx': block_start_idx,
                 'stop_idx': block_stop_idx}
        if debug:
            print('BLOCK -> chrom: {}, start_idx: {}, stop_idx: {}'.format(block['chrom'],
                                                                           block['start_idx'],
                                                                           block['stop_idx']))
        yield block
        assert block_start_idx < block_stop_idx

        offset_idx = block_stop_idx
        if offset_idx > num_vars - 1:
            break
