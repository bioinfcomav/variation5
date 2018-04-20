
import copy

import numpy

from variation import REF_FIELD, ALT_FIELD, POS_FIELD, GT_FIELD, DEF_METADATA
from variation.variations import stats
from variation.variations.filters import NoMissingGTsOrHetFilter, FLT_VARS
from variation.gt_parsers.vcf import _VarParserWithPreRead
from Cython.Compiler.ExprNodes import NoneNode

ALIGNED_ALLELES_FIELD_NAME = b'AA'
NUMBER_OF_SNPS_FIELD_NAME = b'SN'


def _create_new_alleles_and_genotypes(variations):

    gts = variations[GT_FIELD]
    ref_allele = tuple([0] * variations.num_variations)
    alleles = {ref_allele: 0}
    sample_alleles = []
    for sample_idx in range(len(variations.samples)):
        sample_gts = tuple(gts[:, sample_idx, 0]) # we're assuming homozygosity here
        try:
            sample_allele = alleles[sample_gts]
        except KeyError:
            new_allele = len(alleles)
            alleles[sample_gts] = new_allele
            sample_allele = new_allele
        sample_alleles.append(sample_allele)

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

    gts = numpy.repeat(sample_alleles, 2).reshape(len(variations.samples), variations.ploidy)

    return {'ref_allele': ref_allele,
            'alt_alleles': alt_alleles,
            'gts': gts,
            'all_alleles_aligned': info_alleles}


class BlocksVariationGrouper(_VarParserWithPreRead):

    def __init__(self, variations, blocks, min_num_vars_in_block=1,
                 remove_snps_with_hets_or_missing=False,
                 pre_read_max_size=None, max_field_lens=None,
                 max_field_str_lens=None, max_n_vars=None,
                 n_threads=None):

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

        self._variations_to_group = variations
        self.blocks = iter(blocks)
        self.min_num_vars_in_block = min_num_vars_in_block
        self.remove_snps_with_hets_or_missing = remove_snps_with_hets_or_missing
        super().__init__(pre_read_max_size=pre_read_max_size,
                         max_n_vars=max_n_vars)

    @property
    def samples(self):
        return [sample.encode() for sample in self._variations_to_group.samples]

    @property
    def ploidy(self):
        return self._variations_to_group.ploidy

    @property
    def max_field_lens(self):
        return self._max_field_lens

    @property
    def max_field_str_lens(self):
        return self._max_field_str_lens

    def _variations_for_cache(self):
        for snp in self._variations():
            yield snp

    def _variations(self, n_threads=None):
        variations = self._variations_to_group
        index = variations.pos_index

        if self.remove_snps_with_hets_or_missing:
            flt = NoMissingGTsOrHetFilter()

        max_field_lens = self._max_field_lens
        max_field_str_lens = self._max_field_str_lens

        for block in self.blocks:
            chrom = block['chrom']
            start_idx = index.index_pos(chrom, block['start'])
            stop_idx = index.index_pos(chrom, block['stop'])
            block_chunk = variations.get_chunk(slice(start_idx, stop_idx))
            if block_chunk.num_variations < self.min_num_vars_in_block:
                continue

            if self.remove_snps_with_hets_or_missing:
                block_chunk = flt(block_chunk)[FLT_VARS]
            else:
                gt_is_het, gt_is_missing = stats._call_is_het(block_chunk, min_call_dp=0)
                het_or_missing = numpy.logical_or(gt_is_het, gt_is_missing)
                gt_is_hom_and_not_missing = numpy.logical_not(het_or_missing)
                all_gts_are_hom_and_not_missing = numpy.all(gt_is_hom_and_not_missing)
                if not all_gts_are_hom_and_not_missing:
                    raise ValueError('Missing or het genotypes found, you could use remove_snps_with_hets_or_missing=True')

            if block_chunk.num_variations < self.min_num_vars_in_block:
                continue

            snp_id = None
            start = block_chunk[POS_FIELD][0]

            alleles_and_gts = _create_new_alleles_and_genotypes(block_chunk)
            ref_allele = alleles_and_gts['ref_allele']
            alt_alleles = alleles_and_gts['alt_alleles']
            gts = alleles_and_gts['gts']

            if max_field_str_lens['ref'] < len(ref_allele):
                max_field_str_lens['ref'] = len(ref_allele)
            if snp_id is not None and max_field_str_lens['id'] < len(snp_id):
                max_field_str_lens['id'] = len(snp_id)

            if alt_alleles:
                if max_field_lens['alt'] < len(alt_alleles):
                    max_field_lens['alt'] = len(alt_alleles)
                max_len = max(len(allele) for allele in alt_alleles)
                if max_field_str_lens['alt'] < max_len:
                    max_field_str_lens['alt'] = max_len

            aligned_alleles = alleles_and_gts['all_alleles_aligned']
            aligned_alleles = [aligned_alleles[idx] for idx in range(len(aligned_alleles))]

            if len(aligned_alleles) > max_field_lens['INFO'][ALIGNED_ALLELES_FIELD_NAME]:
                max_field_lens['INFO'][ALIGNED_ALLELES_FIELD_NAME] = len(aligned_alleles)
            max_aa_len = max([len(aligned_alleles) for allele in aligned_alleles])
            if max_aa_len > max_field_str_lens['INFO'][ALIGNED_ALLELES_FIELD_NAME]:
                max_field_str_lens['INFO'][ALIGNED_ALLELES_FIELD_NAME] = max_aa_len
            if len(ref_allele) > max_field_str_lens['ref']:
                max_field_str_lens['ref'] = len(ref_allele)
            if len(chrom) > max_field_str_lens['chrom']:
                max_field_str_lens['chrom'] = len(chrom)

            info = {ALIGNED_ALLELES_FIELD_NAME: aligned_alleles,
                    NUMBER_OF_SNPS_FIELD_NAME: block_chunk.num_variations}

            yield (chrom, start, snp_id, ref_allele, alt_alleles,
                   None, None, info, [(b'GT', gts)])
