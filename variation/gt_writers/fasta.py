
from collections import defaultdict, Counter

import numpy

from Bio import pairwise2

from variation import (GT_FIELD, CHROM_FIELD, POS_FIELD, REF_FIELD, ALT_FIELD,
                       MISSING_INT, MISSING_STR)
from variation.variations.filters import IndelFilter, FLT_VARS
from variation.matrix.stats import counts_and_allels_by_row
from variation.gt_writers.vcf import _join_str_array_along_axis0

INDEL_CHAR = b'-'


class AlignmentTooDifficultError(ValueError):
    pass


def _do_easy_multiple_alignment(alleles, lengths):
    longest_allele_idx = numpy.argmax(numpy.array(lengths))
    longest_allele = alleles[longest_allele_idx]

    aligned_alleles = []
    for allele in alleles:
        if allele == longest_allele:
            aligned_allele = allele
        elif not allele.strip():
            aligned_allele == INDEL_CHAR * len(allele)
        else:
            alignment = pairwise2.align.globalxx(longest_allele.decode(),
                                                 allele.decode())[0]
            aligned_allele = alignment[1].encode()
        aligned_alleles.append(aligned_allele)

    # check that this simple multiple alignment is fine
    for idx in range(len(longest_allele)):
        nucleotides = {allele[idx] for allele in aligned_alleles}.difference([45])
        if len(nucleotides) > 1:
            raise AlignmentTooDifficultError('Alignment too difficult')
    return aligned_alleles


def _fix_allele_lengths(alleles, try_to_align_easy_indels,
                        put_hyphens_in_indels):
    if not put_hyphens_in_indels:
        return alleles

    lengths = [len(allele) for allele in alleles]
    one_length = len(alleles[0])
    if all(length == one_length or not length for length in lengths):
        return alleles

    if max(lengths) == 2:
        alleles = [allele + INDEL_CHAR if len(allele) == 1 else allele for allele in alleles]
        return alleles

    if try_to_align_easy_indels:
        alleles = _do_easy_multiple_alignment(alleles, lengths)
    else:
        raise RuntimeError('We should not be here')
    return alleles


def write_fasta(variations, out_fhand, sample_class=None, remove_indels=True,
                set_hets_to_missing=True,
                remove_invariant_snps=False, remove_sites_all_N=False,
                try_to_align_easy_indels=False, put_hyphens_in_indels=True):

    if not set_hets_to_missing:
        raise NotImplementedError('Fixme')

    if try_to_align_easy_indels and not put_hyphens_in_indels:
        msg = 'try_to_align and not hyphens_in_indels are incompatible options'
        raise ValueError(msg)
    if try_to_align_easy_indels and put_hyphens_in_indels:
        pass
    if not try_to_align_easy_indels and not put_hyphens_in_indels:
        pass
    if not try_to_align_easy_indels and put_hyphens_in_indels:
        msg = 'not try_to_align and put hyphens_in_indels are incompatible options'
        raise ValueError(msg)

    stats = {}
    stats['complex_skipped'] = 0
    stats['snps_written'] = 0

    samples = variations.samples

    chroms = variations[CHROM_FIELD] if CHROM_FIELD in variations else None
    poss = variations[POS_FIELD] if POS_FIELD in variations else None

    if remove_indels:
        filter_indels = IndelFilter(report_selection=True)
        result = filter_indels(variations)
        variations = result[FLT_VARS]
        if chroms is not None:
            chroms = chroms[result['selected_vars']]
        if poss is not None:
            poss = poss[result['selected_vars']]

    N = b'N'
    desc = b''

    if chroms is not None and poss is not None:
        chrom0 = chroms[0].astype('S')
        pos0 = poss[0]
        chrom1 = chroms[-1].astype('S')
        pos1 = poss[-1]
        desc = b' From %s:%i to %s:%i' % (chrom0, pos0, chrom1, pos1)
        if stats is not None:
            stats['start_chrom'] = chrom0
            stats['start_pos'] = chrom0
            stats['end_chrom'] = chrom1
            stats['end_pos'] = chrom1
        if chrom0 == chrom1:
            desc += b' length covered:%i' % (pos1 - pos0)
            if stats is not None:
                stats['length_covered'] = pos1 - pos0

    refs = variations[REF_FIELD]
    alts = variations[ALT_FIELD]
    gts = variations[GT_FIELD][...]

    if alts.dtype.itemsize > refs.dtype.itemsize:
        str_dtype = alts.dtype
    else:
        str_dtype = refs.dtype

    if gts.shape[2] != 2:
        raise NotImplementedError('Not implemented yet for non diploids')

    # remove hets
    haps1 = gts[:, :, 0]
    haps2 = gts[:, :, 1]
    haps1[haps1 != haps2] = MISSING_INT
    haps = haps1

    haps_to_keep = None
    if remove_invariant_snps:
        counts = counts_and_allels_by_row(haps, missing_value=MISSING_INT)[0]
        haps_to_keep = numpy.sum(counts, axis=1) - numpy.max(counts, axis=1) > 0
    elif remove_sites_all_N:
        all_missing = numpy.all(haps == MISSING_INT, axis=1)
        haps_to_keep = numpy.logical_not(all_missing)

    if haps_to_keep is not None:
        haps = haps[haps_to_keep]
        alts = alts[haps_to_keep]
        refs = refs[haps_to_keep]

    letter_haps = numpy.full_like(haps, dtype=str_dtype, fill_value=b'')

    for snp_idx in range(haps.shape[0]):
        alleles = [refs[snp_idx]] + list(alts[snp_idx, :])

        lengths = [len(allele) for allele in alleles]
        len_longest_allele = max(lengths)
        empty_allele = N * len_longest_allele
        letter_haps[snp_idx, :] = empty_allele

        try:
            alleles = _fix_allele_lengths(alleles,
                                          try_to_align_easy_indels=try_to_align_easy_indels,
                                          put_hyphens_in_indels=put_hyphens_in_indels)
        except AlignmentTooDifficultError:
            # we don't know how to align this complex, so we skip it
            stats['complex_skipped'] += 1
            continue

        stats['snps_written'] += 1

        ref_allele = alleles[0]
        alt_alleles = alleles[1:]

        letter_haps[snp_idx, :][haps[snp_idx, :] == 0] = ref_allele
        for alt_allele_idx in range(len(alt_alleles)):
            alt_allele = alt_alleles[alt_allele_idx]
            if alt_allele == MISSING_STR:
                break
            letter_haps[snp_idx, :][haps[snp_idx, :] == alt_allele_idx + 1] = alt_allele

    letter_haps = _join_str_array_along_axis0(letter_haps.T,
                                              the_str_array_has_newlines=False)

    if stats is not None:
        stats['snps_written'] = letter_haps.shape[0]

    for smpl_idx, sample in enumerate(samples):
        this_desc = b'>%s' % sample.encode() + desc
        out_fhand.write(this_desc)
        out_fhand.write(b'\n')
        sample_hap = letter_haps[smpl_idx]
        out_fhand.write(sample_hap)
        out_fhand.write(b'\n')

    return {'stats': stats}

