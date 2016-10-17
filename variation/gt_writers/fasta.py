
import numpy

from variation import (GT_FIELD, CHROM_FIELD, POS_FIELD, REF_FIELD, ALT_FIELD,
                       MISSING_INT, MISSING_STR)
from variation.variations.filters import IndelFilter, FLT_VARS


def write_fasta(variations, out_fhand, sample_class=None, remove_indels=True):
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
    else:
        raise NotImplementedError('Fix me if you want indels')

    N = 'N'
    desc = ''

    if chroms is not None and poss is not None:
        chrom0 = chroms[0]
        pos0 = poss[0]
        chrom1 = chroms[-1]
        pos1 = poss[-1]
        desc = ' From %s:%i to %s:%i' % (chrom0, pos0, chrom1, pos1)
        if chrom0 == chrom1:
            desc += ' length covered:%i' % (pos1 - pos0)
    refs = variations[REF_FIELD]
    alts = variations[ALT_FIELD]
    gts = variations[GT_FIELD][...]

    if gts.shape[2] != 2:
        raise NotImplementedError('Not implemented yet for non diploids')

    haps1 = gts[:, :, 0]
    haps2 = gts[:, :, 1]
    haps1[haps1 != haps2] = MISSING_INT
    haps = haps1
    letter_haps = numpy.full_like(haps, dtype='S1', fill_value=N)

    n_alts = alts.shape[1]
    for snp_idx in range(haps.shape[0]):
        letter_haps[snp_idx, :][haps[snp_idx, :] == 0] = refs[snp_idx]
        for alt_allele_idx in range(n_alts):
            alt_allele = alts[snp_idx, alt_allele_idx]
            if alt_allele == MISSING_STR:
                break
            letter_haps[snp_idx, :][haps[snp_idx, :] == alt_allele_idx + 1] = alt_allele

    for smpl_idx, sample in enumerate(samples):
        this_desc = '>%s' % sample + desc
        out_fhand.write(this_desc)
        out_fhand.write('\n')
        sample_hap = letter_haps[:, smpl_idx]
        sample_hap = b''.join(sample_hap).decode()
        out_fhand.write(sample_hap)
        out_fhand.write('\n')
