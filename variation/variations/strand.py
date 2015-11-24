import numpy

from variation import MISSING_STR

COMPLEMENTARY = {b'A': b'T', b'G': b'C', b'T': b'A', b'C': b'G', b'': b''}


def count_compatible_snps(variations, strands_to_check_matrix,
                          ref_strand_alleles, max_snps_check=None):
    ref = variations['/variations/ref'][:]
    alt = variations['/variations/alt'][:]

    strand_counts = [0] * strands_to_check_matrix.shape[1]
    snps_checked = 0
    missing_alleles = [MISSING_STR.encode()]
    for var_mat_ref, var_mat_alt, ref_alleles, snp_strand_to_check in zip(ref, alt,
                                                     ref_strand_alleles,
                                                     strands_to_check_matrix):
        var_mat_alleles = set(var_mat_alt)
        var_mat_alleles.add(var_mat_ref)
        var_mat_alleles = var_mat_alleles.difference(missing_alleles)
        compat_ref_strand, compat_rev_ref_strand = _check_compatible_strands(var_mat_alleles, ref_alleles)
        for strand_idx, strand_is_like_reference in enumerate(snp_strand_to_check):
            if strand_is_like_reference and compat_ref_strand:
                strand_counts[strand_idx] += 1
            elif not strand_is_like_reference and compat_rev_ref_strand:
                strand_counts[strand_idx] += 1
        snps_checked += 1
        if max_snps_check is not None and max_snps_check <= snps_checked:
            break

    return snps_checked, strand_counts


def _check_compatible_strands(var_mat_alleles, ref_alleles):
    rev_comp_ref_alleles = _rev_compl(ref_alleles)
    compatible_ref_strand = False
    compatible_rev_ref_strand = False
    if not var_mat_alleles.difference(ref_alleles):
        compatible_ref_strand = True
    if not var_mat_alleles.difference(rev_comp_ref_alleles):
        compatible_rev_ref_strand = True
    return compatible_ref_strand, compatible_rev_ref_strand


def _rev_compl(seq):
    return numpy.array([COMPLEMENTARY[x] for x in seq])


def change_strand(variations, orig_orientation, final_orientation):
    ref_alleles = variations['/variations/ref'][:]
    alt_alleles = variations['/variations/alt'][:]
    recoded_ref = []
    recoded_alts = []
    mask = orig_orientation == final_orientation
    for ref, alts, mask_ in zip(ref_alleles, alt_alleles, mask):
        if not mask_:
            ref = COMPLEMENTARY[ref]
            alts = _rev_compl(alts)
        recoded_ref.append(ref)
        recoded_alts.append(alts)
    variations['/variations/ref'][:] = recoded_ref
    variations['/variations/alt'][:] = recoded_alts
