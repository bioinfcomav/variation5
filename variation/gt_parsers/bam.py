
import copy
from collections import defaultdict, Counter, OrderedDict
import operator
from heapq import merge as merge_sorted_iterables
import itertools

import pysam

from scipy.stats import chi2
from variation import MISSING_INT, DEF_METADATA


def _get_read_group_samples(samfile):
    read_groups = samfile.header.get('RG')
    read_group_samples = {}
    if read_groups is not None:
        read_group_samples = {rg['ID']: rg.get('SM', rg['ID']) for rg in read_groups}
    read_group_samples[None] = None
    return read_group_samples


def _group_locus(location_and_locus_for_several_bams):
    location, locus_for_several_bams = location_and_locus_for_several_bams
    coverage = 0
    counts = defaultdict(Counter)
    for locus in locus_for_several_bams:
        coverage += locus['coverage']
        for sample, sample_counts in locus['kmer_counts'].items():
            counts[sample] += sample_counts

    locus = {'location': location,
             'coverage': coverage,
             'kmer_counts': counts}
    return locus


def _filter_samples_by_coverage(loci, min_sample_coverage,
                                filter_out_no_kmer_columns,
                                min_num_samples):
    for locus in loci:
        kmer_counts = locus['kmer_counts']
        good_samples = [sample for sample, counts in kmer_counts.items() if sum(counts.values()) >= min_sample_coverage]
        if len(good_samples) < min_num_samples:
            continue
        if filter_out_no_kmer_columns and not good_samples:
            continue
        kmer_counts = {sample: kmer_counts[sample] for sample in good_samples}
        locus['kmer_counts'] = kmer_counts
        yield locus


def _parse_bams(bam_fpaths, kmer_size, filter_out_no_kmer_columns=True,
                min_sample_coverage=0, min_num_samples=0):
    sams = []
    references = None
    samples = set()
    for bam_fpath in bam_fpaths:
        samfile = pysam.AlignmentFile(bam_fpath, 'rb')

        this_references = samfile.references
        if references is None:
            references = this_references
            reference_index = {reference: idx for idx, reference in enumerate(references)}
        else:
            assert references == this_references

        read_group_samples = _get_read_group_samples(samfile)
        samples.update(read_group_samples.values())
        loci = _parse_bam(samfile, read_group_samples, kmer_size,
                          filter_out_no_kmer_columns=filter_out_no_kmer_columns,
                          reference_index=reference_index)
        sams.append({'fpath': bam_fpath,
                     'samfile': samfile,
                     'read_groups': read_group_samples,
                     'loci': loci})

    location_getter = operator.itemgetter('location')
    sorted_loci = merge_sorted_iterables(*(sam['loci'] for sam in sams),
                                         key=location_getter)
    grouped_loci = itertools.groupby(sorted_loci, key=location_getter)
    loci = map(_group_locus, grouped_loci)

    if min_sample_coverage or min_num_samples:
        loci = _filter_samples_by_coverage(loci, min_sample_coverage,
                                           filter_out_no_kmer_columns,
                                           min_num_samples=min_num_samples)

    return {'loci': loci, 'references': references, 'samples': samples}


def _parse_bam(samfile, read_group_samples, kmer_size,
               reference_index, filter_out_no_kmer_columns=True):

    for pileup_column in samfile.pileup():
        ref_seq_name = pileup_column.reference_name
        position = pileup_column.pos
        coverage = pileup_column.n

        kmer_counts = defaultdict(Counter)
        at_least_one_kmer = False
        for pileup_read in pileup_column.pileups:
            if not pileup_read.is_del and not pileup_read.is_refskip:
                start = pileup_read.query_position
                end = start + kmer_size
                kmer = pileup_read.alignment.query_sequence[start:end]

                if len(kmer) < kmer_size:
                    continue

                try:
                    read_group = pileup_read.alignment.get_tag('RG')
                except KeyError:
                    read_group = None
                sample = read_group_samples[read_group]

                kmer_counts[sample][kmer] += 1
                at_least_one_kmer = True

        if filter_out_no_kmer_columns and not at_least_one_kmer:
            continue

        yield {'location': (reference_index[ref_seq_name], position),
               'coverage': coverage,
               'kmer_counts': kmer_counts}


def _allele_distance(allele1, allele2):
    return sum((letter1 != letter2 for letter1, letter2 in zip(allele1, allele2)))


def _fix_sequencing_errors(allele_counts, possible_errors, edit_distance_fix,
                           template_alleles_for_fix):
    allele_counts = dict(allele_counts)
    for kmer, seq_error_count in possible_errors:
        close_alleles = [_allele_distance(kmer, allele) <= edit_distance_fix
                         for allele in template_alleles_for_fix]
        num_close_alleles = sum(close_alleles)
        if not num_close_alleles:
            # no candidate to be the origin of the sequencing error
            continue
        if num_close_alleles > 1:
            # more than one possible allele could have created the error
            continue

        original_allele_with_no_seq_error_index = close_alleles.index(True)
        original_allele_with_no_seq_error = template_alleles_for_fix[original_allele_with_no_seq_error_index]

        original_allele_count = allele_counts.get(original_allele_with_no_seq_error,
                                                  0)
        fixed_count = original_allele_count + seq_error_count

        allele_counts[original_allele_with_no_seq_error] = fixed_count

    return allele_counts


def _chi_squared_yates(allele_counts):
    counts = tuple(allele_counts.values())
    total_counts = sum(counts)
    expected = total_counts / 2
    chi = sum((abs(count - expected) - 0.5) ** 2 / expected for count in counts)
    p_value = chi2.sf(chi, 1)
    return p_value


def _get_abundant_enough_alleles_for_error_correction(locus,
                                                      min_coverage_to_be_template_for_fix):
    allele_total_counts = Counter()
    for sample_counts in locus['kmer_counts'].values():
        for allele, counts in sample_counts.items():
            allele_total_counts[allele] += counts
    template_alleles_for_fix = [(allele, counts) for allele, counts in allele_total_counts.items()
                                if counts >= min_coverage_to_be_template_for_fix]
    return template_alleles_for_fix


def _infer_alleles_for_locus(locus, ploidy, edit_distance_fix,
                             min_coverage_to_be_template_for_fix,
                             low_freq_allele_filter_pvalue):
    template_alleles_for_fix = _get_abundant_enough_alleles_for_error_correction(locus,
                                                                                 min_coverage_to_be_template_for_fix)
    template_alleles_for_fix = [allele for allele, _ in template_alleles_for_fix]
    sample_alleles = {}
    for sample, kmer_counts in locus['kmer_counts'].items():
        kmer_counts = sorted(kmer_counts.items(),
                             key=operator.itemgetter(1),
                             reverse=True)
        allele_counts = kmer_counts[:ploidy]
        possible_errors = kmer_counts[ploidy:]

        if edit_distance_fix and possible_errors:
            allele_counts = _fix_sequencing_errors(allele_counts,
                                                   possible_errors,
                                                   edit_distance_fix,
                                                   template_alleles_for_fix)

        allele_counts = OrderedDict(allele_counts)

        if low_freq_allele_filter_pvalue and len(allele_counts) > 1:
            if ploidy != 2:
                msg = 'Low probability allele filter not implemented for ploidy other than 2'
                raise NotImplementedError(msg)
            p_value = _chi_squared_yates(allele_counts)
            if p_value < low_freq_allele_filter_pvalue:
                # We keep just the most abundant allele
                allele_counts = dict([next(iter(allele_counts.items()))])

        sample_alleles[sample] = allele_counts
    locus['alleles'] = sample_alleles


def _infer_alleles(loci, ploidy, edit_distance_fix=1,
                   min_coverage_to_be_template_for_fix=3,
                   low_freq_allele_filter_pvalue=0.05):
    for locus in loci:
        _infer_alleles_for_locus(locus, ploidy=ploidy,
                                 edit_distance_fix=edit_distance_fix,
                                 min_coverage_to_be_template_for_fix=min_coverage_to_be_template_for_fix,
                                 low_freq_allele_filter_pvalue=low_freq_allele_filter_pvalue)
        yield locus


def _to_byte(str_or_none):

    try:
        return str_or_none.decode()
    except AttributeError:
        return b'None'


class BAMParser():
    def __init__(self, bam_fpaths, kmer_size, ploidy,
                 min_sample_coverage=0, edit_distance_fix=1,
                 min_coverage_to_be_template_for_fix=3,
                 low_freq_allele_filter_pvalue=0.05,
                 min_num_samples=0, max_field_lens=None,
                 max_field_str_lens=None):
        self.ploidy = ploidy
        self.kmer_size = kmer_size

        result = _parse_bams(bam_fpaths=bam_fpaths, kmer_size=kmer_size,
                             filter_out_no_kmer_columns=True,
                             min_sample_coverage=min_sample_coverage,
                             min_num_samples=min_num_samples)
        loci = _infer_alleles(result['loci'], ploidy=ploidy,
                              edit_distance_fix=edit_distance_fix,
                              min_coverage_to_be_template_for_fix=min_coverage_to_be_template_for_fix,
                              low_freq_allele_filter_pvalue=low_freq_allele_filter_pvalue)

        self._loci = loci
        self._references = result['references']
        self._samples = list(sorted(result['samples'],
                                    key=lambda x: x if x else ''))
        self.samples = list(map(_to_byte, self._samples))

        if max_field_lens is None:
            self.max_field_lens = {'alt': 0}
        else:
            self.max_field_lens = max_field_lens
        if max_field_str_lens is None:
            self.max_field_str_lens = {'chrom': 0}
        else:
            self.max_field_str_lens = max_field_str_lens
        self.max_field_str_lens['alt'] = kmer_size
        self.max_field_str_lens['ref'] = kmer_size

        self.metadata = copy.deepcopy(DEF_METADATA)
        self.metadata['CALLS'][b'AD'] = {'Description': 'kmer count',
                                         'dtype': 'int',
                                         'Number': 'R'}

    @property
    def variations(self):
        ploidy = self.ploidy
        if ploidy != 2:
            msg = 'Genotypes not inferred for ploidy other than 2'
            raise NotImplementedError(msg)

        samples = self._samples
        references = {idx: ref for idx, ref in enumerate(self._references)}
        for snp_id, locus in enumerate(self._loci):

            chrom, pos = locus['location']
            chrom = references[chrom]

            # create genotypes
            tot_counts = Counter()
            for kmer_counts in locus['alleles'].values():
                tot_counts += kmer_counts
            alleles = [allele_count[0] for allele_count in tot_counts.most_common()]
            allele_coding = dict({(allele, idx) for idx, allele in enumerate(alleles)})

            gts = []
            allele_depths = []
            locus_kmer_counts = locus['alleles']
            for sample in samples:
                sample_kmer_counts = locus_kmer_counts.get(sample, {None: 0})
                sample_gts = []
                sample_allele_depth = [0] * len(alleles)
                for allele, allele_counts in sample_kmer_counts.items():
                    allele_index = allele_coding.get(allele, MISSING_INT)
                    sample_gts.append(allele_index)
                    if allele_index >= 0:
                        sample_allele_depth[allele_index] = allele_counts
                if len(sample_gts) == 1:
                    # this won't work with ploidy different than 2
                    sample_gts = sample_gts * ploidy
                gts.append(sample_gts)
                allele_depths.append(sample_allele_depth)

            gts = [(b'GT', gts), (b'AD', allele_depths)]

            variation = (chrom, pos, snp_id,
                         alleles[0], alleles[1:], None, None, None, gts)
            yield variation
