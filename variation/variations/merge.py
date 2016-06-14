# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111
from collections import Counter
import copy

import numpy

from variation import MISSING_VALUES, MISSING_BYTE, DEF_METADATA
from variation.iterutils import PeekableIterator
from variation import (CHROM_FIELD, POS_FIELD, REF_FIELD, ALT_FIELD,
                       QUAL_FIELD, GT_FIELD, DP_FIELD)
from variation.matrix.methods import is_dataset


def _iterate_vars(variations):
    kept_fields = [CHROM_FIELD, POS_FIELD]

    optional_fields = [REF_FIELD, ALT_FIELD, GT_FIELD, QUAL_FIELD, DP_FIELD]
    for field in optional_fields:
        if field in variations.keys():
            kept_fields.append(field)

    for chunk in variations.iterate_chunks(kept_fields=kept_fields):
        chunk_keys = chunk.keys()
        vars_chrom = chunk[CHROM_FIELD]
        vars_pos = chunk[POS_FIELD]
        vars_ref = chunk[REF_FIELD] if REF_FIELD in chunk_keys else None
        vars_alt = chunk[ALT_FIELD] if ALT_FIELD in chunk_keys else None
        vars_qual = chunk[QUAL_FIELD] if QUAL_FIELD in chunk_keys else None
        vars_dp = chunk[DP_FIELD] if DP_FIELD in chunk_keys else None
        vars_gts = chunk[GT_FIELD] if GT_FIELD in chunk_keys else None

        if is_dataset(vars_chrom):
            vars_chrom = vars_chrom[:]
            vars_pos = vars_pos[:]
            if vars_ref is not None:
                vars_ref = vars_ref[:]
            if vars_alt is not None:
                vars_alt = vars_alt[:]
            if vars_qual is not None:
                vars_qual = vars_qual[:]
            if vars_dp is not None:
                vars_dp = vars_dp[:]
            if vars_gts is not None:
                vars_gts = vars_gts[:]

        for var_idx in range(chunk.num_variations):
            chrom = vars_chrom[var_idx]
            pos = vars_pos[var_idx]
            ref = None if vars_ref is None else vars_ref[var_idx]
            if vars_alt is None:
                alts = None
            else:
                alts = vars_alt[var_idx]
                alts = [alt for alt in alts if alt != MISSING_BYTE]
                if not alts:
                    alts = None
            qual = None if vars_qual is None else vars_qual[var_idx]
            gts = None if vars_gts is None else vars_gts[var_idx]

            var_ = {'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alts,
                    'qual': qual, 'gts': gts}
            if vars_dp is not None:
                var_['dp'] = vars_dp[var_idx]
            yield var_


def _are_overlapping(var1, var2):
    pos1 = var1['pos']
    pos2 = var2['pos']
    return (var1['chrom'] == var2['chrom'] and
            pos2 >= pos1 and pos2 < pos1 + len(var1['ref']))


def _pos_lt_tuples(tup1, tup2):
    chrom1 = tup1[0]
    chrom2 = tup2[0]

    if chrom1 < chrom2:
        return True
    elif chrom1 > chrom2:
        return False
    pos1 = tup1[1]
    pos2 = tup2[1]

    return True if pos1 < pos2 else False


def _pos_le_tuples(tup1, tup2):
    chrom1 = tup1[0]
    chrom2 = tup2[0]

    if chrom1 < chrom2:
        return True
    elif chrom1 > chrom2:
        return False
    pos1 = tup1[1]
    pos2 = tup2[1]

    return True if pos1 <= pos2 else False


def _pos_lt(snp1, snp2):
    tup1 = snp1['chrom'], snp1['pos']
    tup2 = snp2['chrom'], snp2['pos']
    return _pos_lt_tuples(tup1, tup2)


def _sort_iterators(iter1, iter2):
    prev_item1 = None
    prev_item2 = None
    while True:
        if prev_item1:
            item1 = prev_item1
            prev_item1 = None
        else:
            try:
                item1 = iter1.peek()
            except StopIteration:
                item1 = None
        if prev_item2:
            item2 = prev_item2
            prev_item2 = None
        else:
            try:
                item2 = iter2.peek()
            except StopIteration:
                item2 = None
        if item1 is None and item2 is None:
            break
        if item1 is None:
            yield item2
        elif item2 is None:
            yield item1
        else:
            if _pos_lt(item1, item2):
                prev_item2 = item2
                yield item1
            else:
                prev_item1 = item1
                yield item2


def _var_len(var):
    if var['ref'] is None:
        return 1
    max_len = len(var['ref'])
    if not var['alt']:
        return max_len

    alt_len = max([len(alt_allele) for alt_allele in var['alt']])

    if alt_len > max_len:
        max_len = alt_len
    return max_len


def _get_overlapping_region(sorted_vars):
    region_start = None
    region_stop = None
    for var in sorted_vars:
        var_stop = var['chrom'], var['pos'] + _var_len(var) - 1
        var_start = var['chrom'], var['pos']
        if region_start is None:
            region_start = var_start
        if region_stop is None:
            region_stop = var_stop
        elif _pos_le_tuples(var_start, region_stop):
            if _pos_lt_tuples(region_stop, var_stop):
                region_stop = var_stop
        else:
            return (region_start, region_stop)
    else:
        return (region_start, region_stop)


def _get_vars_in_region(vars_, region):
    vars_in_region = []
    while True:
        try:
            var = vars_.peek()
        except StopIteration:
            return vars_in_region

        if _pos_le_tuples((var['chrom'], var['pos']), region[1]):
            vars_in_region.append(next(vars_))
        else:
            break
    return vars_in_region


def _group_overlaping_vars(variations_1, variations_2,
                           ignore_2_or_more_overlaps=False,
                           check_ref_match=True):

    if isinstance(variations_1, list) and isinstance(variations_2, list):
        # This is intented only for testing and debugin
        snps_1 = PeekableIterator(iter(variations_1))
        snps_2 = PeekableIterator(iter(variations_2))
    else:
        snps_1 = PeekableIterator(_iterate_vars(variations_1))
        snps_2 = PeekableIterator(_iterate_vars(variations_2))

    while True:
        sorted_vars = _sort_iterators(snps_1, snps_2)
        region = _get_overlapping_region(sorted_vars)
        snps_1.reset_peek()
        snps_2.reset_peek()

        snps1_in_region = _get_vars_in_region(snps_1, region)
        snps2_in_region = _get_vars_in_region(snps_2, region)
        snps_1.reset_peek()
        snps_2.reset_peek()
        if not snps1_in_region and not snps2_in_region:
            break
        yield (snps1_in_region, snps2_in_region)


class MalformedVariationError(Exception):
    pass


def _transform_alleles(base_allele, alleles, position, max_allele_length=35):
    new_alleles = []
    for allele in alleles:
        new_allele = bytearray(base_allele)
        if len(allele) == 1:
            try:
                new_allele[position] = allele[0]
            except IndexError as err:
                msg = 'alleles: ' + str(alleles)
                msg += '\nallele: ' + str(allele)
                msg += '\nposition: ' + str(position)
                err.args = (msg,)
                raise MalformedVariationError(msg)
        else:
            # TODO. This will create a non std VCF variation if we're
            # merging with a SNP (not indel)
            new_allele[position: position + 1] = bytearray(allele)
        new_alleles.append(bytes(new_allele))
    return new_alleles


def _transform_gts_to_merge(long_alleles, new_short_alleles, gts):
    alleles_merged = long_alleles[:]

    new_gts = gts.copy()
    for old_short_idx, short_allele in enumerate(new_short_alleles):
        try:
            new_allele_idx = long_alleles.index(short_allele)
        except:
            alleles_merged.append(short_allele)
            new_allele_idx = len(alleles_merged) - 1
        if old_short_idx != new_allele_idx:
            new_gts[gts == old_short_idx] = new_allele_idx

    return alleles_merged, new_gts


class VarMerger():
    def __init__(self, variations1, variations2, suffix_for_sample2=None,
                 ignore_complex_overlaps=False, ignore_malformed_vars=False,
                 allow_non_std_var_creation=False, check_ref_matches=True,
                 max_field_lens=None, ignore_non_matching=False,
                 merge_only_snps=False):
        '''It merges two variation matrices.

        suffix for sample2 is only added to samples in variations2 also
        found in variations1'''
        self.variations1 = variations1
        self.variations2 = variations2
        self.log = Counter()
        self.samples = self._get_samples(suffix_for_sample2)
        self._ignore_complex_overlaps = ignore_complex_overlaps
        self._ignore_malformed_vars = ignore_malformed_vars
        self._ignore_non_matching = ignore_non_matching
        self._allow_non_std_var_creation = allow_non_std_var_creation
        self._merge_only_snps = merge_only_snps
        self._check_ref_matches = check_ref_matches
        self._gt_shape = None
        self._gt_dtype = None
        self._n_samples1 = len(variations1.samples)
        self._n_samples2 = len(variations2.samples)
        if max_field_lens is None:
            self.max_field_lens = {'alt': 0}
        else:
            self.max_field_lens = max_field_lens

        self.max_field_str_lens = self._get_max_field_str_lens()

        if variations1.ploidy != variations2.ploidy:
            raise ValueError('Ploidies should match')
        self.ploidy = variations1.ploidy
        metadata = copy.deepcopy(DEF_METADATA)
        if DP_FIELD in variations1.keys() and DP_FIELD in variations2.keys():
            metadata['CALLS'][b'DP'] = {'Description': 'Depth', 'dtype': 'int'}
        self.metadata = metadata
        self.ignored_fields = []
        self.kept_fields = []

    def _get_max_field_str_lens(self):
        max_lens = {}
        fields = ['/variations/chrom', '/variations/ref',
                  '/variations/alt']
        for field in fields:
            field_siz1 = int(str(self.variations1[field].dtype).split('S')[-1])
            field_siz2 = int(str(self.variations2[field].dtype).split('S')[-1])
            field_size = max([field_siz1, field_siz2])
            field_name = field.split('/')[-1]
            max_lens[field_name] = field_size
        return max_lens

    def _get_samples(self, suffix):

        if suffix is None:
            samples = self.variations1.samples + self.variations2.samples
            return [sample.encode('utf-8') for sample in samples]

        samples = self.variations1.samples[:]
        for sample in self.variations2.samples:
            if sample in self.variations1.samples:
                sample = sample + suffix
            samples.append(sample)
        return [sample.encode('utf-8') for sample in samples]

    @staticmethod
    def _build_error_msg(snp1, snp2, error):
        try:
            chrom1 = snp1['chrom']
        except Exception:
            chrom1 = ''
        try:
            chrom2 = snp2['chrom']
        except Exception:
            chrom2 = ''
        try:
            pos1 = snp1['pos']
        except Exception:
            pos1 = ''
        try:
            pos2 = snp2['pos']
        except Exception:
            pos2 = ''
        msg = 'chrom1: ' + str(chrom1) + ' pos1: ' + str(pos1)
        msg += ' chrom2: ' + str(chrom2) + ' pos2: ' + str(pos2)
        template = "\nAn exception of type {0} occured. Arguments:\n{1!r}"
        msg += template.format(type(error).__name__, error.args)
        return msg

    @property
    def variations(self):
        ignore_non_matching = self._ignore_non_matching
        for snps1, snps2 in _group_overlaping_vars(self.variations1,
                                                   self.variations2):

            if self._snps_are_mergeable(snps1, snps2):
                snp1 = snps1[0] if snps1 else None
                snp2 = snps2[0] if snps2 else None
                if ignore_non_matching and (snp1 is None or snp2 is None):
                    continue
                try:
                    var = self._merge_vars(snp1, snp2)
                except MalformedVariationError:
                    if self._ignore_malformed_vars:
                        continue
                except Exception as error:
                    msg = self._build_error_msg(snp1, snp2, error)
                    error.args = (msg,)
                    raise
                calls = [(b'GT', var['gts'])]
                depth = var.get('dp', None)
                if depth is not None:
                    calls.append((b'DP', depth))
                variation = (var['chrom'], var['pos'], None, var['ref'],
                             var['alt'], var['qual'], [], {}, calls)
                yield variation
            else:
                if not self._ignore_complex_overlaps:
                    poss1 = [(snp['chrom'], str(snp['pos'])) for snp in snps1]
                    poss2 = [(snp['chrom'], str(snp['pos'])) for snp in snps2]
                    msg = 'We can not merge these vars:\n'
                    msg += '{}\n{}\n'
                    raise NotImplementedError(msg.format(poss1, poss2))

    def _len_longer_allele(self, snps):
        alleles = []
        for snp in snps:
            alleles.append(snp['ref'])
            try:
                alleles.extend(snp['alt'])
            except TypeError:
                pass
        if alleles:
            return max(map(len, alleles))
        else:
            return 0

    def _snps_are_mergeable(self, snps1, snps2):
        "it looks only to the conditions we have programmed"
        len_snps1 = len(snps1)
        len_snps2 = len(snps2)

        result = None
        if len_snps1 > 1 or len_snps2 > 1:
            self.log['Too_many_overlaping_vars'] += 1
            result = False

        elif len_snps1 == 1 or len_snps1 == 1:
            if _var_len(snps1[0]) > 1 and _var_len(snps2[0]) > 1:
                self.log['overlaping_complex'] += 1
                result = False
            else:
                result = True
        elif ((len_snps1 == 1 and len_snps2 == 0) or
                (len_snps1 == 0 and len_snps2 == 1)):
            result = True
        if result is None:
            raise RuntimeError("We shouldn't be here")
        if not result:
            return result

        if self._merge_only_snps:
            if self._len_longer_allele(snps1) > 1:
                return False
            if self._len_longer_allele(snps2) > 1:
                return False
        return result

    def _get_alleles(self, snp):
        alleles = [snp['ref']]
        if snp['alt'] is not None:
            alleles.extend([allele for allele in snp['alt'] if allele != b''])
        return alleles

    def _get_qual(self, snp1, snp2):
        qual1 = snp1.get('qual', None)
        qual2 = snp2.get('qual', None)
        if qual1 is None or qual2 is None:
            return None
        else:
            return min([qual1, qual2])

    def _merge_depth(self, snp1, snp2):
        dp1 = snp1.get('dp', None)
        dp2 = snp2.get('dp', None)
        if dp1 is None or dp2 is None:
            return None
        else:
            return numpy.append(dp1, dp2, axis=0)

    def _snp_info_msg(self, short_snp, long_snp, position):
        msg = 'short_snp["pos"]: ' + str(short_snp['pos'])
        msg += '\nshort_snp["ref"]: ' + str(short_snp['ref'])
        msg += '\nshort_alleles: ' + str(self._get_alleles(short_snp))
        msg += '\nlong_snp["pos"]: ' + str(long_snp['pos'])
        msg += '\nlong_snp["ref"]: ' + str(long_snp['ref'])
        msg += '\nlong_alleles: ' + str(self._get_alleles(long_snp))
        msg += '\nposition: ' + str(position)
        return msg

    def _merge_vars(self, snp1, snp2):
        "it assumes that the given snps are overlaping or None"
        if self._gt_shape is None:
            snp = snp1 if snp1 is not None else snp2
            self._gt_shape = (len(self.samples), self.ploidy)
            self._gt_dtype = snp['gts'].dtype

        merged_gts = numpy.full(self._gt_shape, MISSING_VALUES[self._gt_dtype],
                                self._gt_dtype)
        if snp1 is None or snp2 is None:
            if snp1 is None:
                good_snp = snp2
                merged_gts[self._n_samples1:] = snp2['gts']
            else:
                good_snp = snp1
                merged_gts[:self._n_samples1] = snp1['gts']
            new_snp = good_snp.copy()
            new_snp['gts'] = merged_gts
            num_alt = 0 if good_snp['alt'] is None else len(good_snp['alt'])
            if num_alt > self.max_field_lens['alt']:
                self.max_field_lens['alt'] = num_alt
            return new_snp

        # long snp means it is the longest reference, so it has to be
        # the base to build the new snps
        if len(snp1['ref']) >= len(snp2['ref']):
            long_snp = snp1
            short_snp = snp2
            vars1_first = True
        else:
            long_snp = snp2
            short_snp = snp1
            vars1_first = False

        short_alleles = self._get_alleles(short_snp)
        position = short_snp['pos'] - long_snp['pos']
        if position < 0:
            msg = 'The SNP looks malformed: \n'
            msg += self._snp_info_msg(short_snp, long_snp, position)
            raise MalformedVariationError(msg)
        try:
            new_short_alleles = _transform_alleles(long_snp['ref'],
                                                   short_alleles,
                                                   position)
        except Exception as error:
            msg = self._snp_info_msg(short_snp, long_snp, position)
            template = "\nAn exception of type {0} occured. Arguments:\n{1!r}"
            msg += template.format(type(error).__name__, error.args)
            error.args = (msg,)
            raise
        new_short_ref = new_short_alleles[0]

        if new_short_ref != long_snp['ref'] and self._check_ref_matches:
            raise ValueError('Reference alleles do not match')
        long_alleles = self._get_alleles(long_snp)

        res = _transform_gts_to_merge(long_alleles,
                                      new_short_alleles,
                                      short_snp['gts'])
        alleles_merged, new_short_gts = res
        num_alt = len(alleles_merged) - 1
        if num_alt > self.max_field_lens['alt']:
            self.max_field_lens['alt'] = num_alt

        if vars1_first:
            merged_gts = numpy.append(snp1['gts'], new_short_gts, axis=0)
            merged_dp = self._merge_depth(snp1, snp2)
        else:
            merged_gts = numpy.append(new_short_gts, snp2['gts'], axis=0)
            merged_dp = self._merge_depth(snp2, snp1)
        qual = self._get_qual(snp1, snp2)
        alt = alleles_merged[1:]
        if not alt:
            alt = None
        var_ = {'chrom': long_snp['chrom'], 'pos': long_snp['pos'],
                'ref': alleles_merged[0], 'alt': alt, 'gts': merged_gts,
                'qual': qual}
        if merged_dp is not None:
            var_['dp'] = merged_dp
        return var_
