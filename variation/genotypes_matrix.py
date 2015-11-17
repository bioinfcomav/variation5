# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111
from csv import DictReader
from itertools import chain
from variation import MISSING_VALUES, DEF_DSET_PARAMS
from variation.variations.filters import filter_gt_no_data
import numpy
from variation.variations.vars_matrices import VariationsH5
from variation.variations.vars_matrices import _create_matrix


IUPAC = {'A': 'AA', 'T': 'TT', 'C': 'CC', 'G': 'GG',
         'W': 'AT', 'M': 'AC', 'R': 'AG', '': '--',
         'Y': 'TC', 'K': 'TG', 'S': 'CG', '-': '--'}
IUPAC_CODING = 'iupac'
STANDARD_GT = 'standard'
DECODE = {IUPAC_CODING: lambda x: IUPAC[x],
          STANDARD_GT: lambda x: x}


class GenotypesMatrixParser():
    def __init__(self, fhand, gt_coding, max_alt_allele, metadata_fhand=None,
                 sep=',', id_fieldnames=None, ref_field='ref', alt_field='alt',
                 snp_fieldnames=['id', 'chrom', 'pos'], ignore_alt=False):
        self.fhand = fhand
        self.ploidy = 2
        self.ignore_alt = ignore_alt
        self.ref_field = ref_field
        self.alt_field = alt_field
        self.are_ref_alt_fieldnames = True
        if id_fieldnames is None and metadata_fhand is not None:
            raise ValueError('id_fieldnames is required when metadata is set')
        self.id_fieldnames = id_fieldnames
        self.metadata_fhand = metadata_fhand
        self.sep = sep
        if gt_coding not in [IUPAC_CODING, STANDARD_GT]:
            raise ValueError('Genotype coding not supported')
        self.gt_coding = gt_coding
        self.samples = None
        self.max_alt_allele = max_alt_allele
        self.snp_fieldnames = snp_fieldnames
        self.snp_fieldnames_final = snp_fieldnames.copy()
        if self.ref_field not in self.snp_fieldnames:
            self.snp_fieldnames_final.extend([self.ref_field, self.alt_field])
            self.are_ref_alt_fieldnames = False
        self._create_reader()

    def _merge_dicts(self, dict1, dict2, id_fieldnames=['id', 'id']):
        if dict1[id_fieldnames[0]] != dict2[id_fieldnames[1]]:
            raise ValueError('Ids in file do not match')
        else:
            return {key: value for key, value in chain(dict1.items(),
                                                       dict2.items())}

    def _parse_record(self, record):
        parsed_record = {'gts': []}
        gts = []
        alleles = set()
        for key in self.snp_fieldnames:
            parsed_record[key] = record[key]
        for sample in self.samples:
            gt = DECODE[self.gt_coding](record[sample])
            if gt == '':
                gt = '--'
            if len(gt) != 2:
                raise ValueError('Wrong gt coding given')
            gts.append(gt)
            alleles.add(gt[0])
            alleles.add(gt[1])
        alleles = sorted(list(alleles))
        if '-' in alleles and len(alleles) > 1:
            alleles.remove('-')
        if self.are_ref_alt_fieldnames is False:
            ref = alleles.pop()
            alt = alleles
        else:
            ref = record[self.ref_field]
            alleles.remove(ref)
            alt = alleles
        parsed_record['ref'] = ref
        parsed_record['alt'] = alt
        for gt in gts:
            gt_new = [0, 0]
            if gt[0] == '-':
                gt_new = [MISSING_VALUES[int], MISSING_VALUES[int]]
            else:
                for i, alt_allele in enumerate(alt):
                    if alt_allele == gt[0]:
                        gt_new[0] = i + 1
                    if alt_allele == gt[1]:
                        gt_new[1] = i + 1
            parsed_record['gts'].append(gt_new)

        if not parsed_record['alt']:
            parsed_record['alt'] = [MISSING_VALUES[str]]
        try:
            parsed_record['pos'] = int(parsed_record['pos'])
        except KeyError:
            pass
        return parsed_record

    def _create_reader(self):
        reader = DictReader(self.fhand, delimiter=self.sep)
        self.samples = [f for f in reader.fieldnames
                        if f not in self.snp_fieldnames]
        if self.metadata_fhand is not None:
            reader = (self._merge_dicts(meta, record,
                                        id_fieldnames=self.id_fieldnames)
                      for meta, record in zip(DictReader(self.metadata_fhand,
                                                         delimiter=self.sep),
                                              reader))
        self.reader = reader

    def __iter__(self):
        for record in self.reader:
            record = self._parse_record(record)
            if len(record['alt']) <= self.max_alt_allele:
                yield record


def count_compatible_snps_in_chains(variations, chains_matrix,
                                    n_snps_check=10000):
    variations = filter_gt_no_data(variations, n_snps_check)
    ref = variations['/variations/ref'][:]
    alt = variations['/variations/alt'][:]
    alleles = numpy.append(ref.reshape(ref.shape[0], 1), alt, axis=1)
    chain_counts = numpy.zeros(chains_matrix.shape[1])
    for snp_alleles, chains_snp_alleles in zip(alleles, chains_matrix):
        for i, chain_alleles in enumerate(chains_snp_alleles):
            compatible = True
            for allele in snp_alleles:
                if allele not in chain_alleles and allele != b'':
                    compatible = False
                    break
            if compatible:
                chain_counts[i] += 1
    return chain_counts


def _rev_compl(seq):
    complementary = {b'A': b'T', b'G': b'C', b'T': b'A', b'C': b'G', b'': b''}
    return numpy.array([complementary[x] for x in seq])


def change_gts_chain(variations, mask):
    ref = variations['/variations/ref'][:]
    alt = variations['/variations/alt'][:]
    alleles = numpy.append(ref.reshape(ref.shape[0], 1), alt, axis=1)
    alleles[mask] = numpy.apply_along_axis(_rev_compl, 0, alleles[mask])
    return alleles


def merge_matrices(mat1, mat2, rownames1, rownames2, colnames1, colnames2,
                   dtype='int'):
    pass


def _max_length(chars_array):
    return max([len(x) for x in chars_array])


def decode_gts(alleles, gts):
    decoded = gts.copy().astype('str')
    for i, (snp_gts, snp_alleles) in enumerate(zip(gts, alleles)):
        decoded[i] = snp_alleles[snp_gts]
    return decoded


def encode_gts(gts, alleles):
    encoded = numpy.zeros(gts.shape)
    i = 0
    for j in range(alleles.shape[1]):
        allele = alleles[:, j].reshape(alleles.shape[0], 1).T
        if allele != b'':
            encoded[gts == allele] = i
            i += 1
        else:
            encoded[gts == allele] = MISSING_VALUES[int]
    return encoded


def collapse_gts(base_allele, gts, positions, max_allele_length=35):
    # Type pass to str because join expected str not int(binary)
    collapsed_gts = gts[0].copy().reshape(1, gts.shape[1], gts.shape[2])
    collapsed_gts = collapsed_gts.astype(numpy.dtype(('U', max_allele_length)))
    for i in range(gts.shape[1]):
        for j in range(gts.shape[2]):
            allele = [x for x in base_allele.decode('utf-8')]
            for k, pos in enumerate(positions):
                allele[pos] = gts[k, i, j]
            collapsed_gts[0, i, j] = ''.join(allele)
    return collapsed_gts


def merge_alleles(alleles_vcf, alleles_collapse):
    for allele_collapse in alleles_collapse:
        if allele_collapse in alleles_vcf:
            continue
        alleles_merged = numpy.append(alleles_vcf, allele_collapse)
    return alleles_merged


def get_attribute(h5_1, h5_2, path, attribute):
    try:
        return getattr(h5_1[path], attribute)
    except KeyError:
        return getattr(h5_2[path], attribute)


def create_variations_merged(variations_1, variations_2,
                             fpath, ignore_fields=[]):
    variations_merged = VariationsH5(fpath, "w")
    h5_paths = set(variations_1.keys()+variations_2.keys())
    h5_samples = variations_1.samples+variations_2.samples
    for path in h5_paths:
        continue_ = False
        for field in ignore_fields:
            if field in path:
                continue_ = True
        if continue_:
            continue
        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['shape'] = list(get_attribute(variations_1, variations_2,
                                             path, 'shape'))
        kwargs['shape'][0] = (variations_1['/variations/pos'].shape[0] +
                              variations_2['/variations/pos'].shape[0])
        if 'calls' in path:
            kwargs['shape'][1] = len(h5_samples)
        kwargs['shape'] = tuple(kwargs['shape'])
        kwargs['dtype'] = get_attribute(variations_1, variations_2,
                                        path, 'dtype')
        kwargs['maxshape'] = get_attribute(variations_1, variations_2,
                                           path, 'maxshape')
        if 'variations' in path and 'chrom' not in path and 'pos' not in path and 'ref' not in path:
            kwargs['maxshape'] = (None, None)
            ncols = 0
            try:
                ncols += variations_1[path].shape[1]
            except (KeyError, IndexError):
                ncols += 1
            try:
                ncols += variations_2[path].shape[1]
            except (KeyError, IndexError):
                ncols += 1
            kwargs['shape'] = (kwargs['shape'][0], ncols)
        _create_matrix(variations_merged, path, **kwargs)
    return variations_merged


def are_overlapping(var1, var2):
    pos1 = var1['/variations/pos']
    pos2 = var2['/variations/pos']
    return (var1['/variations/chrom'] == var2['/variations/chrom'] and
            pos2 >= pos1 and pos2 < pos1 + len(var1['/variations/ref'][0]))


def merge_snps(snp1, snp2, i, merged, fields_funct={}, ignore_fields=[],
               check_ref_match=True):
    is_merged = 0
    is_added = 0
    is_ignored = 0
    if snp1 is not None and snp2 is not None and 'ref' in snp1.keys() and 'ref' in snp2.keys():
        if snp1['/variations/ref'] != snp2['/variations/ref']:
            if check_ref_match:
                raise 'The reference allele of the SNPs are not the same.'
            else:
                # The snps have been ignored
                is_ignored = 1
                return is_ignored, is_merged, is_added

    if '/variations/qual' not in fields_funct:
        fields_funct['/variations/qual'] = min
    for path in merged.keys():
        continue_ = False
        for field in ignore_fields:
            if field in path:
                continue_ = True
        if continue_:
            continue
        if 'calls' in path:
            if 'GT' in path:
                if snp1 is not None:
                    gts1 = snp1['/calls/GT']
                    pos1 = snp1['/variations/pos']
                    ref1 = snp1['/variations/ref']
                    alt1 = snp1['/variations/alt']
                    alleles1 = numpy.append(ref1.reshape(ref1.shape[0], 1),
                                            alt1, axis=1)
                    chrom1 = snp1['/variations/chrom']

                if snp2 is not None:
                    gts2 = snp2['/calls/GT']
                    pos2 = snp2['/variations/pos']
                    ref2 = snp2['/variations/ref']
                    alt2 = snp2['/variations/alt']
                    alleles2 = numpy.append(ref2.reshape(ref2.shape[0], 1),
                                            alt2, axis=1)
                    chrom2 = snp2['/variations/chrom']

                if snp1 is None:
                    merged['/variations/ref'][i] = ref2
                    merged['/variations/alt'][i, :alt2.shape[1]] = alt2[0]
                    merged['/variations/pos'][i] = pos2
                    merged[path][i, -len(snp2.samples):, ] = gts2
                    merged['/variations/chrom'][i] = chrom2
                    continue
                if snp2 is None:
                    merged['/variations/ref'][i] = ref1
                    merged['/variations/alt'][i, :alt1.shape[1]] = alt1[0]
                    merged['/variations/pos'][i] = pos1
                    merged[path][i, :len(snp1.samples), ] = gts1
                    merged['/variations/chrom'][i] = chrom1
                    continue

                if chrom1 != chrom2:
                    raise ValueError('Chromosome names must be equal')
                merged['/variations/chrom'][i] = chrom1
                if pos1 <= pos2:
                    gts_decoded = decode_gts(alleles2, gts2)
                    gts_collapsed = collapse_gts(alleles1[0, 0], gts_decoded,
                                                 pos2 - pos1)
                    alleles = numpy.array([x.decode('utf-8')
                                           for x in alleles1[0]])
                    alleles_merged = merge_alleles(alleles,
                                                   numpy.unique(gts_collapsed))
                    gts_encoded = encode_gts(gts_collapsed,
                                             alleles_merged.reshape(1, alleles_merged.shape[0]))
                    alleles_merged = alleles_merged.astype(merged['/variations/ref'].dtype)
                    merged['/variations/ref'][i] = alleles_merged[0]
                    merged['/variations/alt'][i, :alleles_merged.shape[0]-1] = alleles_merged[1:]
                    merged['/variations/pos'][i] = pos1
                    merged[path][i, :len(snp1.samples), ] = snp1[path][:]
                    merged[path][i, -len(snp2.samples):, ] = gts_encoded

                else:
                    gts_decoded = decode_gts(alleles1, gts1)
                    gts_collapsed = collapse_gts(alleles2[0, 0], gts_decoded,
                                                 pos1 - pos2)
                    alleles = numpy.array([x.decode('utf-8')
                                           for x in alleles2[0]])
                    alleles_merged = merge_alleles(alleles,
                                                   numpy.unique(gts_collapsed))
                    gts_encoded = encode_gts(gts_collapsed,
                                             alleles_merged.reshape(1, alleles_merged.shape[0]))
                    alleles_merged = alleles_merged.astype(merged['/variations/ref'].dtype)
                    merged['/variations/ref'][i] = alleles_merged[0]
                    merged['/variations/alt'][i, :alleles_merged.shape[0]-1] = alleles_merged[1:]
                    merged['/variations/pos'][i] = pos2
                    merged[path][i, :len(snp1.samples), ] = gts_encoded
                    merged[path][i, -len(snp2.samples):, ] = snp2[path][:]

            elif snp1 is not None:
                try:
                    merged[path][i, :len(snp1.samples), ] = snp1[path][:]
                except KeyError:
                    pass
            elif snp2 is not None:
                try:
                    merged[path][i, -len(snp2.samples):, ] = snp2[path][:]
                except KeyError:
                    pass
        else:
            if 'ref' not in path and 'alt' not in path and 'pos' not in path and 'chrom' not in path:
                missing = numpy.array(MISSING_VALUES[merged[path].dtype])
                if path in fields_funct:
                    # if one snp is None and the other has not a path
                    try:
                        try:
                            try:
                                merged_values = fields_funct[path](snp1[path],
                                                                   snp2[path])
                                merged[path][i] = merged_values
                            except (KeyError, TypeError):
                                merged[path][i] = snp1[path]
                        except (KeyError, TypeError):
                            merged[path][i] = snp2[path]
                    except (KeyError, TypeError):
                        merged[path][i] = missing
                else:
                    try:
                        try:
                            try:
                                merged[path][i, ] = numpy.append(snp1[path],
                                                                 snp2[path])
                            except (KeyError, TypeError):
                                merged[path][i, ] = numpy.append(snp1[path],
                                                                 missing)
                        except (KeyError, TypeError):
                            merged[path][i, ] = numpy.append(missing,
                                                             snp2[path])
                    except (KeyError, TypeError):
                        merged[path][i, ] = numpy.full(merged[path][i, ].shape,
                                                       missing,
                                                       dtype=merged[path][i,].dtype)
    # The snps have been merged succesfully
    if snp1 is not None and snp2 is not None:
        is_merged = 1
        is_added = 0
        return is_ignored, is_merged, is_added
    else:
        is_merged = 0
        is_added = 1
        return is_ignored, is_merged, is_added


def merge_sorted_variations(variations_1, variations_2,
                            ignore_2_or_more_overlaps=False,
                            check_ref_match=True):
    snps_1 = iter(variations_1.iterate_chunks(chunk_size=1))
    snps_2 = iter(variations_2.iterate_chunks(chunk_size=1))
    snp_1 = next(snps_1)
    snp_2 = next(snps_2)
    stop = ""
    are_snps1 = False
    prev_chrom = None
    while True:
        pos1 = snp_1['/variations/pos']
        pos2 = snp_2['/variations/pos']

        if snp_1['/variations/chrom'] == snp_2['/variations/chrom']:
            if are_overlapping(snp_1, snp_2) or are_overlapping(snp_2, snp_1):
                result = snp_1, snp_2
                try:
                    snp_1 = next(snps_1)
                except StopIteration:
                    stop = snps_2
                    yield result
                    break
                try:
                    snp_2 = next(snps_2)
                except StopIteration:
                    stop = snps_1
                    are_snps1 = True
                    yield result
                    break
                overlap = False
                try:
                    while are_overlapping(result[0], snp_2):
                        snp_2 = next(snps_2)
                        overlap = True
                        if not ignore_2_or_more_overlaps:
                            msg = 'More than 2 variations are overlapping'
                            raise ValueError(msg)
                except StopIteration:
                    stop = snps_1
                    are_snps1 = True
                    break
                try:
                    while are_overlapping(result[1], snp_1):
                        snp_1 = next(snps_1)
                        overlap = True
                        if not ignore_2_or_more_overlaps:
                            msg = 'More than 2 variations are overlapping'
                            raise ValueError(msg)
                except StopIteration:
                    stop = snps_2
                    break
                if not overlap:
                    yield result
            elif pos1 < pos2:
                yield snp_1, None
                try:
                    snp_1 = next(snps_1)
                except StopIteration:
                    stop = snps_2
                    break
            elif pos2 < pos1:
                yield None, snp_2
                try:
                    snp_2 = next(snps_2)
                except StopIteration:
                    stop = snps_1
                    are_snps1 = True
                    break
            prev_chrom = snp_1['/variations/chrom']
        else:
            if snp_1['/variations/chrom'] == prev_chrom:
                yield snp_1, None
                try:
                    snp_1 = next(snps_1)
                except StopIteration:
                    stop = snps_2
                    break
            else:
                yield None, snp_2
                try:
                    snp_2 = next(snps_2)
                except StopIteration:
                    stop = snps_1
                    are_snps1 = True
                    break
    if are_snps1:
        yield snp_1, None
    for snp in stop:
        if are_snps1:
            yield snp, None
        else:
            yield None, snp


def merge_variations(variations1, variations2, merged_fpath, fields_funct={},
                     ignore_overlaps=False, ignore_2_or_more_overlaps=False,
                     ignore_fields=[], check_ref_match=True):
    merged_variations = create_variations_merged(variations1, variations2,
                                                 merged_fpath,
                                                 ignore_fields=ignore_fields)
    num_snp1 = variations1['/variations/pos'].shape[0]
    num_snp2 = variations2['/variations/pos'].shape[0]
    log = {'total_merged_snps': 0,
           'modified_merged_snps': 0,
           'added_new_snps': 0,
           'ignored_ref_snps': 0,
           'ignored_overlap_snps': 0}
    prev_snp1, prev_snp2 = None, None
    for i, (snp1, snp2) in enumerate(merge_sorted_variations(variations1,
                                                             variations2,
                        ignore_2_or_more_overlaps=ignore_2_or_more_overlaps)):
        if not ignore_overlaps:
            if prev_snp1 is not None and snp1 is not None and are_overlapping(prev_snp1, snp1):
                raise ValueError('Overlapping variations in variations 1')
            if prev_snp2 is not None and snp2 is not None and are_overlapping(prev_snp2, snp2):
                raise ValueError('Overlapping variations in variations 2')
        result = merge_snps(snp1, snp2, i, merged_variations,
                            fields_funct=fields_funct,
                            ignore_fields=ignore_fields,
                            check_ref_match=check_ref_match)
        ignored_snps, merged_snps, added_snps = result
        log['ignored_ref_snps'] += ignored_snps
        log['modified_merged_snps'] += merged_snps
        log['added_new_snps'] += added_snps
        if snp1 is not None:
            num_snp1 -= 1
            prev_snp1 = snp1
        if snp2 is not None:
            num_snp2 -= 1
            prev_snp2 = snp2
    log['ignored_overlap_snps'] = num_snp1 + num_snp2
    for path in merged_variations.keys():
        new_shape = (i+1, ) + merged_variations[path].shape[1:]
        merged_variations[path].resize(new_shape)
    log['total_merged_snps'] = merged_variations['/variations/pos'].shape[0]
    return merged_variations, log
