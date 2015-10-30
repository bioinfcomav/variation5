# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111
from csv import DictReader
from itertools import chain
from variation import MISSING_VALUES


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
