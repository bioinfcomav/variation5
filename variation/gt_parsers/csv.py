import copy

from itertools import chain
from variation import DEF_METADATA, MISSING_INT

IUPAC = {b'A': b'AA', b'T': b'TT', b'C': b'CC', b'G': b'GG',
         b'W': b'AT', b'M': b'AC', b'R': b'AG', b'': b'-',
         b'Y': b'TC', b'K': b'TG', b'S': b'CG', b'-': b'-'}


MISSING_GT_VALUES = (b'-', b'', b'.', b'--')
MISSING_ALLELE_VALUES = (ord('-'), ord('N'))


def def_gt_allele_splitter(gt):
    if gt in MISSING_GT_VALUES:
        return None
    not_missing = False
    alleles = []
    for allele in gt:
        allele = allele
        if allele in MISSING_ALLELE_VALUES:
            allele = None
        else:
            not_missing = True
        alleles.append(allele)
    if not not_missing:
        alleles = None
    return tuple(alleles)


def create_iupac_allele_splitter(ploidy=2):
    if ploidy != 2:
        msg = 'It is not possible to generate the genotypes for other ploidy'
        raise ValueError(msg)

    def iupac_allele_splitter(gt):
        return def_gt_allele_splitter(IUPAC[gt])
    return iupac_allele_splitter


class CSVParser():
    def __init__(self, fhand, var_info, gt_splitter=def_gt_allele_splitter,
                 first_sample_column=1, first_gt_column=1,
                 sample_line=0, snp_id_column=0, sep=',', max_field_lens=None,
                 max_field_str_lens=None, ignore_empty_vars=False):
        '''It reads genotype calls from a CSV file

        var_info can be either a dict or an OrderedDict with the snp_ids as
        keys and the values should have a dict with, at least, the keys chrom
        and pos. The rest of the key can match the VCF fields.

        gt_splitter should take a genotype as it is stored in the CSV file and
        it should return a tuple with the alleles in byte format.
        '''
        self.fhand = fhand
        self._sample_line = sample_line
        self._first_sample_column = first_sample_column
        self._first_gt_column = first_gt_column
        self._sep = sep
        self._snp_id_column = snp_id_column
        self.gt_splitter = gt_splitter
        self._var_info = var_info
        self._ignore_empty_vars = ignore_empty_vars

        self.samples = self._get_samples()
        self._determine_ploidy()
        if max_field_lens is None:
            self.max_field_lens = {'alt': 0}
        else:
            self.max_field_lens = max_field_lens
        if max_field_str_lens is None:
            self.max_field_str_lens = {'alt': 0,
                                       'chrom': 0}
        else:
            self.max_field_str_lens = max_field_str_lens
        self.metadata = copy.deepcopy(DEF_METADATA)
        self.ignored_fields = []
        self.kept_fields = []

    def _determine_ploidy(self):
        read_lines = []
        one_line = False
        for line in self.fhand:
            one_line = True
            read_lines.append(line)
            items = line.split(self._sep)
            items[-1] = items[-1].strip()
            gts = items[self._first_gt_column:]
            gts = [self.gt_splitter(gt) for gt in gts]
            for gt in gts:
                if gt is None:
                    continue
                else:
                    self.ploidy = len(gt)
                    break

        if not one_line:
            raise RuntimeError('File is empty')
        if 'ploidy' not in dir(self):
            raise RuntimeError('Could not determine ploidy.')

        self.fhand = chain(read_lines, self.fhand)

    def _get_samples(self):
        for line_num, line in enumerate(self.fhand):
            if line_num == self._sample_line:
                return line.rstrip().split(self._sep)[self._first_sample_column:]

        raise RuntimeError("We didn't reach to sample line")

    def _parse_gts(self, items):
        gts = items[self._first_gt_column:]
        recoded_gts = []
        gts = [self.gt_splitter(gt) for gt in gts]
        alleles = set()
        for gt in gts:
            if gt is None:
                continue
            for allele in gt:
                alleles.add(allele)
        allele_coding = {allele: idx for idx, allele in enumerate(alleles)}
        allele_coding[None] = None
        genotype_coding = {None: (MISSING_INT,) * self.ploidy}
        for gt in gts:
            try:
                coded_gt = genotype_coding[gt]
            except KeyError:
                coded_gt = tuple([allele_coding[allele] for allele in gt])
            genotype_coding[gt] = coded_gt
            recoded_gts.append(coded_gt)
        return (tuple([chr(allele).encode() for allele in alleles]),
                [(b'GT', recoded_gts)])

    @property
    def variations(self):
        max_field_lens = self.max_field_lens
        max_field_str_lens = self.max_field_str_lens
        for line in self.fhand:
            items = line.split(self._sep)
            items[-1] = items[-1].strip()

            snp_id = items[self._snp_id_column]
            alleles, gts = self._parse_gts(items)
            var_info = self._var_info[snp_id]

            alt_alleles = list(alleles[1:]) if len(alleles) > 1 else None

            if alt_alleles:
                if max_field_lens['alt'] < len(alt_alleles):
                    max_field_lens['alt'] = len(alt_alleles)
                max_len = max(len(allele) for allele in alt_alleles)
                if max_field_str_lens['alt'] < max_len:
                    max_field_str_lens['alt'] = max_len
            if not alleles:
                if self._ignore_empty_vars:
                    continue
                else:
                    raise RuntimeError('snp {} is empty'.format(snp_id))

            variation = (var_info['chrom'], var_info['pos'], snp_id,
                         alleles[0], alt_alleles, None, None, None, gts)
            yield variation
