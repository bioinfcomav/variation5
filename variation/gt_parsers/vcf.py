from itertools import chain
import re
import subprocess
from multiprocessing import Pool

from variation import (MISSING_VALUES, SNPS_PER_CHUNK)
from variation.iterutils import group_items

# The following functions have to be compiled with
# python setup.py build_ext --inplace
from variation.gt_parsers.vcf_field_parsers import (_parse_info,
                                                    _parse_calls)

# Missing docstring
# pylint: disable=C0111


def read_gzip_file(fpath, pgiz=False):
    if pgiz:
        cmd = ['pigz', '-dc']
    else:
        cmd = ['zcat']

    cmd.append(fpath)
    gz_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in gz_process.stdout:
        yield line


def _gt_data_to_list_old(mapper_function, sample_gt):
    if sample_gt is None:
        # we cannot now at this point how many items compose a gt for a sample
        # so we cannot return [missing_data]
        return None

    sample_gt = sample_gt.split(b',')
    sample_gt = [mapper_function(item) for item in sample_gt]
    return sample_gt


def _detect_fields_in_vcf(metadata, fields):
    check_fields = []
    for field in fields:
        type_ = field.decode('utf-8').split('/')[1].upper()
        check_field = field.decode('utf-8').split('/')[2]
        if type_ == 'CALLS':
            check_field = check_field.encode('utf-8')
        if check_field not in list(metadata[type_]):
            msg = 'Field does not exist in vcf ' + field.decode('utf-8')
            raise ValueError(msg)
        check_fields.append(check_field)
    return check_fields


class VCFParser():

    def __init__(self, fhand, ignored_fields=None, kept_fields=None,
                 max_n_vars=None, n_threads=None):
        if kept_fields is not None and ignored_fields is not None:
            msg = 'kept_fields and ignored_fields can not be set at the same'
            msg += ' time'
            raise ValueError(msg)
        self._fhand = fhand
        self.n_threads = n_threads
        self.metadata = None
        self.vcf_format = None
        self.ploidy = None
        # We remove the unwanted fields
        if ignored_fields is None:
            ignored_fields = []
        ignored_fields = [field.encode('utf-8') for field in ignored_fields]
        if kept_fields is None:
            kept_fields = []
        kept_fields = [field.encode('utf-8') for field in kept_fields]
        self.ignored_fields = ignored_fields
        self.kept_fields = kept_fields
        self._determine_ploidy()

        self._empty_gt = [MISSING_VALUES[int]] * self.ploidy
        self._parse_header()

        self._parsed_gt_fmts = {}
        self._parsed_gt = {}

    def _determine_ploidy(self):
        read_lines = []
        ploidy = None
        for line in self._fhand:
            read_lines.append(line)
            if line.startswith(b'#'):
                continue
            gts = line.split(b'\t')[9:]
            for gt in gts:
                if gt == b'.':
                    continue
                gt = gt.split(b':')[0]
                if gt == b'.':
                    continue
                alleles = gt.split(b'/') if b'/' in gt else gt.split(b'|')
                ploidy = len(alleles)
                break
            if ploidy is not None:
                break
        self.ploidy = ploidy
        # we have to restore the read lines to the iterator
        self._fhand = chain(read_lines, self._fhand)

    def _parse_header(self):
        # read the header lines
        header_lines = []
        for line in self._fhand:
            if line.startswith(b'#CHROM'):
                self.samples = line.strip().split(b'\t')[9:]
                break
            header_lines.append(line)
        metadata = {'CALLS': {}, 'FILTER': {}, 'INFO': {}, 'OTHER': {}}
        metadata['VARIATIONS'] = {'chrom': {'dtype': 'str'},
                                  'pos': {'dtype': 'int32'},
                                  'id': {'dtype': 'str'},
                                  'ref': {'dtype': 'str'},
                                  'qual': {'dtype': 'float16'},
                                  'alt': {'dtype': 'str'}}
        for line in header_lines:
            if line[2:7] in (b'FORMA', b'INFO=', b'FILTE'):
                line = line[2:]
                meta = {}
                if line.startswith(b'FORMAT='):
                    meta_kind = 'CALLS'
                    line = line[8:-2]
                elif line.startswith(b'FILTER='):
                    meta_kind = 'FILTER'
                    line = line[8:-2]
                elif line.startswith(b'INFO='):
                    meta_kind = 'INFO'
                    line = line[6:-2]
                else:
                    msg = 'Unsuported VCF: ' + line.decode("utf-8")
                    raise RuntimeError(msg)

                line = line.decode("utf-8")
                items = re.findall(r'(?:[^,"]|"(?:\\.|[^"])*")+',
                                   line)
                id_ = None

                for item in items:
                    key, val = item.split('=', 1)
                    if key == 'ID':
                        id_ = val.strip()
                    else:
                        if key == 'Type':
                            if val == 'Integer':
                                val2 = 'int16'
                            elif val == 'Float':
                                val2 = 'float16'
                            elif val == 'Flag':
                                val2 = 'bool'
                            else:
                                val2 = 'str'
                            meta['dtype'] = val2
                        val = val.strip('"')
                        meta[key] = val
                if id_ is None:
                    raise RuntimeError('Header line has no ID: ' + line)
                # The fields with a variable number of items
                if 'Number' in meta and meta['Number'].isdigit():
                    meta['Number'] = int(meta['Number'])
            else:
                id_, meta = line[2:].decode('utf-8').split('=', 1)
                meta = meta.strip()
                if id_ == 'fileformat':
                    self.vcf_format = meta
                    continue
                meta_kind = 'OTHER'
            id_ = id_.encode('utf-8')

            metadata[meta_kind][id_] = meta

        self.kept_fields = _detect_fields_in_vcf(metadata, self.kept_fields)
        self.ignored_fields = _detect_fields_in_vcf(metadata,
                                                    self.ignored_fields)

        self.metadata = metadata

    @property
    def variations(self):
        n_threads = self.n_threads
        lines_chunks = group_items(self._fhand, SNPS_PER_CHUNK)
        if n_threads:
            pool = Pool(self.n_threads)
            chunk_size = int(SNPS_PER_CHUNK / (2 * n_threads))
        else:
            pool = None

        parser_args = {'ignored_fields': self.ignored_fields,
                       'kept_fields': self.kept_fields,
                       'metadata': self.metadata,
                       'empty_gt': self._empty_gt}
        line_parser = VCFLineParser(**parser_args)
        for lines_chunk in lines_chunks:
            lines_chunk = list(lines_chunk)
            if n_threads:
                snps = pool.map(line_parser, lines_chunk, chunksize=chunk_size)
            else:
                snps = map(line_parser, lines_chunk)

            for snp in snps:
                if snp is None:
                    continue
                yield snp


class VCFLineParser:

    def __init__(self, ignored_fields, kept_fields, metadata, empty_gt):
        self.ignored_fields = ignored_fields
        self.kept_fields = kept_fields
        self.metadata = metadata
        self.empty_gt = empty_gt

    def __call__(self, line):
        if line is None:
            return None

        ignored_fields = self.ignored_fields
        kept_fields = self.kept_fields
        metadata = self.metadata
        empty_gt = self.empty_gt
        line = line[:-1]
        items = line.split(b'\t')
        chrom, pos, id_, ref, alt, qual, flt, info, fmt = items[:9]

        calls = items[9:]
        pos = int(pos)
        if id_ == b'.':
            id_ = None

        alt = alt.split(b',')
        # there is no alternative allele
        if alt == [b'.']:
            alt = None

        qual = float(qual) if qual != b'.' else None

        if flt == b'PASS':
            flt = []
        elif flt == b'.':
            flt = None
        else:
            flt = flt.split(b';')

        info = _parse_info(info, ignored_fields, metadata)

        calls = _parse_calls(fmt, calls, ignored_fields, kept_fields, metadata,
                             empty_gt)

        return chrom, pos, id_, ref, alt, qual, flt, info, calls
