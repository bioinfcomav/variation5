from itertools import chain, islice
import re
import subprocess
from multiprocessing import Pool
import math

from variation import (MISSING_VALUES, SNPS_PER_CHUNK)
from variation.utils.compressed_queue import CCache
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


class _VarParserWithPreRead():

    def __init__(self, pre_read_max_size=None, max_n_vars=None):
        self.pre_read_max_size = pre_read_max_size
        self.max_n_vars = max_n_vars
        self._variations_cache = CCache()
        self._read_snps_in_compressed_cache()

    @property
    def max_field_lens(self):
        if self.n_threads is not None:
            msg = 'We do not know how to share dict between processes'
            raise NotImplementedError(msg)
        return self._max_field_lens

    @property
    def max_field_str_lens(self):
        if self.n_threads is not None:
            msg = 'We do not know how to share dict between processes'
            raise NotImplementedError(msg)
        return self._max_field_str_lens

    def _read_snps_in_compressed_cache(self):
        if not self.pre_read_max_size:
            return

        snps = self._variations_for_cache()
        # we store some snps in the cache
        self._variations_cache.put_iterable(snps,
                                            max_size=self.pre_read_max_size)

    @property
    def variations(self):
        if (self.pre_read_max_size is not None and
            math.isinf(self.pre_read_max_size) and
            (not self.max_n_vars or math.isinf(self.max_n_vars))):
            snps = self._variations_cache.items
        else:
            snps = chain(self._variations_cache.items,
                         self._variations(self.n_threads))
        if self.max_n_vars:
            snps = islice(snps, self.max_n_vars)
        return snps

    def _variations_for_cache(self):
        parser_args = {'max_field_lens': self._max_field_lens,
                       'max_field_str_lens': self._max_field_str_lens,
                       'ignored_fields': self.ignored_fields,
                       'kept_fields': self.kept_fields,
                       'metadata': self.metadata,
                       'empty_gt': self._empty_gt,
                       }
        line_parser = VCFLineParser(**parser_args)
        for line in self._fhand:
            snp = line_parser(line)
            yield snp


class VCFParser(_VarParserWithPreRead):

    def __init__(self, fhand, pre_read_max_size=None,
                 ignored_fields=None, kept_fields=None,
                 max_field_lens=None, max_n_vars=None,
                 n_threads=None):
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
        if max_field_lens is None:
            user_max_field_lens = {}
        else:
            user_max_field_lens = max_field_lens
        self._max_field_lens = {'alt': 0, 'FILTER': 0, 'INFO': {}, 'CALLS': {}}
        self._max_field_str_lens = {'FILTER': 0, 'INFO': {},
                                    'chrom': 0, 'alt': 0, 'ref': 0, 'id': 10}
        self._init_max_field_lens()
        for key1, value1 in user_max_field_lens.items():
            if isinstance(value1, dict):
                for key2, value2 in value1.items():
                    self._max_field_lens[key1][key2] = value2
            else:
                self._max_field_lens[key1] = value1

        self._parsed_gt_fmts = {}
        self._parsed_gt = {}
        super().__init__(pre_read_max_size=pre_read_max_size,
                         max_n_vars=max_n_vars)

    def _init_max_field_lens(self):
        meta = self.metadata
        for section in ('INFO', 'CALLS'):
            for field, meta_field in meta[section].items():

                if isinstance(meta_field['Number'], int):
                    self._max_field_lens[section][field] = meta_field['Number']
                    if 'bool' in meta_field['dtype']:
                        self._max_field_lens[section][field] = 1
                    continue
                self._max_field_lens[section][field] = 0
                if 'str' in meta_field['dtype']:
                    self._max_field_str_lens[section][field] = 0

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

    def _variations(self, n_threads):
        lines_chunks = group_items(self._fhand, SNPS_PER_CHUNK)
        if n_threads:
            pool = Pool(self.n_threads)
            chunk_size = int(SNPS_PER_CHUNK / (2 * n_threads))
        else:
            pool = None

        parser_args = {'max_field_lens': self._max_field_lens,
                       'max_field_str_lens': self._max_field_str_lens,
                       'ignored_fields': self.ignored_fields,
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

    def __init__(self, max_field_lens, max_field_str_lens, ignored_fields,
                 kept_fields, metadata, empty_gt):
        self.max_field_lens = max_field_lens
        self.max_field_str_lens = max_field_str_lens
        self.ignored_fields = ignored_fields
        self.kept_fields = kept_fields
        self.metadata = metadata
        self.empty_gt = empty_gt

    def __call__(self, line):
        if line is None:
            return None

        max_field_lens = self.max_field_lens
        max_field_str_lens = self.max_field_str_lens
        ignored_fields = self.ignored_fields
        kept_fields = self.kept_fields
        metadata = self.metadata
        empty_gt = self.empty_gt
        line = line[:-1]
        items = line.split(b'\t')
        chrom, pos, id_, ref, alt, qual, flt, info, fmt = items[:9]
        if max_field_str_lens['chrom'] < len(chrom):
            max_field_str_lens['chrom'] = len(chrom)

        if max_field_str_lens['ref'] < len(ref):
            max_field_str_lens['ref'] = len(ref)

        calls = items[9:]
        pos = int(pos)
        if id_ == b'.':
            id_ = None

        alt = alt.split(b',')
        # there is no alternative allele
        if alt == [b'.']:
            alt = None

        if alt is not None:
            if max_field_lens['alt'] < len(alt):
                max_field_lens['alt'] = len(alt)
            max_alt_str_len = max(len(allele) for allele in alt)
            if max_field_str_lens['alt'] < max_alt_str_len:
                max_field_str_lens['alt'] = max_alt_str_len

        qual = float(qual) if qual != b'.' else None

        if flt == b'PASS':
            flt = []
            flt_len = 0
        elif flt == b'.':
            flt = None
            flt_len = 0
        else:
            flt = flt.split(b';')
            flt_len = len(flt)
        if max_field_lens['FILTER'] < flt_len:
            max_field_lens['FILTER'] = flt_len
        info = _parse_info(info, ignored_fields, metadata, max_field_lens,
                           max_field_str_lens)
        calls = _parse_calls(fmt, calls, ignored_fields, kept_fields,
                             max_field_lens, metadata, empty_gt)

        return chrom, pos, id_, ref, alt, qual, flt, info, calls
