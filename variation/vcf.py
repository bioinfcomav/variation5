
import gzip
import re
from itertools import zip_longest, chain
import os
import subprocess

import numpy
import h5py

from variation.compressed_queue import CCache

# Test imports
import inspect
from os.path import dirname, abspath, join

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                         '..', 'test_data'))
TEST_VCF = join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.vcf.gz')

TEST_VCF2 = join(TEST_DATA_DIR, 'format_def.vcf')

# Speed is related to chunksize, so if you change snps-per-chunk check the
# performance
SNPS_PER_CHUNK = 200

MISSING_INT = -1
MISSING_GT = MISSING_INT
MISSING_FLOAT = float('nan')
MISSING_STR = ''
FILLING_INT = -2
FILLING_FLOAT = float('-inf')
FILLING_STR = MISSING_STR

def _do_nothing(value):
    return value


def _to_int(string):
    if string in ('', '.', None, b'.'):
        return MISSING_INT
    return int(string)


def _to_float(string):
    if string in ('', '.', None):
        return MISSING_FLOAT
    return float(string)


def _gt_data_to_list(mapper_function, sample_gt, missing_data):
    if sample_gt is None:
        # we cannot now at this point how many items compose a gt for a sample
        # so we cannot return [missing_data]
        return None

    sample_gt = sample_gt.split(b',')
    sample_gt = [mapper_function(item) for item in sample_gt]
    return sample_gt


def _missing_val(dtype_str):
    if 'int' in dtype_str:
        missing_val = MISSING_INT
    elif 'float' in dtype_str:
        missing_val = MISSING_FLOAT
    elif 'str' in dtype_str:
        missing_val = MISSING_STR
    return missing_val


def _filling_val(dtype_str):
    if 'int' in dtype_str:
        missing_val = FILLING_INT
    elif 'float' in dtype_str:
        missing_val = FILLING_FLOAT
    elif 'str' in dtype_str:
        missing_val = FILLING_STR
    return missing_val


class VCF():
    def __init__(self, fhand, pre_read_max_size=None,
                 ignored_fields=None, max_field_lens=None):
        self._fhand = fhand
        self.metadata = None
        self.vcf_format = None
        self.ploidy = None
        if ignored_fields is None:
            ignored_fields = []
        ignored_fields = [field.encode('utf-8') for field in ignored_fields]
        self.ignored_fields = ignored_fields

        self._determine_ploidy()

        self._empty_gt = [MISSING_GT] * self.ploidy
        self._parse_header()

        if max_field_lens is None:
            user_max_field_lens = {}
        else:
            user_max_field_lens = max_field_lens
        max_field_lens = {'alt': 0, 'FILTER': 0, 'INFO': {}, 'FORMAT': {}}
        max_field_lens.update(user_max_field_lens)
        self.max_field_lens = max_field_lens

        self.max_field_str_lens = {'FILTER': 0, 'INFO': {}, 'chrom': 0,
                                   'alt': 0}
        self._init_max_field_lens()

        self._parsed_gt_fmts = {}
        self._parsed_gt = {}

        self.pre_read_max_size = pre_read_max_size
        self._variations_cache = CCache()
        self._read_snps_in_compressed_cache()


    def _init_max_field_lens(self):
        meta = self.metadata
        for section in ('INFO', 'FORMAT'):
            for field, meta_field in meta[section].items():
                if isinstance(meta_field['Number'], int):
                    continue
                self.max_field_lens[section][field] = 0
                if 'str' in meta_field['dtype']:
                    self.max_field_str_lens[section][field] = 0


    def _read_snps_in_compressed_cache(self):
        if not self.pre_read_max_size:
            return
        self._variations_cache.put_iterable(self._variations(),
                                            max_size=self.pre_read_max_size)

    def _determine_ploidy(self):
        read_lines = []
        ploidy = None
        for line in self._fhand:
            read_lines.append(line)
            if line.startswith(b'#'):
                continue
            gts = line.split(b'\t')[9:]
            for gt in gts:
                if gt is b'.':
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

        metadata = {'FORMAT': {}, 'FILTER': {}, 'INFO': {}, 'OTHER': {}}
        metadata['VARIATIONS'] = {'chrom': {'dtype': 'str',
                                            'type': _do_nothing},
                                  'pos': {'dtype': 'int32',
                                          'type': _to_int},
                                  'id': {'dtype': 'str',
                                         'type': _do_nothing},
                                  'ref': {'dtype': 'str',
                                          'type': _do_nothing},
                                  'qual': {'dtype': 'float16',
                                          'type': _to_float},
                                  'alt': {'dtype': 'str',
                                         'type': _do_nothing},}
        for line in header_lines:
            if line[2:7] in (b'FORMA', b'INFO=', b'FILTE'):
                line = line[2:]
                meta = {}
                if line.startswith(b'FORMAT='):
                    meta_kind = 'FORMAT'
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
                                val = _to_int
                                val2 = 'int16'
                            elif val == 'Float':
                                val = _to_float
                                val2 = 'float16'
                            else:
                                val = _do_nothing
                                val2 = 'str'
                            meta['dtype'] = val2
                        meta[key] = val
                if id_ is None:
                    raise RuntimeError('Header line has no ID: ' + line)
                # The fields with a variable number of items
                if 'Number' in meta and meta['Number'].isdigit():
                    meta['Number'] = int(meta['Number'])
            else:
                id_, meta = line[2:].decode('utf-8').split('=', 1)
                if id_ == 'fileformat':
                    self.vcf_format = meta
                    continue
                meta_kind = 'OTHER'
            id_ = id_.encode('utf-8')
            metadata[meta_kind][id_] = meta
        self.metadata = metadata

    def _parse_info(self, info):
        infos = info.split(b';')
        parsed_infos = {}
        ignored_fields = self.ignored_fields
        for info in infos:
            if b'=' in info:
                key, val = info.split(b'=', 1)
            else:
                key, val = info, True
            if key in ignored_fields:
                continue
            try:
                meta = self.metadata['INFO'][key]
            except KeyError:
                msg = 'INFO metadata was not defined in header: '
                msg += key.decode('utf-8')
                raise RuntimeError(msg)

            type_ = meta['Type']
            if isinstance(val, bool):
                pass
            elif b',' in val:
                val = [type_(val) for val in val.split(b',')]
                val_to_check_len = val
            else:
                val = type_(val)
                val_to_check_len = [val]
            if not isinstance(meta['Number'], int):
                if self.max_field_lens['INFO'][key] < len(val_to_check_len):
                    self.max_field_lens['INFO'][key] = len(val_to_check_len)
                if 'str' in meta['dtype']:
                    max_str = max([len(val_) for val_ in val_to_check_len])
                    if self.max_field_str_lens['INFO'][key] < max_str:
                        self.max_field_str_lens['INFO'][key] = max_str

            parsed_infos[key] = val
        return parsed_infos

    def _parse_gt_fmt(self, fmt):
        orig_fmt = fmt
        try:
            return self._parsed_gt_fmts[fmt]
        except KeyError:
            pass

        meta = self.metadata['FORMAT']
        format_ = []
        for fmt in fmt.split(b':'):
            try:
                fmt_meta = meta[fmt]
            except KeyError:
                msg = 'FORMAT metadata was not defined in header: '
                msg += fmt.decode('utf-8')
                raise RuntimeError(msg)
            format_.append((fmt, fmt_meta['Type'],
                            fmt_meta['Number'] != 1,  # Is list
                            fmt_meta,
                            _missing_val(fmt_meta['dtype'])))
        self._parsed_gt_fmts[orig_fmt] = format_
        return format_

    def _parse_gt(self, gt):
        gt_str = gt
        try:
            return self._parsed_gt[gt]
        except KeyError:
            pass

        if gt is None:
            gt = self._empty_gt
        elif b'|' in gt:
            is_phased = True
            gt = gt.split(b'|')
        else:
            is_phased = False
            gt = gt.split(b'/')
        if gt is not None:
            gt = [MISSING_GT if allele == b'.' else int(allele) for allele in gt]
        self._parsed_gt[gt_str] = gt
        return gt

    def _parse_gts(self, fmt, gts):
        fmt = self._parse_gt_fmt(fmt)
        empty_gt = [None] * len(fmt)

        gts = [empty_gt if gt == b'.' else gt.split(b':') for gt in gts]
        gts = zip(*gts)

        parsed_gts = []
        ignored_fields = self.ignored_fields
        for fmt, gt_data in zip(fmt, gts):
            if fmt[0] in ignored_fields:
                continue
            if fmt[0] == b'GT':
                gt_data = [self._parse_gt(sample_gt) for sample_gt in gt_data]
            else:
                if fmt[2]:  # the info for a sample in this field is or should
                            # be a list
                    gt_data = [_gt_data_to_list(fmt[1], sample_gt, fmt[4]) for sample_gt in gt_data]
                else:
                    gt_data = [fmt[1](sample_gt) for sample_gt in gt_data]

            meta = fmt[3]
            if not isinstance(meta['Number'], int):
                max_len = max([0 if data is None else len(data) for data in gt_data])
                if self.max_field_lens['FORMAT'][fmt[0]] < max_len:
                    self.max_field_lens['FORMAT'][fmt[0]] = max_len
                if 'str' in meta['dtype'] and fmt[0] != b'GT':
                    # if your file has variable length str fields you
                    # should check and fix the following part of the code
                    raise NotImplementedError('Fixme')
                    print(gt_data)
                    print( [val for smpl_data in gt_data for val in smpl_data])
                    max_len = max([len(val) for smpl_data in gt_data for val in smpl_data])
                    max_str = max([len(val) for val_ in val])
                    if self.max_field_str_lens['FORMAT'][key] < max_str:
                        self.max_field_str_lens['FORMAT'][key] = max_str

            parsed_gts.append((fmt[0], gt_data))

        return parsed_gts

    @property
    def variations(self):
        return chain(self._variations_cache.items, self._variations())

    def _variations(self):
        for line in self._fhand:
            line = line[:-1]
            items = line.split(b'\t')
            chrom, pos, id_, ref, alt, qual, flt, info, fmt = items[:9]

            if self.max_field_str_lens['chrom'] < len(chrom):
                self.max_field_str_lens['chrom'] = len(chrom)

            gts = items[9:]
            pos = int(pos)
            if id_ == b'.':
                id_ = None

            alt = alt.split(b',')
            if self.max_field_lens['alt'] < len(alt):
                self.max_field_lens['alt'] = len(alt)
            max_alt_str_len = max(len(allele) for allele in alt)
            if self.max_field_str_lens['alt'] < max_alt_str_len:
                self.max_field_str_lens['alt'] = max_alt_str_len

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
            if self.max_field_lens['FILTER'] < flt_len:
                self.max_field_lens['FILTER'] = flt_len
            qual = float(qual) if qual != b'.' else None

            info = self._parse_info(info)
            gts = self._parse_gts(fmt, gts)
            yield chrom, pos, id_, ref, alt, qual, flt, info, gts


def _grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


TYPES = {'int16': numpy.int16,
         'int32': numpy.int32,
         'float16': numpy.float16}

DEF_DSET_PARAMS = {
    'compression': 'gzip',  # Not much slower than lzf, but compresses much
                            # more
    'shuffle': True,
    'fletcher32': True  # checksum, slower but safer
}


def _numpy_dtype(dtype, field, max_field_str_lens):
    if 'str' in dtype:
        if field in max_field_str_lens:
            dtype = 'S{}'.format(max_field_str_lens[field] + 5)
        else:
            # the field is empty
            dtype = 'S1'
    else:
        dtype = TYPES[dtype]
    return dtype


def _prepare_info_datasets(vcf, hdf5, vars_in_chunk):
    meta = vcf.metadata['INFO']
    var_grp = hdf5['variations']
    info_grp = var_grp.create_group('info')

    info_fields = set(meta.keys()).difference(vcf.ignored_fields)
    info_fields = list(info_fields)

    ok_fields = []
    for field in info_fields:
        meta_fld = meta[field]
        dtype = _numpy_dtype(meta_fld['dtype'], field,
                             vcf.max_field_str_lens)

        if field not in vcf.max_field_lens['INFO']:
            # We assume that it is not used by any SNP
            continue

        y_axes_size = vcf.max_field_lens['INFO'][field]

        if y_axes_size == 1:
            size = (vars_in_chunk,)
            maxshape = (None,)
            chunks = (vars_in_chunk,)
        else:
            size = [vars_in_chunk, y_axes_size]
            maxshape = (None, y_axes_size)
            chunks = (vars_in_chunk, y_axes_size)

        missing_val = None

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        kwargs['fillvalue'] = missing_val
        info_grp.create_dataset(field, size, **kwargs)
        ok_fields.append(field)
    return ok_fields


def _prepate_call_datasets(vcf, hdf5, vars_in_chunk):
    n_samples = len(vcf.samples)

    calldata = hdf5['calls']
    ploidy = vcf.ploidy

    fmt_fields = set(vcf.metadata['FORMAT'].keys()).difference(vcf.ignored_fields)
    fmt_fields = list(fmt_fields)

    for field in fmt_fields:
        fmt = vcf.metadata['FORMAT'][field]
        if field == b'GT':
            z_axes_size = ploidy
            dtype = numpy.int8
        else:
            dtype = _numpy_dtype(fmt['dtype'], field,
                             vcf.max_field_str_lens)
            if isinstance(fmt['Number'], int):
                z_axes_size = fmt['Number']
            else:
                if field == b'GT':
                    z_axes_size = vcf.ploidy
                else:
                    z_axes_size = vcf.max_field_lens['FORMAT'][field]

        size = [vars_in_chunk, n_samples, z_axes_size]
        maxshape = (None, n_samples, z_axes_size)
        chunks = (vars_in_chunk, n_samples, z_axes_size)

        if field == b'GT':
            missing_val = MISSING_GT
        else:
            missing_val = _filling_val(fmt['dtype'])

        # If the last dimension only has one of len we can work with only
        # two dimensions (variations x samples)
        if size[-1] == 1:
            size = size[:-1]
            maxshape = maxshape[:-1]
            chunks = chunks[:-1]

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        kwargs['fillvalue'] = missing_val
        calldata.create_dataset(field, size, **kwargs)
    return fmt_fields


def _prepare_variation_datasets(vcf, hdf5, vars_in_chunk):

    meta = vcf.metadata['VARIATIONS']
    var_grp = hdf5['variations']

    one_item_fields = ['chrom', 'pos', 'id', 'ref', 'qual']
    multi_item_fields = ['alt']
    fields = one_item_fields + multi_item_fields
    for field in fields:
        if field in one_item_fields:
            size = [vars_in_chunk]
            maxshape = (None,)  # is resizable, we can add SNPs
            chunks=(vars_in_chunk,)
        else:
            y_axes_size = vcf.max_field_lens[field]
            size = [vars_in_chunk, y_axes_size]
            maxshape = (None, y_axes_size)  # is resizable, we can add SNPs
            chunks=(vars_in_chunk,  y_axes_size)

        dtype = meta[field]['dtype']
        dtype = _numpy_dtype(meta[field]['dtype'], field,
                             vcf.max_field_str_lens)

        missing_val = None

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        kwargs['fillvalue'] = missing_val
        var_grp.create_dataset(field, size, **kwargs)


def _prepare_filter_datasets(vcf, hdf5, vars_in_chunk):

    var_grp = hdf5['variations']
    filter_grp = var_grp.create_group('filter')

    meta = vcf.metadata['FILTER']
    filter_fields = set(meta.keys()).difference(vcf.ignored_fields)
    filter_fields = list(filter_fields)
    if not filter_fields:
        return []

    filter_fields.append('no_filters')

    for field in filter_fields:
        dtype = numpy.bool_

        size = (vars_in_chunk,)
        maxshape = (None,)
        chunks = (vars_in_chunk,)

        missing_val = None

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        kwargs['fillvalue'] = missing_val
        filter_grp.create_dataset(field, size, **kwargs)
    return filter_fields


def _expand_list_to_size(items, desired_size, filling):
    extra_empty_items = [filling[0]] * (desired_size - len(items))
    items.extend(extra_empty_items)


def vcf_to_hdf5(vcf, out_fpath, vars_in_chunk=SNPS_PER_CHUNK):
    snps = vcf.variations

    log = {'data_no_fit': {},
           'variations_processed': 0}

    hdf5 = h5py.File(out_fpath)

    var_grp = hdf5.create_group('variations')
    calldata = hdf5.create_group('calls')

    fmt_fields = _prepate_call_datasets(vcf, hdf5, vars_in_chunk)
    info_fields = _prepare_info_datasets(vcf, hdf5, vars_in_chunk)
    filter_fields = _prepare_filter_datasets(vcf, hdf5, vars_in_chunk)
    _prepare_variation_datasets(vcf, hdf5, vars_in_chunk)
    var_fields = ['chrom', 'pos', 'id', 'ref', 'qual', 'alt']

    info_grp = var_grp['info']
    filter_grp = var_grp['filter']

    fields = var_fields[:]
    fields.extend(fmt_fields)
    fields.extend(info_fields)
    fields.extend(filter_fields)

    snp_chunks = _grouper(snps, vars_in_chunk)
    for chunk_i, chunk in enumerate(snp_chunks):
        chunk = list(chunk)

        first_field = True
        for field in fields:
            if field in var_fields:
                dset = var_grp[field]
                grp = 'VARIATIONS'
            elif field in info_fields:
                dset = info_grp[field]
                grp = 'INFO'
            elif field in filter_fields:
                dset = filter_grp[field]
                grp = 'FILTER'
            else:
                dset = calldata[field]
                grp = 'FORMAT'

            # resize the dataset to fit the new chunk
            size = dset.shape
            new_size = list(size)
            new_size[0] = vars_in_chunk * (chunk_i + 1)
            dset.resize(new_size)

            if grp == 'FILTER':
                missing = False
                filling = False
            else:
                missing = _missing_val(vcf.metadata[grp][field]['dtype'])
                filling = _filling_val(vcf.metadata[grp][field]['dtype'])

            if len(size) == 3 and field != b'GT':
                missing = [missing] * size[2]
                filling = [filling] * size[2]
            elif len(size) == 2:
                missing = [missing] * size[1]
                filling = [filling] * size[1]

            # We store the information
            for snp_i, snp in enumerate(chunk):
                try:
                    gt_data = snp[-1]
                except TypeError:
                    # SNP is None
                    break
                if first_field:
                    log['variations_processed'] += 1

                snp_n = snp_i + chunk_i * vars_in_chunk

                if grp == 'FILTER':
                    data = snp[6]
                    if field == 'no_filters':
                        data = data is None
                    else:
                        data = field in data
                elif grp == 'INFO':
                    info_data = snp[7]
                    info_data = info_data.get(field, None)
                    if info_data is not None:
                        if len(size) == 1:
                            # we're expecting one item or a list with one item
                            print (field, info_data)
                            if isinstance(info_data, (list, tuple)):
                                assert len(info_data) == 1
                                info_data = info_data[0]
                        try:
                            dset[snp_n] = info_data
                        except TypeError as error:
                            if 'broadcast' in str(error):
                                if field not in log['data_no_fit']:
                                    log['data_no_fit'][field] = 0
                                log['data_no_fit'][field] += 1

                elif grp == 'VARIATIONS':
                    if field == 'chrom':
                        item = snp[0]
                    elif field == 'pos':
                        item = snp[1]
                    elif field == 'id':
                        item = snp[2]
                    elif field == 'ref':
                        item = snp[3]
                    elif field == 'alt':
                        item = snp[4]
                        _expand_list_to_size(item, size[1], [b''])
                    elif field == 'qual':
                        item = snp[5]
                    if item is not None:
                        try:
                            dset[snp_n] = item
                        except TypeError as error:
                            if 'broadcast' in str(error) and field == 'alt':
                                msg = 'More alt alleles than expected.'
                                msg2 = 'Expected, present: {}, {}'
                                msg2 = msg2.format(size[1], len(item))
                                msg += msg2
                                msg = '\nYou might fix it prereading more'
                                msg += ' SNPs, or passing: '
                                msg += 'max_field_lens={'
                                msg += 'alt:{}'.format(len(item))
                                msg += '}\nto VCF reader'
                                raise TypeError(msg)

                elif grp == 'FORMAT':
                    # store the calldata
                    gt_data = dict(gt_data)
                    call_sample_data = gt_data.get(field, None)

                    if call_sample_data is not None:
                        if len(size) == 2:
                            # we're expecting a single item or a list with one item
                            if isinstance(call_sample_data[0], (list, tuple)):
                                # We have a list in each item
                                # we're assuming that all items have length 1
                                assert max(map(len, call_sample_data)) == 1
                                call_sample_data =  [item[0] for item in call_sample_data]
                        elif field == b'GT':
                            pass
                        else:
                            call_sample_data = _expand_list_to_size(call_sample_data,
                                                                    size[2],
                                                                    filling)

                        if call_sample_data is not None:
                            try:
                                dset[snp_n] = call_sample_data
                            except TypeError as error:
                                if 'broadcast' in str(error):
                                    if field not in log['data_no_fit']:
                                        log['data_no_fit'][field] = 0
                                    log['data_no_fit'][field] += 1
            first_field = False

    # we have to remove the empty snps from the last chunk
    for field in fields:
        if field in var_fields:
            dset = var_grp[field]
        elif field in filter_fields:
            dset = filter_grp[field]
        elif field in info_fields:
            dset = info_grp[field]
        else:
            dset = calldata[field]

        size = dset.shape
        new_size = list(size)
        snp_n = snp_i + chunk_i * vars_in_chunk
        new_size[0] = snp_n
        dset.resize(new_size)

    return log


def read_gzip_file(fpath):
    zcat = ['zcat']
    pigz = ['pigz', '-dc']
    cmd = zcat
    cmd.append(fpath)
    gz_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in gz_process.stdout:
        yield line


def test():

    out_fhand = 'snps.hdf5'

    if os.path.exists(out_fhand):
        os.remove(out_fhand)
    fhand = open(TEST_VCF2, 'rb')
    vcf = VCF(fhand, pre_read_max_size=1000)

    log = vcf_to_hdf5(vcf, out_fhand)
    print(log)

    if os.path.exists(out_fhand):
        os.remove(out_fhand)

    if False:
        fhand = gzip.open(TEST_VCF, 'rb')
    else:
        lines = read_gzip_file(TEST_VCF)
        fhand = lines

    max_size_cache = 1024**3
    max_size_cache = 1000
    ignored_fields = ['RO', 'AO', 'DP', 'GQ', 'QA', 'QR', 'GL']
    ignored_fields = ['QA', 'QR', 'GL']
    ignored_fields = ['RO', 'AO', 'DP', 'GQ', 'QA', 'QR', 'GL']
    #ignored_fields = []
    vcf = VCF(fhand, ignored_fields=ignored_fields,
	          pre_read_max_size=max_size_cache, max_field_lens={'alt': 4})


    print (vcf.max_field_lens)
    print (vcf.max_field_str_lens)

    log = vcf_to_hdf5(vcf, out_fhand)
    print(log)
    #hdf5 = h5py.File(out_fhand, 'r')

    #read_chunks(open('snps.hdf5'), ['caldata/GT'])

if __name__ == '__main__':
    test()

