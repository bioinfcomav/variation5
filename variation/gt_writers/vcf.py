import os
from math import factorial

import numpy as np

from variation import (VCF_FORMAT, MISSING_FLOAT, MISSING_STR, MISSING_INT,
                       MISSING_VALUES)
from variation.utils.misc import remove_nans
from multiprocessing import Pool
from functools import partial
from tempfile import NamedTemporaryFile
from variation.utils.file_utils import remove_temp_file_in_dir
# from variation.gt_writers.vcf_field_writer import cy_create_snv_line


GROUP_FIELD_MAPPING = {'/variations/info': 'INFO', '/calls': 'FORMAT',
                       '/variations/filter': 'FILTER', '/other/alt': 'ALT',
                       '/other/contig': 'contig', '/other/sample': 'SAMPLE',
                       '/other/pedigree': 'PEDIGREE',
                       '/other/pedigreedb': 'pedigreeDB'}

VCF_FIELDS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO',
              'FORMAT', 'CALLS']


ONEDATA_FIELD_GROUP_MAPPING = {'CHROM': {'path': '/variations/chrom',
                                         'dtype': 'str',
                                         'missing': MISSING_STR},
                               'POS': {'path': '/variations/pos',
                                       'dtype': 'int', 'missing': MISSING_INT},
                               'REF': {'path': '/variations/ref',
                                       'dtype': 'str', 'missing': MISSING_STR},
                               'QUAL': {'path': '/variations/qual',
                                        'dtype': 'float',
                                        'missing': MISSING_FLOAT},
                               'ID': {'path': '/variations/id',
                                      'missing': MISSING_STR, 'dtype': 'str'}}


def _write_header_line(_id, record, group=None):
    required_fields = {'INFO': ['Number', 'Type', 'Description'],
                       'FILTER': ['Description'],
                       'FORMAT': ['Number', 'Type', 'Description'],
                       'ATL': ['Description']}
    if group is None:
        line = '##{}={}\n'.format(_id.strip('/'), record)
    else:
        line = '##{}=<ID={}'.format(group, _id)
        for key in required_fields[group]:
            value = record[key]
            if key == 'Description':
                value = '"{}"'.format(value)
            line += ',{}={}'.format(key, value)
        line += '>\n'
    return line


def _parse_group_id(key):
    splitted_keys = key.split('/')
    group = '/'.join(splitted_keys[:-1])
    return GROUP_FIELD_MAPPING[group], splitted_keys[-1]


def _write_vcf_meta(variations, out_fhand, vcf_format):
    out_fhand.write('##fileformat={}\n'.format(vcf_format).encode())
    metadata = variations.metadata
    for key, value in metadata.items():
        if not isinstance(value, dict):
            line = _write_header_line(key, value, group=None)
            out_fhand.write(line.encode())

    for key, value in sorted(metadata.items()):
        if isinstance(value, dict) and key in variations:
            group, id_ = _parse_group_id(key)
            line = _write_header_line(id_, value, group=group)
            out_fhand.write(line.encode())


def _write_vcf_header(variations, out_fhand):
    header_items = VCF_FIELDS[:-1] + variations.samples
    header = '#' + '\t'.join(header_items) + '\n'
    out_fhand.write(header.encode())


def numbered_chunks(variations):
    for index, chunk in enumerate(variations.iterate_chunks()):
        yield index, chunk


def _merge_vcfs(vcf_fpaths, out_fhand):
    for vcf_fpath in sorted(vcf_fpaths):
        vcf_fhand = open(vcf_fpath, 'rb')
        out_fhand.write(vcf_fhand.read())
        vcf_fhand.close()


def write_vcf_parallel(variations, out_fhand, n_threads, tmp_dir,
                       chunk_size=None, vcf_format=VCF_FORMAT):
    _write_vcf_meta(variations, out_fhand, vcf_format=vcf_format)
    _write_vcf_header(variations, out_fhand)

    grouped_paths = _group_variations_paths(variations)
    _partial_write_snvs = partial(_write_snvs_parallel, tmp_dir=tmp_dir,
                                  grouped_paths=grouped_paths)

    with Pool(n_threads) as pool:
        try:
            vcf_fpaths = pool.map(_partial_write_snvs,
                                  numbered_chunks(variations))
        except Exception:
            remove_temp_file_in_dir(tmp_dir, '.vcf.h5')
            raise
    try:
        _merge_vcfs(vcf_fpaths, out_fhand)
    except Exception:
        raise
    finally:
        for vcf_fpath in vcf_fpaths:
            if os.path.exists(vcf_fpath):
                os.remove(vcf_fpath)


def _write_snvs_parallel(numbered_chunk, tmp_dir, grouped_paths):
    order, variations = numbered_chunk
    tmp_fhand = NamedTemporaryFile(delete=False, dir=tmp_dir,
                                   prefix='{}-'.format(order),
                                   suffix='.tmp.vcf')
    _write_snvs(variations, tmp_fhand, grouped_paths)
    tmp_fhand.close()
    return tmp_fhand.name


def write_vcf(variations, out_fhand, vcf_format=VCF_FORMAT):
    _write_vcf_meta(variations, out_fhand, vcf_format=vcf_format)
    _write_vcf_header(variations, out_fhand)

    grouped_paths = _group_variations_paths(variations)
    for chunk in variations.iterate_chunks():
        _write_snvs(chunk, out_fhand, grouped_paths)


def _write_snvs(variations, out_fhand, grouped_paths):
    metadata = variations.metadata
    for var_index in range(variations['/calls/GT'][:].shape[0]):
        line = _create_snv_line(variations, var_index, grouped_paths, metadata)
        out_fhand.write(line.encode())


def _create_snv_line(variations, var_index, grouped_paths, metadata):
    var = {}
    for field, data in ONEDATA_FIELD_GROUP_MAPPING.items():
        path = data['path']
        if path in variations.keys():
            value = variations[path][var_index]
            if not value or value == data['missing']:
                value = '.'
            elif data['dtype'] == 'str':
                value = value.decode()
            else:
                value = str(value)
            var[field] = value
        else:
            var[field] = '.'
    if '/variations/alt' in variations:
        alt = [x.decode() for x in variations['/variations/alt'][var_index]
               if x.decode() != MISSING_STR]
        num_alt = len(alt)
        var['ALT'] = '.' if num_alt == 0 else ','.join(alt)
    else:
        var['ALT'] = '.'
        num_alt = 0

    if grouped_paths['filter']:
        var['FILTER'] = _get_filters_value(variations, var_index,
                                           grouped_paths['filter'])
    else:
        var['FILTER'] = '.'

    if grouped_paths['info']:
        var['INFO'] = _get_info_value(variations, var_index,
                                      grouped_paths['info'], metadata,
                                      num_alt)
    else:
        var['INFO'] = '.'

    new_paths = _preprocess_format_calls_paths(variations, var_index,
                                               grouped_paths['calls'])

    new_calls_paths, new_format_paths = new_paths
    var['FORMAT'] = ':'.join(new_format_paths)
    var['CALLS'] = _get_calls_samples(variations, var_index,
                                      new_calls_paths, num_alt,
                                      metadata, variations.ploidy)
    snp_line = '\t'.join([var[field] for field in VCF_FIELDS]) + '\n'
    return snp_line


def _preprocess_format_calls_paths(variations, var_index, calls_paths):

    new_format_paths, new_calls_paths = [], []
    for key in calls_paths:
        values = remove_nans(variations[key][var_index])
        if (not np.all(values == MISSING_VALUES[values.dtype]) and
                values.shape[0] != 0):
            new_calls_paths.append(key)
            new_format_paths.append(key.split('/')[-1])
    return new_calls_paths, new_format_paths


def _get_info_value(variations, index, info_paths, metadata, num_alt):
    info = []
    for key in info_paths:
        dtype = str(variations[key].dtype)
        value = variations[key][index]
        field_key = key.split('/')[-1]
        meta_number = metadata[key]['Number']

        if 'bool' in dtype:
            if value:
                info.append(field_key)
        elif meta_number == 'A':
            try:
                value = value[:num_alt]
            except IndexError:
                if num_alt == 1:
                    value = np.array(value)
                else:
                    raise
            if '|S' in dtype:
                value = [val.decode() for val in value]
            else:
                value = remove_nans(value)
                value = [str(val) for val in value]
            value = ','.join(value) if value else None
        elif meta_number == 1:

            if '|S' in dtype:
                value = value.decode()
            else:
                value = str(value)
        elif meta_number > 1:
            value = [str(val) for val in value if not np.isnan(val)]
            value = ','.join(value) if value else None
        elif not value:
            value = None
        else:
            raise(NotImplemented)

        if 'bool' not in dtype and value and value is not None:
            info.append('{}={}'.format(field_key, value))

    return ';'.join(info)


def _get_filters_value(variations, index, filter_paths):
    filters = None
    for key in filter_paths:
        if filters is None:
            filters = []
        if variations[key][index]:
            filters.append(key.split('/')[-1])

    if filters is None:
        return '.'

    if 'PASS' in filters and len(filters) > 1:
        msg = "FILTER value is wrong. PASS not allowed with another filter"
        raise RuntimeError(msg)
    return ';'.join(filters)

INT_STR_CONVERTER = {num: str(num) for num in range(30000)}
INT_STR_CONVERTER[MISSING_INT] = '.'
INT_STR_CONVERTER[MISSING_FLOAT] = '.'
INT_STR_CONVERTER[MISSING_STR] = '.'

POSSIBLE_GENOS_CACHE = {1: {}, 2: {}, 3: {}}


def _possible_genotypes(num_alleles, ploidy):
    try:
        return POSSIBLE_GENOS_CACHE[ploidy][num_alleles]
    except KeyError:
        possible_geno = factorial(num_alleles + ploidy - 1)
        possible_geno /= (factorial(ploidy) * factorial(num_alleles - 1))
        possible_geno = int(possible_geno)
        POSSIBLE_GENOS_CACHE[ploidy][num_alleles] = possible_geno
        return possible_geno


def _get_calls_per_sample(calls_data, n_sample, calls_path, num_alt, metadata,
                          ploidy):
    calls_sample = []
    for key in calls_path:
        value = calls_data[key][n_sample]

        format_data_number = metadata[key]['Number']
        if format_data_number == 'A':

            try:
                value = value[:num_alt]
            except IndexError:
                print(key, value, format_data_number, num_alt)
                if num_alt == 1:
                    value = [value]
                else:
                    raise

            if 'AO' in key:
                value = [INT_STR_CONVERTER[val] for val in value]
            else:
                value = [str(val) for val in value]
            value = ','.join(value)

        elif format_data_number == 'G':
            value = value[:_possible_genotypes(num_alt + 1, ploidy)]
            value = [str(val) for val in value]
            value = ','.join(value)

        elif format_data_number not in ('A', 'G', 1):
            value = [str(x) if MISSING_VALUES[value.dtype] != x else '.'
                     for x in value[:format_data_number]]
            value = ','.join(value)
        elif format_data_number == 1:
            if 'GT' in key:
                value = [INT_STR_CONVERTER[x] for x in value]
                value = '/'.join(value)

                if '.' in value:
                    return '.'
            elif value == MISSING_VALUES[value.dtype]:
                value = '.'
            elif key in ['/calls/DP', '/calls/RO']:
                try:
                    value = INT_STR_CONVERTER[value]
                except KeyError:
                    value = str(value)
            else:
                value = str(value)
        else:
            raise NotImplemented('We dont know this vcf format number')

        if not value or value is None:
            value = '.'

        calls_sample.append(value)

    return ':'.join(calls_sample)


def _get_calls_samples(h5, var_index, calls_paths, num_alt, metadata, ploidy):
    calls_samples = []
    call_data = {}
    for key in calls_paths:
        call_data[key] = h5[key][var_index]

    for n_sample in range(len(h5.samples)):
        calls_samples.append(_get_calls_per_sample(call_data, n_sample,
                                                   calls_paths, num_alt,
                                                   metadata, ploidy))
    return '\t'.join(calls_samples)


def _group_variations_paths(variations):
    grouped_paths = {'filter': [], 'info': [], 'format': [], 'calls': []}

    if '/calls/GT' in variations.keys():
        grouped_paths['format'] = ['GT']
        grouped_paths['calls'] = ['/calls/GT']
    for key in sorted(variations.keys()):
        if 'calls' in key:
            if 'GT' not in key:
                grouped_paths['format'].append(key.split('/')[-1])
                grouped_paths['calls'].append(key)
        elif 'info' in key:
            grouped_paths['info'].append(key)
        elif 'filter' in key:
            grouped_paths['filter'].append(key)
    return grouped_paths

