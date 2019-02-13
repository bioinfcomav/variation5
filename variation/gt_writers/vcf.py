from functools import partial
from math import factorial
from multiprocessing import Pool
import os
from tempfile import NamedTemporaryFile

import numpy
import re
from collections import OrderedDict

from variation import (VCF_FORMAT, MISSING_FLOAT, MISSING_STR, MISSING_INT,
                       MISSING_VALUES)
from variation.utils.file_utils import remove_temp_file_in_dir
from variation.utils.misc import remove_nans

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

str_type_regex = re.compile('\|S')
int_type_regex = re.compile('[\|<]i')
float_type_regex = re.compile('<f')
bool_type_regex = re.compile('\|b')
consecutive_digits_regex = re.compile('\d+')

_CACHE_WITH_SEP_MATRICES = {}


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


def _numbered_chunks(variations, chunk_size):
    chunks = enumerate(variations.iterate_chunks(chunk_size=chunk_size))
    for index, chunk in chunks:
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
                                  _numbered_chunks(variations, chunk_size))
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
    VCF_body_lines = _get_VCF_body_lines(variations)
    for line in VCF_body_lines:
        #print(line)
        out_fhand.write(line + b"\n")


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
                    value = numpy.array(value)
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
            value = [str(val) for val in value if not numpy.isnan(val)]
            value = ','.join(value) if value else None
        elif not value:
            value = None
        else:
            raise(NotImplemented)

        if 'bool' not in dtype and value and value is not None:
            info.append('{}={}'.format(field_key, value))

    return ';'.join(info)


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


def _join_value(master_data, detail_data, separator=b'', detail_is_sliced=False,
                index=0):
    #use index only if data is 3d
    if detail_is_sliced:
        detail_data = detail_data[...,index]
    #add separator before all elements except first
    if index != 0:
        chararray_separator = numpy.full(master_data.shape, separator.encode())
        master_data = numpy.char.add(master_data, chararray_separator)
    #add detail
    master_data = numpy.char.add(master_data, detail_data)
    return master_data


def _get_str_mask_from_bool_array(bool_ndarray):
    int_mask = bool_ndarray.astype('int')
    str_mask = int_mask.astype('|S1')
    return str_mask


def _stringify_array(data):
    data_type = data.dtype.str
    data = data[...].copy()

    #check type before casting
    if str_type_regex.match(data_type):
        stringified_data = data
        bool_mask = data == MISSING_STR.encode()
        stringified_data[bool_mask] = '.'
    elif int_type_regex.match(data_type):
        byte_depth = int(consecutive_digits_regex.search(data_type).group(0))
        number_length = len(str(2**(byte_depth*8)))
        target_type = '|S' + str(number_length)
        int_data = data
        stringified_data = data.astype(target_type)
        stringified_data[int_data == MISSING_INT] = b'.'
    elif float_type_regex.match(data_type):
        #float data is rounded to make sure it fits a 16bytes string
        rounded_data = data.astype(numpy.float128).round(decimals=4)
        stringified_data = rounded_data[()].astype('|S16')
        if numpy.isnan(MISSING_FLOAT):
            stringified_data[numpy.isnan(data)] = b'.'
        else:
            raise RuntimeError('FIXME I used to work with nan as misssing float')
    elif bool_type_regex.match(data_type):
        stringified_data = _get_str_mask_from_bool_array(data)

    return stringified_data


def _get_group_variations_paths(variations):
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


def _sum_str_arrays(str_arrays, sep=None):
    concatenated_result = str_arrays[0]
    if sep is not None:
        sep_matrix = numpy.full(concatenated_result.shape, sep)
    for str_array in str_arrays[1:]:
        if sep is not None:
            concatenated_result = numpy.char.add(concatenated_result, sep_matrix)
        concatenated_result = numpy.char.add(concatenated_result, str_array)

    return concatenated_result


def _join_str_array_along_axis0(str_array, sep=None,
                                the_str_array_has_newlines=True):
    if the_str_array_has_newlines:
        raise NotImplementedError('If you want newlinex fix the implementation')

    num_snps = str_array.shape[0]
    if sep:
        shape = str_array.shape
        key = shape, sep
        if key in _CACHE_WITH_SEP_MATRICES:
            sep_matrix = _CACHE_WITH_SEP_MATRICES[key]
        else:
            sep_matrix = numpy.full(shape, sep)
            sep_matrix[:, -1] = b''
            _CACHE_WITH_SEP_MATRICES[key] = sep_matrix

        str_array = _sum_str_arrays([str_array, sep_matrix])

    new_line_column = numpy.full((str_array.shape[0], 1), b'\n')
    str_array = numpy.hstack((str_array, new_line_column))
    str_array_by_snp = numpy.array(str_array.tobytes().replace(b'\x00', b'').split(b'\n')[:-1])
    assert str_array_by_snp.shape[0] == num_snps

    return str_array_by_snp


def _alt_array_to_str_array(variations):
    if '/variations/alt' in variations:
        alt_array = _stringify_array(variations['/variations/alt'])
        alt_field_data = _join_str_array_along_axis0(alt_array,
                                                 sep=b',',
                                                 the_str_array_has_newlines=False)
        return numpy.char.replace(alt_field_data, b',.', b'')
    else:
        return numpy.full((variations.num_variations,), b'.')


def _filter_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if grouped_paths['filter']:
        filter_str_arrays = []
        for filter_path in grouped_paths['filter']:
            if not variations[filter_path].any():
                continue
            filter_name = filter_path.split('/')[-1]
            filter_str_array = numpy.full((variations.num_variations,), b'.')
            filter_bool_array = variations[filter_path]
            filter_str_array[filter_bool_array] = filter_name
            filter_str_arrays.append(filter_str_array)

        filter_field_data = _sum_str_arrays(filter_str_arrays, sep=b';')
        filter_field_data = numpy.char.replace(filter_field_data, b';.', b'')
        return filter_field_data
    else:
        return numpy.full((variations.num_variations,), b'.')


def _info_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if not grouped_paths['info']:
        return numpy.full((variations.num_variations,), b'.')

    info_field_data = numpy.full((variations.num_variations,), b'')
    for i, info_path in enumerate(grouped_paths['info']):
        if variations[info_path] is not None:
            dtype = str(variations[info_path].dtype)
            field_data_holder = numpy.full((variations.num_variations,), b'.')
            field_key = info_path.split('/')[-1]

            #boolean info is translated to <id> by masking
            if 'bool' in dtype:
                if variations[info_path] is not None:
                    bool_mask = variations[info_path]
                    field_data_holder = numpy.full(bool_mask.shape, field_key.encode())
                    field_data_holder[~bool_mask] = '.'

            #non boolean info is preceded by <id>=
            elif 'bool' not in dtype:
                info_data = _stringify_array(variations[info_path])
                bool_mask = info_data == b'.'
                info_equals_string = field_key + '='
                field_data_holder = numpy.full((info_data.shape[0],), info_equals_string.encode())

                #info containing single values is retrieved
                if info_data.ndim == 1:
                    field_data_holder = _sum_str_arrays([field_data_holder, info_data])
                    field_data_holder[bool_mask] = '.'

                #info containing several values is collapsed into a coma separated string
                elif info_data.ndim == 2:
                    for index in range(0, info_data.shape[-1]):
                        info_data_slice = info_data[..., index]
                        bool_mask_slice = bool_mask[..., index]
                        info_data_slice[bool_mask_slice] = '.'
                        field_data_holder = _sum_str_arrays([field_data_holder, info_data_slice], sep=b',')
                    field_data_holder = numpy.char.replace(field_data_holder, b',.', b'')
                    field_data_holder = numpy.char.replace(field_data_holder, b'=,', b'=')

            #data is collapsed into a semicolon separated string
            if i == 0:
                info_field_data = _sum_str_arrays([info_field_data, field_data_holder])
            else:
                info_field_data = _sum_str_arrays([info_field_data, field_data_holder], sep=b';')

    #return numpy.char.replace(info_field_data, b';.', b'')
    return numpy.char.replace(info_field_data, b';.', b'')


def _format_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if grouped_paths['format']:
        format_string = ':'.join(grouped_paths['format'])

        return numpy.full((variations.num_variations,), format_string.encode())
    else:
        return numpy.full((variations.num_variations,), b'.')


def _calls_arrays_to_str_array(variations):
    sample_quantity = len(variations.samples)
    grouped_paths = _get_group_variations_paths(variations)
    if grouped_paths['calls']:

        call_2d_matrices = []
        for i, calls_path in enumerate(grouped_paths['calls']):

            if variations[calls_path] is None:
                continue
            calls_data_uncollapsed = _stringify_array(variations[calls_path])
            data_id = calls_path.split('/')[-1]

            if calls_data_uncollapsed.ndim == 2:
                str_array_for_field = calls_data_uncollapsed
            elif calls_data_uncollapsed.ndim == 3:
                str_array_for_field = calls_data_uncollapsed[..., 0]
                sep = b'/' if data_id == 'GT' else b','
                for chromosome_set in range(1, calls_data_uncollapsed.shape[-1]):
                    str_array_for_field = _sum_str_arrays([str_array_for_field, calls_data_uncollapsed[..., chromosome_set]], sep)

            call_2d_matrices.append(str_array_for_field)

        field_str_array_by_sample = _sum_str_arrays(call_2d_matrices, b':')
        calls_field_data = _join_str_array_along_axis0(field_str_array_by_sample, sep=b'\t',
                                                 the_str_array_has_newlines=False)
        return numpy.char.replace(calls_field_data, b',.', b'')
    else:
        return numpy.full((variations.num_variations,), b'.')


def _one_field_array_to_str_array(variations, field_path):
    if field_path in variations.keys():
        one_field_data = _stringify_array(variations[field_path])
    else:
        one_field_data = numpy.full((variations.num_variations,), b'.')
    return one_field_data


def _get_VCF_body_lines(variations):
    to_str_arrays = (('/variations/chrom', partial(_one_field_array_to_str_array,
                                                  field_path='/variations/chrom')),
                     ('/variations/pos', partial(_one_field_array_to_str_array,
                                                                  field_path='/variations/pos')),
                     ('/variations/id', partial(_one_field_array_to_str_array,
                                                                field_path='/variations/id')),
                     ('/variations/ref', partial(_one_field_array_to_str_array,
                                                                 field_path='/variations/ref')),
                     ('/variations/alt', _alt_array_to_str_array),
                     ('/variations/qual', partial(_one_field_array_to_str_array,
                                                                 field_path='/variations/qual')),
                     ('/variations/filter', _filter_arrays_to_str_array),
                     ('/variations/info', _info_arrays_to_str_array),
                     ('/variations/format', _format_arrays_to_str_array),
                     ('/variations/calls', _calls_arrays_to_str_array),
                     )
    to_str_arrays = OrderedDict(to_str_arrays)

    VCF_body_stringified_fields = OrderedDict()
    for field_path in to_str_arrays.keys():
        VCF_body_stringified_fields[field_path] = to_str_arrays[field_path](variations)

    vcf_lines_array = _sum_str_arrays(list(VCF_body_stringified_fields.values()), sep=b'\t')

    return vcf_lines_array
