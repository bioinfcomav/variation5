import posixpath
import json
import copy
from collections import Counter

import numpy
import h5py

from variation import (SNPS_PER_CHUNK, MISSING_VALUES, VCF_FORMAT,
                       DEF_DSET_PARAMS, MISSING_INT, MISSING_STR,
                       MISSING_FLOAT)
from variation.iterutils import first, group_items
from variation.matrix.stats import counts_by_row
from variation.matrix.methods import append_matrix, is_dataset
from variation.utils.misc import remove_nans
from variation.variations.stats import GT_FIELD
# Missing docstring
# pylint: disable=C0111


TYPES = {'int16': numpy.int16,
         'int32': numpy.int32,
         'float16': numpy.float16,
         'bool': numpy.bool}


def _to_str(str_or_byte):
    try:
        str_ = str(str_or_byte, 'utf-8')
    except TypeError:
        str_ = str_or_byte
    return str_


def _dset_metadata_from_matrix(mat):
    shape = mat.shape
    dtype = mat.dtype

    if hasattr(mat, 'chunks'):
        chunks = mat.chunks
        maxshape = mat.maxshape
    else:
        chunks = list(shape)
        chunks[0] = SNPS_PER_CHUNK
        chunks = tuple(chunks)
        maxshape = list(shape)
        maxshape[0] = None
        maxshape = tuple(maxshape)
    fillvalue = MISSING_VALUES[dtype]
    return shape, dtype, chunks, maxshape, fillvalue


def _prepare_metadata(vcf_metadata):
    groups = ['INFO', 'FILTER', 'CALLS', 'OTHER']
    meta = {}
    for group in groups:
        for field, field_meta in vcf_metadata[group].items():
            if group == 'INFO':
                dir_ = '/variations/info'
            elif group == 'FILTER':
                dir_ = '/variations/filter'
            elif group == 'CALLS':
                dir_ = '/calls'
            elif group == 'OTHER':
                dir_ = '/'
            path = posixpath.join(dir_, _to_str(field))
            meta[path] = field_meta
    return meta


def _build_matrix_structures(vars_parser, vars_in_chunk, kept_fields,
                             ignored_fields, ignore_undefined_fields, log,
                             max_field_lens, max_field_str_lens):
    structure = {}
    metadata = vars_parser.metadata
    n_samples = len(vars_parser.samples)
    ploidy = vars_parser.ploidy

    if max_field_lens is None:
        max_field_lens = vars_parser.max_field_lens
    if max_field_str_lens is None:
        max_field_str_lens = vars_parser.max_field_str_lens

    # filters
    filters = list(metadata['FILTER'].keys())
    if filters:
        filters.append(b'PASS')
        for field in filters:
            path = posixpath.join('/variations/filter', field.decode())
            structure[path] = {'dtype': numpy.bool, 'shape': (vars_in_chunk,),
                               'missing_value': False, 'missing_item': False,
                               'basepath': 'FILTER', 'field': field}

    for basepath in metadata:
        if basepath in ('OTHER', 'FILTER'):
            continue
        for field in metadata[basepath]:
            try:
                field_str = field.decode()
            except AttributeError:
                field_str = field

            basepath_ = basepath.lower()
            if basepath_ == 'info':
                basepath_ = '/variations/info'
            path = posixpath.join('/', basepath_, field_str)

            # dtype
            dtype = getattr(numpy, metadata[basepath][field]['dtype'])
            if path == '/calls/GT':
                dtype = numpy.int8
            missing_value = MISSING_VALUES[dtype]
            if 'str' in str(dtype):
                if basepath in ('INFO', 'CALLS'):
                    max_field_str_lens_ = max_field_str_lens[basepath]
                else:
                    max_field_str_lens_ = max_field_str_lens

                try:
                    str_len = max_field_str_lens_[field]
                except KeyError:
                    if not ignore_undefined_fields:
                        msg = 'No str len defined for field: {}'.format(field)
                        raise RuntimeError(msg)
                    log['undefined_fields'].append(path)
                    continue
                dtype = numpy.dtype((bytes, str_len))
            # extra dimension
            number_dims = metadata[basepath][field].get('Number', '')
            try:
                if path == '/calls/GT':
                    number_dims = ploidy
                if path in ('/variations/pos', '/variations/ref',
                            '/variations/qual', '/variations/chrom',
                            '/variations/id'):
                    number_dims = 1
                else:
                    number_dims = int(number_dims)
            except ValueError:
                try:
                    if basepath in ('INFO', 'CALLS'):
                        number_dims = max_field_lens[basepath][field]
                    else:
                        number_dims = max_field_lens[field]
                except KeyError:
                    if not ignore_undefined_fields:
                        msg = 'No len defined for field: {}'.format(field)
                        raise RuntimeError(msg)
                    log['undefined_fields'].append(path)
                    continue
            # shape
            if basepath == 'VARIATIONS':
                if field == 'alt':
                    shape = (vars_in_chunk, number_dims)
                else:
                    shape = (vars_in_chunk,)
            elif basepath == 'CALLS':
                if number_dims > 1:
                    shape = (vars_in_chunk, n_samples, number_dims)
                else:
                    shape = (vars_in_chunk, n_samples)
            else:
                if number_dims > 1:
                    shape = (vars_in_chunk, number_dims)
                else:
                    shape = (vars_in_chunk,)

            structure[path] = {'dtype': dtype, 'shape': shape,
                               'missing_value': missing_value,
                               'basepath': basepath, 'field': field}

    fields = set(structure.keys())
    if ignored_fields:
        fields = fields.difference(ignored_fields)
    if kept_fields:
        fields = fields.intersection(kept_fields)

    structure = {fld: struct
                 for fld, struct in structure.items() if fld in fields}
    return structure


class _ChunkGenerator:
    def __init__(self, vars_parser, hdf5, vars_in_chunk, kept_fields=None,
                 ignored_fields=None, max_field_lens=None,
                 max_field_str_lens=None):
        self.vars_parser = vars_parser
        self.hdf5 = hdf5
        self.vars_in_chunk = vars_in_chunk
        self.kept_fields = kept_fields
        self.ignored_fields = ignored_fields
        self.log = {'data_no_fit': Counter(),
                    'variations_processed': 0,
                    'variations_stored': 0,
                    'undefined_fields': []}
        self.max_field_lens = max_field_lens
        self.max_field_str_lens = max_field_str_lens

    @property
    def chunks(self):
        vars_parser = self.vars_parser
        hdf5 = self.hdf5
        vars_in_chunk = self.vars_in_chunk
        kept_fields = self.kept_fields
        ignored_fields = self.ignored_fields
        max_field_lens = self.max_field_lens
        max_field_str_lens = self.max_field_str_lens
        log = self.log

        ignore_overflows = hdf5.ignore_overflows
        snps = vars_parser.variations

        mat_structure = _build_matrix_structures(vars_parser, vars_in_chunk,
                                                 kept_fields, ignored_fields,
                                                 hdf5.ignore_undefined_fields,
                                                 log, max_field_lens,
                                                 max_field_str_lens)
        for chunk in group_items(snps, vars_in_chunk):
            mats = {}
            for path, struct in mat_structure.items():
                mat = numpy.full(struct['shape'], struct['missing_value'],
                                 struct['dtype'])
                mats[path] = mat
            good_snp_idxs = []
            for idx, snp in enumerate(chunk):
                if snp is None:
                    break
                log['variations_processed'] += 1

                filters = snp[6]
                info = snp[7]
                calls = snp[8]
                info = dict(info) if info else {}
                calls = dict(calls) if calls else {}
                ignore_snp = False
                for path, struct in mat_structure.items():
                    basepath = struct['basepath']
                    if path == '/variations/chrom':
                        item = snp[0]
                    elif path == '/variations/pos':
                        item = snp[1]
                    elif path == '/variations/id':
                        item = snp[2]
                    elif path == '/variations/ref':
                        item = snp[3]
                    elif path == '/variations/alt':
                        item = snp[4]
                    elif path == '/variations/qual':
                        item = snp[5]
                    elif basepath == 'FILTER':
                        if struct['field'] == b'PASS':
                            item = True if filters == [] else False
                        else:
                            item = struct['field'] in filters
                    elif basepath == 'INFO':
                        item = info.get(struct['field'], None)
                    elif basepath == 'CALLS':
                        item = calls.get(struct['field'], None)
                    shape = struct['shape']

                    if item is not None:
                        n_dims = len(shape)
                        mat = mats[path]
                        if n_dims == 1:
                            try:
                                mat[idx] = item
                            except ValueError:
                                if hasattr(item, '__len__'):
                                    if len(item) == 1:
                                        mat[idx] = item[0]
                                    else:
                                        log['data_no_fit'][path] += 1
                                        break
                                else:
                                    raise
                        elif n_dims == 2:
                            if len(item) > mat.shape[1]:
                                if ignore_overflows:
                                    ignore_snp = True
                                    log['data_no_fit'][path] += 1
                                    break
                                else:
                                    msg = 'Data no fit in field:'
                                    msg += path
                                    msg += '\n'
                                    msg += str(item)
                                    raise RuntimeError(msg)
                            try:
                                mat[idx, 0:len(item)] = item
                            except (ValueError, TypeError):
                                missing_val = struct['missing_value']
                                item = [missing_val if val is None else val[0]
                                        for val in item]
                                mat[idx, 0:len(item)] = item

                        elif n_dims == 3:
                            if len(item[0]) > mat.shape[2]:
                                if ignore_overflows:
                                    ignore_snp = True
                                    log['data_no_fit'][path] += 1
                                    break
                                else:
                                    msg = 'Data no fit in field:'
                                    msg += path
                                    msg += '\n'
                                    msg += str(item)
                                    raise RuntimeError(msg)
                            # mat[idx, :, 0:len(item[0])] = item
                            try:
                                mat[idx, :, 0:len(item[0])] = item
                            except ValueError:
                                print(path, item)
                                raise

                        else:
                            raise RuntimeError('Fixme, we should not be here.')
                if not ignore_snp:
                    good_snp_idxs.append(idx)
                    log['variations_stored'] += 1

            varis = VariationsArrays()
            for path, mat in mats.items():
                varis[path] = mat[good_snp_idxs]
            samples = [sample.decode() for sample in vars_parser.samples]
            varis.samples = samples

            metadata = _prepare_metadata(vars_parser.metadata)
            varis._set_metadata(metadata)

            yield varis


def _put_vars_in_mats(vars_parser, hdf5, vars_in_chunk, kept_fields=None,
                      ignored_fields=None, max_field_lens=None,
                      max_field_str_lens=None):
    chunker = _ChunkGenerator(vars_parser, hdf5, vars_in_chunk,
                              kept_fields=kept_fields,
                              ignored_fields=ignored_fields,
                              max_field_lens=max_field_lens,
                              max_field_str_lens=max_field_str_lens)
    hdf5.put_chunks(chunker.chunks)
    return chunker.log


def _preprocess_header_line(h5, _id, record, group=None):
        required_fields = {'INFO': ['Number', 'Type', 'Description'],
                           'FILTER': ['Description'],
                           'FORMAT': ['Number', 'Type', 'Description'],
                           'ATL': ['Description']}
        if group is None:
            line = '##{}={}'.format(_id.strip('/'), record)
        else:
            line = '##{}=<ID={}'.format(group, _id)
            for key in required_fields[group]:
                value = record[key]
                if key == 'Description':
                    value = '"{}"'.format(value)
                line += ',{}={}'.format(key, value)
#             for key, value in record.items():
#                 print(record.items(), group)
#                 print(key in required_fields[group])
#                 if key in required_fields[group]:
#                     continue
#                 line += ',{}={}'.format(key, value)
            line += '>'
        return line


def _prepare_vcf_header(h5, vcf_format):
    metadata = h5.metadata
    yield '##fileformat={}'.format(vcf_format)
    for key in sorted(metadata.keys()):
        if isinstance(metadata[key], dict):
            continue
        yield _preprocess_header_line(h5, key, metadata[key], group=None)

    groups = ['INFO', 'FILTER', 'FORMAT', 'ALT', 'contig', 'SAMPLE',
              'PEDIGREE', 'pedigreeDB']
    fields = ['/variations/info', '/variations/filter',
              '/calls', '/other/alt', '/other/contig', '/other/sample',
              '/other/pedigree', '/other/pedigreedb']
    for group, field in zip(groups, fields):
        for key in sorted(metadata.keys()):
            if not isinstance(metadata[key], dict) or field not in key:
                continue
            _id = key.split('/')[-1]
            yield _preprocess_header_line(h5, _id, metadata[key], group)


def _prepare_fieldnames_line(h5, fieldnames):
    fieldnames.extend(h5.samples)
    return '#{}'.format('\t'.join(fieldnames))


def _get_value_filter(h5, n_snp, filter_paths):
    filters = None
    for key in filter_paths:
        if filters is None:
            filters = []
        if h5[key][n_snp] != False:
            filters.append(key.split('/')[-1])
    if filters is None:
        return '.'
    if 'PASS' in filters and len(filters) > 1:
        msg = "FILTER value is wrong. PASS not allowed with another filter"
        raise RuntimeError(msg)
    return ';'.join(filters)


def _get_value_info(h5, var_index, info_paths):
    info = []
    for key in info_paths:
        if 'bool' in str(h5[key][var_index].dtype):
            if h5[key][var_index]:
                info.append(key.split('/')[-1])
        elif '|S' in str(h5[key][var_index].dtype):
            info_value = h5[key][var_index]
            if not isinstance(info_value, (type(numpy.array), h5py.Dataset)):
                info_value = [h5[key][var_index]]
            new_info = ''
            for value in info_value:
                if not isinstance(value, numpy.ndarray):
                    value = [value]
                for x in value:
                    if b'' != x:
                        new_info += '{}={}'.format(key.split('/')[-1],
                                                   x.decode('utf-8'))
            info.append(new_info)
        else:
            new_info = ''
            # TODO: change to a more elegant way
            value = remove_nans(h5[key][var_index])
            value = [str(x) for x in value]
            if 'numpy' in str(type(h5[key][var_index])):
                if len(value) > 0:
                    val = []
                    for x in value:
                        if x != '-1':
                            val.append(x)
                    new_info = '{}={}'.format(key.split('/')[-1],
                                              ','.join(val))
            elif len(value) == 0:
                new_info = '{}={}'.format(key.split('/')[-1], '.')
            else:
                new_info = '{}={}'.format(key.split('/')[-1], value)
            info.append(new_info)
    if '' in info:
        info.remove('')
    return ';'.join(info)


GT_CONVERTER_CACHE = {num: str(num) for num in range(200)}
GT_CONVERTER_CACHE[MISSING_INT] = '.'
GT_CONVERTER_CACHE[MISSING_FLOAT] = '.'
GT_CONVERTER_CACHE[MISSING_STR] = '.'


def _get_calls_per_sample(h5, var_index, n_sample, calls_path):
    calls_sample = []
    for key in calls_path:
        value = remove_nans(h5[key][var_index][n_sample])
        if 'GT' in key:
            value = [GT_CONVERTER_CACHE[x] for x in value]
            value = '/'.join(value)

            if '.' in value:
                return '.'

        elif h5.metadata[key]['Number'] != 1:
            value = [str(x) if MISSING_VALUES[value.dtype] != x else '.'
                     for x in value]
            value = ','.join(value)
        else:
            if value == MISSING_VALUES[value.dtype]:
                value = '.'
            else:
                value = str(value[0])
        calls_sample.append(value)
    return ':'.join(calls_sample)


def _get_calls_samples(h5, var_index, calls_paths):
    calls_samples = []
    for n_sample in range(len(h5.samples)):
        calls_samples.append(_get_calls_per_sample(h5, var_index, n_sample,
                                                   calls_paths))
    return '\t'.join(calls_samples)


def _preprocess_format_calls_paths(variations, var_index, format_paths,
                                   calls_paths):
    new_format_paths, new_calls_paths = [], []
    for key in calls_paths:
        values = remove_nans(variations[key][var_index])
        if (not numpy.all(values == MISSING_VALUES[values.dtype]) and
                values.shape[0] != 0):
            new_calls_paths.append(key)
            new_format_paths.append(key.split('/')[-1])
    return new_calls_paths, new_format_paths


def _to_vcf(variations, vcf_format=VCF_FORMAT):
    for line in _prepare_vcf_header(variations, vcf_format=vcf_format):
        yield line

    filter_paths = []
    info_paths = []
    format_paths, calls_paths = [], []
    if '/calls/GT' in variations.keys():
        format_paths = ['GT']
        calls_paths = ['/calls/GT']
    for key in sorted(variations.keys()):
        if 'calls' in key:
            if 'GT' not in key:
                format_paths.append(key.split('/')[-1])
                calls_paths.append(key)
        elif 'info' in key:
            info_paths.append(key)
        elif 'filter' in key:
            filter_paths.append(key)

    fieldnames = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
                  'INFO', 'FORMAT']
    yield _prepare_fieldnames_line(variations, fieldnames)
    fieldnames = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
                  'INFO', 'FORMAT', 'CALLS']

    fields_vcf_paths = {'CHROM': '/variations/chrom',
                        'POS': '/variations/pos',
                        'REF': '/variations/ref',
                        'QUAL': '/variations/qual',
                        'ID': '/variations/id'}
    for var_index in range(variations['/calls/GT'][:].shape[0]):
        var = {}
        for vcf_field, var_path in fields_vcf_paths.items():
            if var_path in variations.keys():
                value = variations[var_path][var_index]
                if value == MISSING_VALUES[variations[var_path].dtype]:
                    value = '.'
                elif '|S' in str(variations[var_path].dtype):
                    value = value.decode()
                var[vcf_field] = str(value)
        alt = [x.decode() for x in variations['/variations/alt'][var_index]
               if x.decode() != MISSING_VALUES['str']]
        if len(alt) == 0:
            alt = '.'
        else:
            alt = ','.join(alt)
        var['ALT'] = alt

        if len(filter_paths) == 0:
            var['FILTER'] = '.'
        else:
            var['FILTER'] = _get_value_filter(variations, var_index,
                                              filter_paths)
        if len(info_paths) == 0:
            var['INFO'] = '.'
        else:
            var['INFO'] = _get_value_info(variations, var_index, info_paths)

        # Remove fields that are missing in all samples
        new_paths = _preprocess_format_calls_paths(variations, var_index,
                                                   format_paths, calls_paths)
        new_calls_paths, new_format_paths = new_paths
        var['FORMAT'] = ':'.join(new_format_paths)
        var['CALLS'] = _get_calls_samples(variations, var_index,
                                          new_calls_paths)
        yield '\t'.join([var[field] for field in fieldnames])


class _VariationMatrices():
    def __init__(self, vars_in_chunk=SNPS_PER_CHUNK,
                 ignore_overflows=False, ignore_undefined_fields=False,
                 kept_fields=None, ignored_fields=None):
        self._vars_in_chunk = vars_in_chunk
        self.ignore_overflows = ignore_overflows
        self.ignore_undefined_fields = ignore_undefined_fields
        self.kept_fields = kept_fields
        self.ignored_fields = ignored_fields

    @property
    def ploidy(self):
        return self['/calls/GT'].shape[2]

    def create_matrix_from_matrix(self, path, matrix):

        result = _dset_metadata_from_matrix(matrix)
        shape, dtype, chunks, maxshape, fillvalue = result
        try:
            dset = self._create_matrix(path, shape=shape,
                                       dtype=dtype,
                                       chunks=chunks,
                                       maxshape=maxshape,
                                       fillvalue=fillvalue)
            new_matrix = dset
        except TypeError:
            array = self._create_matrix(path, shape=shape, dtype=dtype,
                                        fillvalue=fillvalue)
            new_matrix = array
        if is_dataset(matrix):
            array = matrix[:]
        else:
            array = matrix
        new_matrix[:] = array
        return new_matrix

    def _create_mats_from_chunks(self, mats_chunks):
        matrices = {}
        for path in mats_chunks.keys():
            mat_chunk = mats_chunks[path]
            matrix = self.create_matrix_from_matrix(path, mat_chunk)
            matrices[path] = matrix
        return matrices

    def write_vcf(self, vcf_fhand):
        for line in _to_vcf(self):
            vcf_fhand.write(line + '\n')

    @staticmethod
    def _check_shape_matches(mat1, mat2, field):
        shape1 = mat1.shape
        shape2 = mat2.shape
        msg = 'matrix in chunk and in self have not matching shape: ' + field
        if len(shape1) != len(shape2):
            raise ValueError(msg)
        if len(shape1) > 1:
            if shape1[1] != shape2[1]:
                raise ValueError(msg)
        if len(shape1) > 2:
            if shape1[2] != shape2[2]:
                raise ValueError(msg)

    def _get_mats_for_chunk(self, variations):
        field_paths = variations.keys()
        diff_fields = set(self.keys()).difference(set(field_paths))

        if diff_fields:
            msg = 'Previous matrices do not match matrices in chunk'
            raise ValueError(msg)

        matrices = {}

        for field in field_paths:
            mat1 = variations[field]
            mat2 = self[field]
            self._check_shape_matches(mat1, mat2, field)
            append_matrix(mat2, mat1)
            matrices[field] = mat2
        return matrices

    def _create_or_get_mats_from_chunk(self, variations):
        field_paths = variations.keys()
        if first(field_paths) in self:
            matrices = self._get_mats_for_chunk(variations)
        else:
            if self.keys():
                raise ValueError('There are previous no matching matrices')
            matrices = self._create_mats_from_chunks(variations)
            self._set_metadata(variations.metadata)
            self._set_samples(variations.samples)
        return matrices

    def put_chunks(self, chunks, kept_fields=None, ignored_fields=None):
        matrices = None
        for chunk in chunks:
            if matrices is None:
                matrices = self._create_or_get_mats_from_chunk(chunk)
                continue
            # check all chunks have the same number of snps
            nsnps = [chunk[path].data.shape[0]
                     for path in chunk.keys()]
            num_snps = nsnps[0]
            assert all(num_snps == nsnp for nsnp in nsnps)

            for path in chunk.keys():
                dset_chunk = chunk[path]
                dset = matrices[path]
                append_matrix(dset, dset_chunk)

        if hasattr(self, 'flush'):
            self._h5file.flush()

    def get_chunk(self, index, kept_fields=None, ignored_fields=None):

        paths = self._filter_fields(kept_fields=kept_fields,
                                    ignored_fields=ignored_fields)

        dsets = {field: self[field] for field in paths}

        var_array = None
        for path, dset in dsets.items():
            matrix = dset[index]
            if var_array is None:
                var_array = VariationsArrays(vars_in_chunk=matrix.shape[0])
            var_array[path] = dset[index]

        var_array._set_metadata(self.metadata)
        var_array._set_samples(self.samples)
        return var_array

    def _filter_fields(self, kept_fields, ignored_fields):
        if kept_fields is not None and ignored_fields is not None:
            msg = 'kept_fields and ignored_fields can not be set at the same'
            msg += ' time'
            raise ValueError(msg)

        # We remove the unwanted fields
        paths = self.keys()
        if kept_fields:
            kept_fields = set(kept_fields)
            not_in_matrix = kept_fields.difference(paths)
            if not_in_matrix:
                msg = 'Some fields are not in this VarMatrices: '
                msg += ', '.join(not_in_matrix)
                raise ValueError(msg)
            paths = kept_fields.intersection(paths)
        if ignored_fields:
            not_in_matrix = set(ignored_fields).difference(paths)
            if not_in_matrix:
                msg = 'Some fields are not in this VarMatrices: '
                msg += ', '.join(not_in_matrix)
                raise ValueError(msg)
            paths = set(paths).difference(ignored_fields)
        return paths

    def iterate_chunks(self, kept_fields=None, ignored_fields=None,
                       chunk_size=None):
        paths = self._filter_fields(kept_fields=kept_fields,
                                    ignored_fields=ignored_fields)

        dsets = {field: self[field] for field in paths}

        # how many snps are per chunk?

        if chunk_size is None:
            chunk_size = self._vars_in_chunk
        nsnps = self.num_variations

        for start in range(0, nsnps, chunk_size):
            var_array = VariationsArrays(vars_in_chunk=chunk_size)
            stop = start + chunk_size
            if stop > nsnps:
                stop = nsnps
            for path, dset in dsets.items():
                var_array[path] = dset[start:stop]
            var_array._set_metadata(self.metadata)
            var_array._set_samples(self.samples)
            yield var_array

    def _set_metadata(self, metadata):
        self._metadata = metadata

    def _get_metadata(self):
        return copy.deepcopy(self._metadata)

    metadata = property(_get_metadata, _set_metadata)

    def _set_samples(self, samples):
        self._samples = samples

    def _get_samples(self):
        return self._samples

    samples = property(_get_samples, _set_samples)

    def values(self):
        return [self[key] for key in self.keys()]

    def __contains__(self, key):
        return key in self.keys()

    @property
    def num_variations(self):
        try:
            one_path = first(self.keys())
        except ValueError:
            return 0
        one_mat = self[one_path]
        return one_mat.shape[0]

    def put_vars(self, var_parser, max_field_lens=None,
                 max_field_str_lens=None):
        return _put_vars_in_mats(var_parser, self, self._vars_in_chunk,
                                 max_field_lens=max_field_lens,
                                 max_field_str_lens=max_field_str_lens,
                                 kept_fields=self.kept_fields,
                                 ignored_fields=self.ignored_fields)

    @property
    def gts_as_mat012(self):
        '''It transforms the GT matrix into 0 (major allele homo), 1 (het),
        2(other hom)'''
        gts = self[GT_FIELD]
        counts = counts_by_row(gts, missing_value=MISSING_INT)
        if counts is None:
            return numpy.full((gts.shape[0], gts.shape[1]),
                              fill_value=MISSING_INT)

        major_alleles = numpy.argmax(counts, axis=1)
        if is_dataset(gts):
            gts = gts[:]
        gts012 = numpy.sum(gts != major_alleles[:, None, None], axis=2)
        gts012[numpy.any(gts == MISSING_INT, axis=2)] = MISSING_INT
        return gts012


def _get_hdf5_dsets(dsets, h5_or_group_or_dset, var_mat):
    if var_mat is not None:
        if not hasattr(dsets, 'keys'):
            msg = 'If you want a dict path:matrix provide a dict for the dsets'
            msg += ', otherwise use _get_hdf5_dset_paths'
            raise ValueError(msg)
    item = h5_or_group_or_dset
    if hasattr(item, 'values'):
        # _h5file or group
        for subitem in item.values():
            _get_hdf5_dsets(dsets, subitem, var_mat)
    else:
        # dset
        path = item.name
        if hasattr(dsets, 'keys'):
            dsets[path] = var_mat[path]
        else:
            dsets.append(path)


def _get_hdf5_dset_paths(dsets, h5_or_group_or_dset):
    # dsets has to be a list
    _get_hdf5_dsets(dsets, h5_or_group_or_dset, None)


class VariationsH5(_VariationMatrices):
    def __init__(self, fpath, mode, vars_in_chunk=SNPS_PER_CHUNK,
                 ignore_overflows=False, ignore_undefined_fields=False,
                 kept_fields=None, ignored_fields=None):
        super().__init__(vars_in_chunk=vars_in_chunk,
                         ignore_overflows=ignore_overflows,
                         ignore_undefined_fields=ignore_undefined_fields,
                         kept_fields=kept_fields,
                         ignored_fields=ignored_fields)
        self._fpath = fpath
        if mode not in ('r', 'w', 'r+'):
            msg = 'mode should be r or w'
            raise ValueError(msg)
        elif mode == 'w':
            mode = 'w-'
        self.mode = mode
        self._h5file = h5py.File(fpath, mode)

    def __getitem__(self, path):
        return self._h5file[path]

    def keys(self):
        dsets = []
        _get_hdf5_dset_paths(dsets, self._h5file)
        return dsets

    def flush(self):
        self._h5file.flush()

    def close(self):
        self._h5file.close()

    @property
    def allele_count(self):
        counts = None
        for gt_chunk in select_dset_from_chunks(self.iterate_chunks(),
                                                '/calls/GT'):
            chunk_counts = counts_by_row(gt_chunk,
                                         missing_value=MISSING_VALUES[int])
            if counts is None:
                counts = chunk_counts
            else:
                if counts.shape[1:] < chunk_counts.shape[1:]:
                    n_extra_cols = chunk_counts.shape[-1] - counts.shape[-1]
                    shape = list(counts.shape)
                    shape[-1] = n_extra_cols
                    extra_cols = numpy.zeros(shape, dtype=chunk_counts.dtype)
                    counts = numpy.hstack((counts, extra_cols))
                elif counts.shape[1:] > chunk_counts.shape[1:]:
                    n_extra_cols = counts.shape[-1] - chunk_counts.shape[-1]
                    shape = list(chunk_counts.shape)
                    shape[-1] = n_extra_cols
                    extra_cols = numpy.zeros(shape, dtype=chunk_counts.dtype)
                    chunk_counts = numpy.hstack((chunk_counts, extra_cols))
                counts = numpy.concatenate([counts, chunk_counts], axis=0)
        return counts

    def _create_matrix(self, path, *args, **kwargs):
        hdf5 = self._h5file
        group_name, dset_name = posixpath.split(path)
        if not dset_name:
            msg = 'The path should include a dset name: ' + path
            raise ValueError(msg)

        try:
            hdf5[path]
            msg = 'The dataset already exists: ' + path
            raise ValueError(msg)
        except KeyError:
            pass

        try:
            group = hdf5[group_name]
        except KeyError:
            group = hdf5.create_group(group_name)

        for key, value in DEF_DSET_PARAMS.items():
            if key not in kwargs:
                kwargs[key] = value

        if 'fillvalue' not in kwargs:
            if 'dtype' in kwargs:
                dtype = kwargs['dtype']
            else:
                if len(args) > 2:
                    dtype = args[2]
                else:
                    dtype = None
            if dtype is not None:
                fillvalue = MISSING_VALUES[dtype]
                kwargs['fillvalue'] = fillvalue
        if 'maxshape' not in kwargs:
            kwargs['maxshape'] = (None,) * len(kwargs['shape'])
        args = list(args)
        args.insert(0, dset_name)
        dset = group.create_dataset(*args, **kwargs)
        return dset

    def _set_metadata(self, metadata):
        self._h5file.attrs['metadata'] = json.dumps(metadata)

    @property
    def metadata(self):
        if 'metadata' in self._h5file.attrs:
            metadata = json.loads(self._h5file.attrs['metadata'])
        else:
            metadata = {}
        return metadata

    def _set_samples(self, samples):
        self._h5file.attrs['samples'] = json.dumps(samples)

    @property
    def samples(self):
        if 'samples' in self._h5file.attrs:
            samples = json.loads(self._h5file.attrs['samples'])
        else:
            if '/calls/GT' not in self.keys():
                raise 'There are not genotypes in hdf5 file'
            samples = None
        return samples


def select_dset_from_chunks(chunks, dset_path):
    return (chunk[dset_path] for chunk in chunks)


class VariationsArrays(_VariationMatrices):
    def __init__(self, vars_in_chunk=SNPS_PER_CHUNK,
                 ignore_overflows=False, ignore_undefined_fields=False,
                 kept_fields=None, ignored_fields=None):
        super().__init__(vars_in_chunk=vars_in_chunk,
                         ignore_overflows=ignore_overflows,
                         ignore_undefined_fields=ignore_undefined_fields,
                         kept_fields=kept_fields,
                         ignored_fields=ignored_fields)
        self._hArrays = {}
        self._metadata = {}
        self._samples = []

    def __getitem__(self, path):
        return self._hArrays[path]

    def __setitem__(self, path, array):
        assert isinstance(array, numpy.ndarray)
        if self.num_variations != 0:
            assert self.num_variations == array.shape[0]
        if path in self._hArrays:
            raise ValueError('This path was already in the var_array', path)
        self._hArrays[path] = array

    def __delitem__(self, path):
        if path in self._hArrays:
            del self._hArrays[path]
        else:
            raise KeyError('The path is not in the variation_array', path)

    def keys(self):
        return self._hArrays.keys()

    @property
    def allele_count(self):
        gts = self['/calls/GT']
        counts = counts_by_row(gts, missing_value=MISSING_VALUES[int])
        return counts

    def _create_matrix(self, path, shape, dtype, fillvalue):
        arrays = self._hArrays
        array_name = posixpath.basename(path)
        if not array_name:
            msg = 'The path should include a array name: ' + path
            raise ValueError(msg)

        try:
            arrays[path]
            msg = 'The array already exists: ' + path
            raise ValueError(msg)
        except KeyError:
            pass
        array = numpy.full(shape, fillvalue, dtype)
        arrays[path] = array
        return array
