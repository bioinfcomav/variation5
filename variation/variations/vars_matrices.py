
from itertools import zip_longest
import warnings
import posixpath
import json
import copy

import numpy
from numpy import dtype
import h5py
from collections import OrderedDict

from variation import SNPS_PER_CHUNK, DEF_DSET_PARAMS, MISSING_VALUES
from variation.iterutils import first
from variation.matrix.stats import counts_by_row
from variation.matrix.methods import (append_matrix, is_dataset, resize_matrix)
# Missing docstring
# pylint: disable=C0111


def _grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


TYPES = {'int16': numpy.int16,
         'int32': numpy.int32,
         'float16': numpy.float16}


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
    info_grp_name = '/variations/info'
    info_fields = meta.keys()
    info_fields = set(info_fields).difference(vcf.ignored_fields)
    if vcf.kept_fields:
        info_fields = info_fields.intersection(vcf.kept_fields)
    info_fields = list(info_fields)

    info_matrices = OrderedDict()
    for field in info_fields:
        meta_fld = meta[field]
        dtype = _numpy_dtype(meta_fld['dtype'], field,
                             vcf.max_field_str_lens)

        if field not in vcf.max_field_lens['INFO']:
            # We assume that it is not used by any SNP
            continue

        y_axes_size = vcf.max_field_lens['INFO'][field]
        if not y_axes_size:
            msg = 'This field is empty in the preread SNPs: '
            msg += field.decode("utf-8")
            warnings.warn(msg, RuntimeWarning)
            continue

        if y_axes_size == 1:
            size = (vars_in_chunk,)
            maxshape = (None,)
            chunks = (vars_in_chunk,)
        else:
            size = [vars_in_chunk, y_axes_size]
            maxshape = (None, y_axes_size)
            chunks = (vars_in_chunk, y_axes_size)

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['shape'] = size
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        path = posixpath.join(info_grp_name, str(field, 'utf-8'))
        matrix = _create_matrix(hdf5, path, **kwargs)
        info_matrices[path] = matrix
    return info_matrices


def _prepate_call_datasets(vcf, hdf5, vars_in_chunk):
    n_samples = len(vcf.samples)
    ploidy = vcf.ploidy
    fmt_fields = vcf.metadata['CALLS'].keys()
    fmt_fields = set(fmt_fields).difference(vcf.ignored_fields)
    if vcf.kept_fields:
        fmt_fields = fmt_fields.intersection(vcf.kept_fields)
    fmt_fields = list(fmt_fields)

    fmt_matrices = OrderedDict()
    for field in fmt_fields:
        fmt = vcf.metadata['CALLS'][field]
        if field == b'GT':
            z_axes_size = ploidy
            dtype = numpy.int8
        else:
            dtype = _numpy_dtype(fmt['dtype'], field, vcf.max_field_str_lens)
            if isinstance(fmt['Number'], int):
                z_axes_size = fmt['Number']
            else:
                if field == b'GT':
                    z_axes_size = vcf.ploidy
                else:
                    z_axes_size = vcf.max_field_lens['CALLS'][field]
                    if not z_axes_size:
                        msg = 'This field is empty in the preread SNPs: '
                        #msg += field.decode("utf-8")
                        msg += 'CALLS/' + field.decode("utf-8")
                        warnings.warn(msg, RuntimeWarning)
                        continue

        size = [vars_in_chunk, n_samples, z_axes_size]
        maxshape = (None, None, z_axes_size)
        chunks = (vars_in_chunk, n_samples, z_axes_size)

        # If the last dimension only has one of len we can work with only
        # two dimensions (variations x samples)
        if size[-1] == 1:
            size = size[:-1]
            maxshape = maxshape[:-1]
            chunks = chunks[:-1]

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['shape'] = size
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        path = posixpath.join('/calls', str(field, 'utf-8'))
        matrix = _create_matrix(hdf5, path, **kwargs)
        fmt_matrices[path] = matrix

    return fmt_matrices


def _create_matrix(var_matrices, path, **kwargs):
    fillvalue = MISSING_VALUES[kwargs['dtype']]
    kwargs['fillvalue'] = fillvalue

    try:
        matrix = var_matrices._create_matrix(path, **kwargs)
    except TypeError:
        dtype = kwargs['dtype']
        fillvalue = MISSING_VALUES[dtype]
        shape = kwargs['shape']
        matrix = var_matrices._create_matrix(path, dtype=dtype,
                                             fillvalue=fillvalue,
                                             shape=shape)
    return matrix


def _prepare_variation_datasets(vcf, hdf5, vars_in_chunk):

    meta = vcf.metadata['VARIATIONS']
    var_grp_name = '/variations'
    one_item_fields = ['chrom', 'pos', 'id', 'ref', 'qual']
    multi_item_fields = ['alt']
    fields = one_item_fields + multi_item_fields
    var_matrices = OrderedDict()
    for field in fields:
        str_field = _to_str(field)
        if field in one_item_fields:
            size = [vars_in_chunk]
            maxshape = (None,)  # is resizable, we can add SNPs
            chunks = (vars_in_chunk,)
        else:
            y_axes_size = vcf.max_field_lens[str_field]
            if not y_axes_size:
                msg = 'No max size for field. Try prereading some SNPs: '
                msg += field
                raise RuntimeError(msg)
            size = [vars_in_chunk, y_axes_size]
            maxshape = (None, y_axes_size)  # is resizable, we can add SNPs
            chunks = (vars_in_chunk,  y_axes_size)

        dtype = meta[str_field]['dtype']
        dtype = _numpy_dtype(meta[str_field]['dtype'], field,
                             vcf.max_field_str_lens)

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['shape'] = size
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        path = posixpath.join(var_grp_name, field)
        matrix = _create_matrix(hdf5, path, **kwargs)
        var_matrices[path] = matrix

    return var_matrices


def _to_str(str_or_byte):
    try:
        str_ = str(str_or_byte, 'utf-8')
    except TypeError:
        str_ = str_or_byte
    return str_


def _prepare_filter_datasets(vcf, hdf5, vars_in_chunk):

    filter_grp_name = '/variations/filter'
    meta = vcf.metadata['FILTER']
    filter_fields = set(meta.keys()).difference(vcf.ignored_fields)
    filter_fields = list(filter_fields)

    filter_matrices = OrderedDict()
    if not filter_fields:
        return filter_matrices

    filter_fields.append('no_filters')
    for field in filter_fields:
        dtype = numpy.bool_

        size = (vars_in_chunk,)
        maxshape = (None,)
        chunks = (vars_in_chunk,)

        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['shape'] = size
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        path = posixpath.join(filter_grp_name, _to_str(field))
        matrix = _create_matrix(hdf5, path, **kwargs)
        filter_matrices[path] = matrix

    return filter_matrices


_MISSING_ITEMS = {}


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


def _create_dsets_from_chunks(hdf5, dset_chunks, vars_in_chunk):
    dsets = {}
    for path, matrix in dset_chunks.items():
        grp_name, name = posixpath.split(path)
        try:
            grp = hdf5[grp_name]
        except KeyError:
            grp = hdf5.create_group(grp_name)
        shape = list(matrix.shape)
        shape[0] = 0    # No snps yet
        shape, dtype, chunks, maxshape = _dset_metadata_from_matrix(matrix,
                                                                    vars_in_chunk)
        dset = grp.create_dataset(name, shape=shape,
                                  dtype=dtype,
                                  chunks=chunks,
                                  maxshape=maxshape)
        dsets[path] = dset

    return dsets


def _size_recur(size, item):
    if hasattr(item, '__len__') and not isinstance(item, (str, bytes)):
        is_list = True
    else:
        is_list = False
    if is_list:
        size.append(len(item))
        _size_recur(size, item[0])


def _size(item):
    size = []
    _size_recur(size, item)

    if not size:
        return None
    else:
        return size


def _create_slice(snp_n, item):
    size = _size(item)
    if size is None:
        return snp_n
    else:
        slice_ = [snp_n]
        slice_.extend([slice(dim_len) for dim_len in size])
        return tuple(slice_)


def _prepare_metadata(vcf_metadata):
    unwanted_fields = ['dtype', 'type_cast']
    groups = ['INFO', 'FILTER', 'CALLS', 'OTHER']
    meta = {}
    for group in groups:
        for field, field_meta in vcf_metadata[group].items():
            for ufield in unwanted_fields:
                if ufield in field_meta:
                    del field_meta[ufield]
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


def _prepare_gt_dataset(csv, hdf5, vars_in_chunk):
    n_samples = len(csv.samples)
    ploidy = csv.ploidy
    field = 'GT'
    dtype = numpy.int8
    size = [vars_in_chunk, n_samples, ploidy]
    maxshape = (None, None, ploidy)
    chunks = (vars_in_chunk, n_samples, ploidy)
    # If the last dimension only has one of len we can work with only
    # two dimensions (variations x samples)
    if size[-1] == 1:
        size = size[:-1]
        maxshape = maxshape[:-1]
        chunks = chunks[:-1]

    kwargs = DEF_DSET_PARAMS.copy()
    kwargs['shape'] = size
    kwargs['dtype'] = dtype
    kwargs['maxshape'] = maxshape
    kwargs['chunks'] = chunks
    path = posixpath.join('/calls', field)
    matrix = _create_matrix(hdf5, path, **kwargs)
    return {path: matrix}


def _prepare_snp_info_datasets(csv, hdf5, vars_in_chunk):
    var_grp_name = '/variations'
    one_item_fields = csv.snp_fieldnames_final
    multi_item_fields = []
    if csv.alt_field in one_item_fields:
        one_item_fields.remove(csv.alt_field)
        multi_item_fields = [csv.alt_field]
    fields = one_item_fields + multi_item_fields
    var_matrices = OrderedDict()
    for field in fields:
        if field in one_item_fields:
            size = [vars_in_chunk]
            maxshape = (None,)  # is resizable, we can add SNPs
            chunks = (vars_in_chunk,)
        else:
            y_axes_size = csv.max_alt_allele
            if not y_axes_size:
                msg = 'No max size for field. Try prereading some SNPs: '
                msg += field
                raise RuntimeError(msg)
            size = [vars_in_chunk, y_axes_size]
            maxshape = (None, y_axes_size)  # is resizable, we can add SNPs
            chunks = (vars_in_chunk,  y_axes_size)
        dtype = _numpy_dtype('str', field, {field: 20})
        if 'pos' in field:
            dtype = _numpy_dtype('int32', field, {field: 20})
        kwargs = DEF_DSET_PARAMS.copy()
        kwargs['shape'] = size
        kwargs['dtype'] = dtype
        kwargs['maxshape'] = maxshape
        kwargs['chunks'] = chunks
        path = posixpath.join(var_grp_name, field)
        matrix = _create_matrix(hdf5, path, **kwargs)
        var_matrices[path] = matrix
    return var_matrices


def put_vars_from_csv(csv, hdf5, vars_in_chunk):

    ignore_alt = csv.ignore_alt
    log = {'data_no_fit': {},
           'variations_processed': 0,
           'alt_max_detected': 0,
           'num_alt_item_descarted': 0}

    fmt_matrices = _prepare_gt_dataset(csv, hdf5, vars_in_chunk)
    var_matrices = _prepare_snp_info_datasets(csv, hdf5, vars_in_chunk)

    paths = list(var_matrices.keys())
    paths.extend(fmt_matrices.keys())

    snp_chunks = _grouper(csv, vars_in_chunk)
    for chunk_i, chunk in enumerate(snp_chunks):
        chunk = list(chunk)
        first_field = True
        for path in paths:
            if path in var_matrices:
                grp = 'VARIATIONS'
            else:
                grp = 'CALLS'
            matrix = hdf5[path]
            # resize the dataset to fit the new chunk
            size = matrix.shape
            new_size = list(size)
            new_size[0] = vars_in_chunk * (chunk_i + 1)

            resize_matrix(matrix, new_size)

            field = posixpath.basename(path)

            # We store the information
            for snp_i, snp in enumerate(chunk):
                if snp is None:
                    break
                if first_field:
                    log['variations_processed'] += 1
                #snp_n es el indice que se moverá en cada array
                snp_n = snp_i + chunk_i * vars_in_chunk
                if grp == 'VARIATIONS':
                    item = snp[field]
                    if item is not None:
                        try:
                            slice_ = _create_slice(snp_n, item)
                            if isinstance(item, list):
                                item = [x.encode('utf-8') for x in item]
                            elif 'pos' not in field:
                                item = item.encode('utf-8')
                            matrix[slice_] = item
                        except TypeError as error:
                            if 'broadcast' in str(error) and field == 'alt':
                                if not ignore_alt:
                                    msg = 'More alt alleles than expected.'
                                    msg2 = 'Expected, present: {}, {}'
                                    msg2 = msg2.format(size[1], len(item))
                                    msg += msg2
                                    msg = '\nYou might fix it prereading more'
                                    msg += ' SNPs, or passing: '
                                    msg += 'max_field_lens={'
                                    msg += '"alt":{}'.format(len(item))
                                    msg += '}\nto VCF reader'
                                    raise TypeError(msg)
                                else:
                                    log['num_alt_item_descarted'] += 1
                                    if log['alt_max_detected'] < len(item):
                                        log['alt_max_detected'] = len(item)
                                    continue
                elif grp == 'CALLS':
                    # store the calldata
                    try:
                        gts = snp['gts']
                    except TypeError:
                    # SNP is None
                        break
                    if gts is not None:

                        try:
                            slice_ = _create_slice(snp_n, gts)
                            matrix[slice_] = gts
                        except TypeError as error:
                            if 'broadcast' in str(error):
                                if field not in log['data_no_fit']:
                                    log['data_no_fit'][field] = 0
                                log['data_no_fit'][field] += 1
                        except:
                            print('snp_id', snp_i)
                            print('field', field)
                            print('failed data', gts)
                            raise
            first_field = False
    # we have to remove the empty snps from the last chunk

    for path in paths:
        matrix = hdf5[path]
        size = matrix.shape
        new_size = list(size)
        snp_n = snp_i + chunk_i * vars_in_chunk
        new_size[0] = snp_n

        resize_matrix(matrix, new_size)
    hdf5._set_samples(csv.samples)

    if hasattr(hdf5, 'flush'):
        hdf5.flush()
    return log


def _put_vars_from_vcf(vcf, hdf5, vars_in_chunk, kept_fields=None,
                       ignored_fields=None):

    ignore_alt = vcf.ignore_alt
    snps = vcf.variations
    log = {'data_no_fit': {},
           'variations_processed': 0,
           'alt_max_detected' : 0,
           'num_alt_item_descarted': 0}

    fmt_matrices = _prepate_call_datasets(vcf, hdf5, vars_in_chunk)
    info_matrices = _prepare_info_datasets(vcf, hdf5, vars_in_chunk)
    filter_matrices = _prepare_filter_datasets(vcf, hdf5, vars_in_chunk)
    var_matrices = _prepare_variation_datasets(vcf, hdf5, vars_in_chunk)

    paths = list(var_matrices.keys())
    paths.extend(fmt_matrices.keys())
    paths.extend(info_matrices.keys())
    paths.extend(filter_matrices.keys())

    snp_chunks = _grouper(snps, vars_in_chunk)
    for chunk_i, chunk in enumerate(snp_chunks):
        chunk = list(chunk)
        first_field = True
        for path in paths:
            if path in var_matrices:
                grp = 'VARIATIONS'
            elif path in info_matrices:
                grp = 'INFO'
            elif path in filter_matrices:
                grp = 'FILTER'
            else:
                grp = 'CALLS'

            matrix = hdf5[path]
            # resize the dataset to fit the new chunk
            size = matrix.shape
            new_size = list(size)
            new_size[0] = vars_in_chunk * (chunk_i + 1)

            resize_matrix(matrix, new_size)

            field = posixpath.basename(path)
            field = _to_str(field)
            byte_field = field.encode('utf-8')

            if grp == 'FILTER':
                missing_val = False
            else:
                try:
                    dtype = vcf.metadata[grp][field]['dtype']
                except KeyError:
                    dtype = vcf.metadata[grp][byte_field]['dtype']
                missing_val = MISSING_VALUES[dtype]

            # We store the information
            for snp_i, snp in enumerate(chunk):
                try:
                    gt_data = snp[-1]
                except TypeError:
                    # SNP is None
                    break
                if first_field:
                    log['variations_processed'] += 1
                #snp_n es el indice que se moverá en cada array
                snp_n = snp_i + chunk_i * vars_in_chunk
                if grp == 'FILTER':
                    data = snp[6]
                    if field == 'no_filters':
                        data = data is None
                    else:
                        data = field in data
                elif grp == 'INFO':
                    info_data = snp[7]
                    info_data = info_data.get(byte_field, None)
                    if info_data is not None:
                        if len(size) == 1:
                            # we're expecting one item or a list with one item
                            if isinstance(info_data, (list, tuple)):
                                if len(info_data) != 1:
                                    if field not in log['data_no_fit']:
                                        log['data_no_fit'][field] = 0
                                    log['data_no_fit'][field] += 1
                                info_data = info_data[0]

                        try:
                            slice_ = _create_slice(snp_n, info_data)
                            matrix[slice_] = info_data
                        except TypeError as error:
                            if 'broadcast' in str(error):
                                if field not in log['data_no_fit']:
                                    log['data_no_fit'][field] = 0
                                log['data_no_fit'][field] += 1
                    #TODO: FIXME=1
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
                        #_expand_list_to_size(item, size[1], b'')
                    elif field == 'qual':
                        item = snp[5]
                    if item is not None:
                        try:
                            slice_ = _create_slice(snp_n, item)
                            matrix[slice_] = item
                        except TypeError as error:
                            if 'broadcast' in str(error) and field == 'alt':
                                if not ignore_alt:
                                    msg = 'More alt alleles than expected.'
                                    msg2 = 'Expected, present: {}, {}'
                                    msg2 = msg2.format(size[1], len(item))
                                    msg += msg2
                                    msg = '\nYou might fix it prereading more'
                                    msg += ' SNPs, or passing: '
                                    msg += 'max_field_lens={'
                                    msg += '"alt":{}'.format(len(item))
                                    msg += '}\nto VCF reader'
                                    raise TypeError(msg)
                                else:
                                    log['num_alt_item_descarted'] += 1
                                    if log['alt_max_detected'] < len(item):
                                        log['alt_max_detected'] = len(item)
                                    continue
                elif grp == 'CALLS':
                    # store the calldata
                    gt_data = dict(gt_data)
                    call_sample_data = gt_data.get(byte_field, None)
                    if call_sample_data is not None:
                        if len(size) == 2:
                            # we're expecting a single item or a list with one item
                            try:
                                one_element = first(filter(lambda x: x is not None,
                                                           call_sample_data))
                            except ValueError:
                                one_element = None
                            if isinstance(one_element, (list, tuple)):
                                # We have a list in each item
                                # we're assuming that all items have length 1
                                if max([len(cll) for cll in call_sample_data
                                        if cll is not None]) == 1:
                                    call_sample_data = [missing_val if item is None
                                                        else item[0]
                                                        for item in call_sample_data]
                                else:
                                    if field not in log['data_no_fit']:
                                        log['data_no_fit'][field] = 0
                                    log['data_no_fit'][field] += 1
                                    call_sample_data = None
                            else:
                                call_sample_data = [missing_val if item is None
                                                    else item
                                                    for item in call_sample_data]
                        #In case of GL and GT [[[1,2,3],[1,2,3],[1,2,3], none]]
                        elif len(size) > 2:
                            call_sample_data = [[missing_val]*size[-1]
                                                if item is None else item
                                                for item in call_sample_data]
                        if call_sample_data is not None:

                            try:
                                slice_ = _create_slice(snp_n, call_sample_data)
                                matrix[slice_] = call_sample_data
                            except TypeError as error:
                                if 'broadcast' in str(error):
                                    if field not in log['data_no_fit']:
                                        log['data_no_fit'][field] = 0
                                    log['data_no_fit'][field] += 1
                            except:
                                print('snp_id', snp_i)
                                print('field', field)
                                print('failed data', call_sample_data)
                                raise
            first_field = False
    # we have to remove the empty snps from the last chunk

    for path in paths:
        matrix = hdf5[path]
        size = matrix.shape
        new_size = list(size)
        snp_n = snp_i + chunk_i * vars_in_chunk
        new_size[0] = snp_n

        resize_matrix(matrix, new_size)
    metadata = _prepare_metadata(vcf.metadata)
    hdf5._set_metadata(metadata)

    samples = [sample.decode('utf-8') for sample in vcf.samples]
    hdf5._set_samples(samples)

    if hasattr(hdf5, 'flush'):
        hdf5.flush()
    return log


class _VariationMatrices():
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

    def put_chunks(self, chunks, kept_fields=None, ignored_fields=None):
        matrices = None
        for mats_chunks in chunks:
            if matrices is None:
                matrices = self._create_mats_from_chunks(mats_chunks)
                self._set_metadata(mats_chunks.metadata)
                self._set_samples(mats_chunks.samples)
                continue
            # check all chunks have the same number of snps
            nsnps = [mats_chunks[path].data.shape[0]
                     for path in mats_chunks.keys()]
            num_snps = nsnps[0]
            assert all(num_snps == nsnp for nsnp in nsnps)

            for path in mats_chunks.keys():
                dset_chunk = mats_chunks[path]
                dset = matrices[path]
                append_matrix(dset, dset_chunk)

        if hasattr(self, 'flush'):
            self._h5file.flush()

    def iterate_chunks(self, kept_fields=None, ignored_fields=None,
                       chunk_size=None):

        if kept_fields is not None and ignored_fields is not None:
            msg = 'kept_fields and ignored_fields can not be set at the same time'
            raise ValueError(msg)
        # We read the hdf5 file to keep the datasets metadata
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

    @property
    def num_variations(self):
        try:
            one_path = first(self.keys())
        except ValueError:
            return 0
        one_mat = self[one_path]
        return one_mat.shape[0]


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
    def __init__(self, fpath, mode, vars_in_chunk=SNPS_PER_CHUNK):
        self._fpath = fpath
        if mode not in ('r', 'w', 'r+'):
            msg = 'mode should be r or w'
            raise ValueError(msg)
        elif mode == 'w':
            mode = 'w-'
        self.mode = mode
        self._h5file = h5py.File(fpath, mode)
        self._vars_in_chunk = vars_in_chunk

    def put_vars_from_vcf(self, vcf):
        return _put_vars_from_vcf(vcf, self, self._vars_in_chunk)

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
    def __init__(self, vars_in_chunk=SNPS_PER_CHUNK):
        self._vars_in_chunk = vars_in_chunk
        self._hArrays = {}
        self._metadata = {}
        self._samples = []

    def put_vars_from_vcf(self, vcf):
        return _put_vars_from_vcf(vcf, self, self._vars_in_chunk)

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
