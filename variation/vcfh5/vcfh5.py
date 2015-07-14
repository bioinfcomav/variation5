
from itertools import zip_longest
import warnings
import posixpath

import numpy
import h5py
from collections import OrderedDict

from variation import SNPS_PER_CHUNK, DEF_DSET_PARAMS, MISSING_VALUES
from variation.iterutils import first
from variation.matrix.stats import counts_by_row
from numpy import dtype

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
    info_grp_name = b'/variations/info'
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
        path = posixpath.join(info_grp_name, field)
        matrix = _create_matrix(hdf5, path, **kwargs)
        info_matrices[path] = matrix
    return info_matrices


def _prepate_call_datasets(vcf, hdf5, vars_in_chunk):
    n_samples = len(vcf.samples)
    ploidy = vcf.ploidy
    fmt_fields = vcf.metadata['FORMAT'].keys()
    fmt_fields = set(fmt_fields).difference(vcf.ignored_fields)
    if vcf.kept_fields:
        fmt_fields = fmt_fields.intersection(vcf.kept_fields)
    fmt_fields = list(fmt_fields)

    fmt_matrices = OrderedDict()
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
                    if not z_axes_size:
                        msg = 'This field is empty in the preread SNPs: '
                        msg += field.decode("utf-8")
                        warnings.warn(msg, RuntimeWarning)
                        continue

        size = [vars_in_chunk, n_samples, z_axes_size]
        maxshape = (None, n_samples, z_axes_size)
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
        path = posixpath.join(b'/calls', field)
        matrix = _create_matrix(hdf5, path, **kwargs)
        fmt_matrices[path] = matrix

    return fmt_matrices


def _create_matrix(var_matrices, path, **kwargs):
    fillvalue = MISSING_VALUES[kwargs['dtype']]
    kwargs['fillvalue'] = fillvalue

    try:
        matrix = var_matrices.create_matrix(path, **kwargs)
    except TypeError:
        dtype = kwargs['dtype']
        fillvalue = MISSING_VALUES[dtype]
        shape = kwargs['shape']
        matrix = var_matrices.create_matrix(path, dtype=dtype,
                                            fillvalue=fillvalue,
                                            shape=shape)
    return matrix


def _prepare_variation_datasets(vcf, hdf5, vars_in_chunk):

    meta = vcf.metadata['VARIATIONS']
    var_grp_name = b'variations'
    one_item_fields = [b'chrom', b'pos', b'id', b'ref', b'qual']
    multi_item_fields = [b'alt']
    fields = one_item_fields + multi_item_fields
    var_matrices = OrderedDict()
    for field in fields:
        str_field = field.decode('utf-8')
        if field in one_item_fields:
            size = [vars_in_chunk]
            maxshape = (None,)  # is resizable, we can add SNPs
            chunks=(vars_in_chunk,)
        else:
            y_axes_size = vcf.max_field_lens[str_field]
            if not y_axes_size:
                msg = 'No max size for field. Try prereading some SNPs: '
                msg += field
                raise RuntimeError(msg)
            size = [vars_in_chunk, y_axes_size]
            maxshape = (None, y_axes_size)  # is resizable, we can add SNPs
            chunks=(vars_in_chunk,  y_axes_size)

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


def _prepare_filter_datasets(vcf, hdf5, vars_in_chunk):

    filter_grp_name = b'/variations/filter'
    meta = vcf.metadata['FILTER']
    filter_fields = set(meta.keys()).difference(vcf.ignored_fields)
    filter_fields = list(filter_fields)
    if not filter_fields:
        return []

    filter_fields.append(b'no_filters')
    filter_matrices = OrderedDict()
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
        path = posixpath.join(filter_grp_name,field)
        matrix = _create_matrix(hdf5, path, **kwargs)
        filter_matrices[path] = matrix

    return filter_matrices


def _expand_list_to_size(items, desired_size, missing):
    extra_empty_items = [missing[0]] * (desired_size - len(items))
    items.extend(extra_empty_items)


def _dset_metadata_from_matrix(mat, vars_in_chunk):
    shape = mat.shape
    dtype = mat.dtype
    if hasattr(mat, 'chunks'):
        chunks = mat.chunks
        maxshape = mat.maxshape
    else:
        chunks = list(shape)
        chunks[0] = vars_in_chunk
        chunks = tuple(chunks)
        maxshape = list(shape)
        maxshape[0] = None
        maxshape = tuple(maxshape)
    return shape, dtype, chunks, maxshape


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


# def _create_arrays_from_chunks(arrays_chunks, vars_in_chunk):
#     for path, matrix in dset_chunks.items():
#         grp_name, name = posixpath.split(path)
#         try:
#             grp = hdf5[grp_name]
#         except KeyError:
#             grp = hdf5.create_group(grp_name)
#         shape = list(matrix.shape)
#         shape[0] = 0    # No snps yet
#         shape, dtype, chunks, maxshape = _dset_metadata_from_matrix(matrix,
#                                                                   vars_in_chunk)
#         dset = grp.create_dataset(name, shape=shape,
#                                   dtype=dtype,
#                                   chunks=chunks,
#                                   maxshape=maxshape)
#         dsets[path] = dset
#     return dsets



def _write_vars_from_vcf(vcf, hdf5, vars_in_chunk, kept_fields=None,
                         ignored_fields=None):
    snps = vcf.variations
    log = {'data_no_fit': {},
           'variations_processed': 0}

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
                grp = 'FORMAT'

            matrix = hdf5[path]
            #print(matrix)
            # resize the dataset to fit the new chunk
            size = matrix.shape
            new_size = list(size)
            new_size[0] = vars_in_chunk * (chunk_i + 1)
            matrix.resize(new_size)

            field = posixpath.basename(path)
            str_field = field.decode('utf-8')

            if grp == 'FILTER':
                missing = False
            else:
                try:
                    dtype= vcf.metadata[grp][str_field]['dtype']
                except KeyError:
                    dtype= vcf.metadata[grp][field]['dtype']
                missing = MISSING_VALUES[dtype]

            if len(size) == 3 and field != b'GT':
                missing = [missing] * size[2]
            elif len(size) == 2:
                missing = [missing] * size[1]
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
                    info_data = info_data.get(field, None)
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
                            matrix[snp_n] = info_data
                        except TypeError as error:
                            if 'broadcast' in str(error):
                                if field not in log['data_no_fit']:
                                    log['data_no_fit'][field] = 0
                                log['data_no_fit'][field] += 1
                    #TODO: FIXME=1
                elif grp == 'VARIATIONS':
                    if field == b'chrom':
                        item = snp[0]
                    elif field == b'pos':
                        item = snp[1]
                    elif field == b'id':
                        item = snp[2]
                    elif field == b'ref':
                        item = snp[3]
                    elif field == b'alt':
                        item = snp[4]
                        _expand_list_to_size(item, size[1], [b''])
                    elif field == b'qual':
                        item = snp[5]
                    if item is not None:
                        try:
                            matrix[snp_n] = item
                        except TypeError as error:
                            if 'broadcast' in str(error) and field == 'alt':
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
                            _expand_list_to_size(call_sample_data, size[2],
                                                 missing)

                        if call_sample_data is not None:
                            try:
                                matrix[snp_n] = call_sample_data
                            except TypeError as error:
                                if 'broadcast' in str(error):
                                    if field not in log['data_no_fit']:
                                        log['data_no_fit'][field] = 0
                                    log['data_no_fit'][field] += 1
            first_field = False
    # we have to remove the empty snps from the last chunk
    for path in paths:
        matrix = hdf5[path]

        size = matrix.shape
        new_size = list(size)
        snp_n = snp_i + chunk_i * vars_in_chunk
        new_size[0] = snp_n
        if hasattr(hdf5, 'resize'):
            matrix.resize(new_size)
        else:
            matrix = numpy.resize(matrix, new_size)

    if hasattr(hdf5, 'flush'):
        hdf5.flush()
    print(hdf5.hArrays.keys())
    return log


class VcfH5:
    def __init__(self, fpath, mode, vars_in_chunk=SNPS_PER_CHUNK):
        self._fpath = fpath
        if mode not in ('r', 'w'):
            msg = 'mode should be r or w'
            raise ValueError(msg)
        elif mode == 'w':
            mode = 'w-'
        self.mode = mode
        self.h5file = h5py.File(fpath, mode)
        self._vars_in_chunk = vars_in_chunk


    def write_vars_from_vcf(self, vcf):
        return _write_vars_from_vcf(vcf, self, self._vars_in_chunk)


    def iterate_chunks(self, kept_fields=None, ignored_fields=None):

        if kept_fields is not None and ignored_fields is not None:
            msg = 'kept_fields and ignored_fields can not be set at the same time'
            raise ValueError(msg)

        # We read the hdf5 file to keep the datasets metadata
        dsets = {}
        for grp in self.h5file.values():
            if not isinstance(grp, h5py.Group):
                # We're assuming no dsets in root
                continue
            for name, item in grp.items():
                if isinstance(item, h5py.Dataset):
                    dset = item
                    key = posixpath.join(grp.name, name)
                    dsets[key] = dset
                else:
                    grp = item
                    for sname, dset in grp.items():
                        if isinstance(dset, h5py.Dataset):
                            key = posixpath.join(grp.name, sname)
                            dsets[key] = dset

        # We remove the unwanted fields
        fields = dsets.keys()
        if kept_fields:
            fields = set(kept_fields).intersection(fields)
        if ignored_fields:
            fields = set(fields).difference(ignored_fields)
        dsets = {field: dsets[field] for field in fields}

        # how many snps are per chunk?
        one_dset = dsets[first(dsets.keys())]
        chunk_size = one_dset.chunks[0]
        nsnps = one_dset.shape
        if isinstance(nsnps, (tuple, list)):
            nsnps = nsnps[0]

        # Now we can yield the chunks
        for start in range(0, nsnps, chunk_size):
            stop = start + chunk_size
            if stop > nsnps:
                stop = nsnps
            chunks = {path: dset[start:stop] for path, dset in dsets.items()}
            yield chunks

    def __getitem__(self, path):
        return self.h5file[path]

    def write_chunks(self, chunks, kept_fields=None, ignored_fields=None):
        dsets = None
        current_snp_index = 0
        for dsets_chunks in chunks:
            if dsets is None:
                dsets = _create_dsets_from_chunks(self.h5file, dsets_chunks,
                                                  self._vars_in_chunk)

            # check all chunks have the same number of snps
            nsnps = [chunk.data.shape[0] for chunk in dsets_chunks.values()]
            num_snps = nsnps[0]
            assert all(num_snps == nsnp for nsnp in nsnps)

            for dset_name, dset_chunk in dsets_chunks.items():
                dset = dsets[dset_name]
                start = current_snp_index
                stop = current_snp_index + num_snps
                # the dataset should fit the new data
                size = dset.shape
                new_size = list(size)
                new_size[0] = stop
                dset.resize(new_size)

                dset[start:stop] = dset_chunk.data

            current_snp_index += num_snps
        self.h5file.flush()

    def flush(self):
        self.h5file.flush()

    def close(self):
        self.h5file.close()

    @property
    def num_variations(self):
        return self['/variations/chrom'].shape[0]

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


    def create_matrix(self, path, *args, **kwargs):
        hdf5 = self.h5file
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
        args = list(args)
        args.insert(0, dset_name)
        dset = group.create_dataset(*args, **kwargs)

        return dset

def select_dset_from_chunks(chunks, dset_path):
    return (chunk[dset_path] for chunk in chunks)

#No hay que pasarle ningun path porque se guarda en memoria
class VcfArrays:
    def __init__(self, vars_in_chunk=SNPS_PER_CHUNK):
        self._vars_in_chunk = vars_in_chunk
        self.hArrays = {}

    def write_vars_from_vcf(self, vcf):
        return _write_vars_from_vcf(vcf, self, self._vars_in_chunk)


#     def iterate_chunks(self, kept_fields=None, ignored_fields=None):
#
#         if kept_fields is not None and ignored_fields is not None:
#             msg = 'kept_fields and ignored_fields can not be set at the same time'
#             raise ValueError(msg)
#
#         # We read the hdf5 file to keep the datasets metadata
#         dsets = {}
#         for grp in self.h5file.values():
#             if not isinstance(grp, h5py.Group):
#                 # We're assuming no dsets in root
#                 continue
#             for name, item in grp.items():
#                 if isinstance(item, h5py.Dataset):
#                     dset = item
#                     key = posixpath.join(grp.name, name)
#                     dsets[key] = dset
#                 else:
#                     grp = item
#                     for sname, dset in grp.items():
#                         if isinstance(dset, h5py.Dataset):
#                             key = posixpath.join(grp.name, sname)
#                             dsets[key] = dset
#
#         # We remove the unwanted fields
#         fields = dsets.keys()
#         if kept_fields:
#             fields = set(kept_fields).intersection(fields)
#         if ignored_fields:
#             fields = set(fields).difference(ignored_fields)
#         dsets = {field: dsets[field] for field in fields}
#
#         # how many snps are per chunk?
#         one_dset = dsets[first(dsets.keys())]
#         chunk_size = one_dset.chunks[0]
#         nsnps = one_dset.shape
#         if isinstance(nsnps, (tuple, list)):
#             nsnps = nsnps[0]
#
#         # Now we can yield the chunks
#         for start in range(0, nsnps, chunk_size):
#             stop = start + chunk_size
#             if stop > nsnps:
#                 stop = nsnps
#             chunks = {path: dset[start:stop] for path, dset in dsets.items()}
#             yield chunks

    def __getitem__(self, path):
        return self.hArrays[path]

#     def write_chunks(self, chunks, kept_fields=None, ignored_fields=None):
#         arrays = None
#         current_snp_index = 0
#         for arrays_chunks in chunks:
#             if arrays is None:
#                 arrays = _create_dsets_from_chunks(self.h5file, dsets_chunks,
#                                                   self._vars_in_chunk)
#
#             # check all chunks have the same number of snps
#             nsnps = [chunk.data.shape[0] for chunk in dsets_chunks.values()]
#             num_snps = nsnps[0]
#             assert all(num_snps == nsnp for nsnp in nsnps)
#
#             for dset_name, dset_chunk in dsets_chunks.items():
#                 dset = dsets[dset_name]
#                 start = current_snp_index
#                 stop = current_snp_index + num_snps
#                 # the dataset should fit the new data
#                 size = dset.shape
#                 new_size = list(size)
#                 new_size[0] = stop
#                 dset.resize(new_size)
#
#                 dset[start:stop] = dset_chunk.data
#
#             current_snp_index += num_snps
#         self.h5file.flush()


    @property
    def num_variations(self):
        return self.hArrays['/variations/chrom'].shape[0]

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


    def create_matrix(self, path, shape, dtype, fillvalue):
        hArrays = self.hArrays
        array_name = posixpath.basename(path)
        if not array_name:
            msg = 'The path should include a array name: ' + path
            raise ValueError(msg)

        try:
            hArrays[path]
            msg = 'The array already exists: ' + path
            raise ValueError(msg)
        except KeyError:
            pass
        array = numpy.full(shape, fillvalue, dtype)
        hArrays[path] = array
        return array
