
from itertools import zip_longest
import warnings
import posixpath

import numpy
import h5py

from variation import SNPS_PER_CHUNK, DEF_DSET_PARAMS, MISSING_GT
from variation.vcf import _missing_val, _filling_val
from variation.iterutils import first
from variation.matrix.stats import counts_by_row

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
    var_grp = hdf5['variations']
    info_grp = var_grp.create_group('info')

    info_fields = meta.keys()
    info_fields = set(info_fields).difference(vcf.ignored_fields)
    if vcf.kept_fields:
        info_fields = info_fields.intersection(vcf.kept_fields)
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

    fmt_fields = vcf.metadata['FORMAT'].keys()
    fmt_fields = set(fmt_fields).difference(vcf.ignored_fields)
    if vcf.kept_fields:
        fmt_fields = fmt_fields.intersection(vcf.kept_fields)
    fmt_fields = list(fmt_fields)

    empty_fields = set()
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
                        empty_fields.add(field)
                        continue

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

    if empty_fields:
        fmt_fields = list(set(fmt_fields).difference(empty_fields))
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
            if not y_axes_size:
                msg = 'No max size for field. Try prereading some SNPs: '
                msg += field
                raise RuntimeError(msg)
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
        snps = vcf.variations

        log = {'data_no_fit': {},
               'variations_processed': 0}

        hdf5 = self.h5file
        vars_in_chunk = self._vars_in_chunk

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
                                if isinstance(info_data, (list, tuple)):
                                    if len(info_data) != 1:
                                        if field not in log['data_no_fit']:
                                            log['data_no_fit'][field] = 0
                                        log['data_no_fit'][field] += 1
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

        hdf5.flush()

        return log

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

    def __getitem__(self, index):
        return self.h5file[index]

    def write_chunks(self, chunks):
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
            chunk_counts = counts_by_row(gt_chunk, missing_value=MISSING_GT)
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

def select_dset_from_chunks(chunks, dset_path):
    return (chunk[dset_path] for chunk in chunks)
