import posixpath
import json
import copy
from collections import Counter

import numpy
import h5py


from variation import (SNPS_PER_CHUNK, MISSING_VALUES, DEF_DSET_PARAMS,
                       MISSING_INT)
from variation.iterutils import first, group_items
from variation.matrix.stats import counts_by_row
from variation.matrix.methods import append_matrix, is_dataset
from variation.variations.stats import GT_FIELD
from variation.variations.index import PosIndex
from variation.gt_writers.vcf import write_vcf

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
                elif path == '/calls/DP':
                    number_dims = 1

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


class _VariationMatrices():
    def __init__(self, vars_in_chunk=SNPS_PER_CHUNK,
                 ignore_overflows=False, ignore_undefined_fields=False,
                 kept_fields=None, ignored_fields=None):
        self._vars_in_chunk = vars_in_chunk
        self.ignore_overflows = ignore_overflows
        self.ignore_undefined_fields = ignore_undefined_fields
        self.kept_fields = kept_fields
        self.ignored_fields = ignored_fields
        self._index = None

    @property
    def ploidy(self):
        if self[GT_FIELD].shape[0] == 0:
            return None
        else:
            return self[GT_FIELD].shape[2]

    def _create_matrix_from_matrix(self, path, matrix):

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
            matrix = self._create_matrix_from_matrix(path, mat_chunk)
            matrices[path] = matrix
        return matrices

    def write_vcf(self, vcf_fhand):
        write_vcf(self, vcf_fhand)
#         for line in _to_vcf(self):
#             vcf_fhand.write(line + '\n')

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
        if chunks is None:
            return

        for chunk in chunks:
            if chunk.num_variations == 0:
                continue
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

        self._index = None

        if hasattr(self, 'flush'):
            self._h5file.flush()

    def get_chunk(self, index, kept_fields=None, ignored_fields=None):

        paths = self._filter_fields(kept_fields=kept_fields,
                                    ignored_fields=ignored_fields)

        dsets = {field: self[field] for field in paths}

        var_array = None
        for path, dset in dsets.items():
            if var_array is None:
                try:
                    matrix = dset[index, ...]
                except UnboundLocalError:
                    # This is a workaround for an error in h5py
                    if (isinstance(index, numpy.ndarray) and
                        numpy.all(index == False)):
                        matrix = numpy.array([])
                    else:
                        raise
                var_array = VariationsArrays(vars_in_chunk=matrix.shape[0])
            try:
                matrix = dset[index, ...]
            except UnboundLocalError:
                # This is a workaround for an error in h5py
                if (isinstance(index, numpy.ndarray) and
                    numpy.all(index == False)):
                    matrix = numpy.array([])
                else:
                    raise
            var_array[path] = matrix

        if var_array is None:
            var_array = self.__class__()

        var_array._set_metadata(self.metadata)
        var_array._set_samples(self.samples)
        return var_array

    def get_genome_chunk(self, chrom, start, end):
        # with index
        # bisect
        pass

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

    def _create_iterate_chunk_slices(self, chunk_size, random_sample_rate=1):
        nsnps = self.num_variations
        for start in range(0, nsnps, chunk_size):

            stop = start + chunk_size
            if stop > nsnps:
                stop = nsnps
            if random_sample_rate == 1:
                slice_ = slice(start, stop)
            else:
                num_vars = stop - start
                num_vars_keep = round(num_vars * random_sample_rate)
                if not num_vars_keep:
                    continue
                slice_ = numpy.random.choice(num_vars, num_vars_keep,
                                             replace=False)
                slice_.sort()
                slice_ += start

                if len(slice_) == 1:
                    slice_ = slice(slice_[0], slice_[0] + 1)
            yield slice_

    def iterate_chunks(self, kept_fields=None, ignored_fields=None,
                       chunk_size=None, random_sample_rate=1):
        if chunk_size is None:
            chunk_size = self._vars_in_chunk

        slices = self._create_iterate_chunk_slices(chunk_size=chunk_size,
                                                   random_sample_rate=random_sample_rate)
        for slice_ in slices:
            yield self.get_chunk(slice_, kept_fields=kept_fields,
                                 ignored_fields=ignored_fields)

    @property
    def pos_index(self):
        if self._index is None:
            self._index = PosIndex(self)
        return self._index

    def iterate_wins(self, win_size, win_step=None, kept_fields=None,
                     ignored_fields=None, chroms=None):
        if win_step is None:
            win_step = win_size
        index = self.pos_index

        if chroms is None:
            chroms = index.chroms
        chrom_mat = self['/variations/chrom']
        for chrom in chroms:
            chrom_start, chrom_end = index.get_chrom_range_pos(chrom)
            pos = chrom_start
            while True:
                if pos > chrom_end:
                    break
                try:
                    index.get_chrom_range_pos(chrom)
                except IndexError:
                    # No snps for this chrom
                    break
                idx0 = index.index_pos(chrom, pos)
                idx1 = index.index_pos(chrom, pos + win_step)
                if chrom_mat[idx0] != chrom:
                    msg = 'chroms do not match: ' + str(chrom_mat[idx0])
                    msg += ', ' + str(chrom)
                    raise RuntimeError(msg)
                yield self.get_chunk(slice(idx0, idx1),
                                     kept_fields=kept_fields,
                                     ignored_fields=ignored_fields)
                pos += win_step

    @property
    def chroms(self):
        index = self.pos_index

        return index.chroms

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
        self._index = None
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
    def fpath(self):
        return self._h5file.filename

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

    def get_samples(self):
        if 'samples' in self._h5file.attrs:
            samples = json.loads(self._h5file.attrs['samples'])
        else:
            if GT_FIELD not in self.keys():
                raise RuntimeError('There are not genotypes in hdf5 file')
            samples = None
        return samples

    def set_samples(self, samples):
        old_samples = self.get_samples()
        if old_samples is None:
            if GT_FIELD in self.keys():
                n_samples = self[GT_FIELD].shape[1]
                if n_samples != len(samples):
                    msg = 'New samples should have the same length as GTs'
                    raise ValueError(msg)
        else:
            if len(samples) != len(old_samples):
                msg = 'New samples should have the same length as old samples'
                raise ValueError(msg)
        self._h5file.attrs['samples'] = json.dumps(samples)

    samples = property(get_samples, set_samples)


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
