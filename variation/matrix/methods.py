from itertools import chain
import sys
import math

import h5py
import numpy

from variation.iterutils import group_items, first
from variation import SNPS_PER_CHUNK, MISSING_VALUES

# Missing docstring
# pylint: disable=C0111

AVAILABLE_MEM = 0.5 * 1024 * 1024


def resize_matrix(matrix, new_size):
    if is_array(matrix):
        matrix.resize(new_size, refcheck=False)

    elif is_dataset(matrix):
        matrix.resize(new_size)


def _extend_dset(dset, arrays):
    for array in arrays:
        append_matrix(dset, array)


def is_array(array):
    if isinstance(array, numpy.ndarray):
        return True
    else:
        return False


def is_dataset(array):
    if isinstance(array, h5py.Dataset):
        return True
    else:
        return False


def append_matrix(matrix, matrix_to_append):
    start = matrix.shape[0]
    stop = start + matrix_to_append.shape[0]
    size = matrix.shape
    new_size = list(size)
    new_size[0] = stop

    if is_dataset(matrix_to_append):
        array = matrix_to_append[:]
    else:
        array = matrix_to_append
    resize_matrix(matrix, new_size)

    matrix[start:stop] = array


def fill_array(array, size, dim=0, constant_values=0):
    num_fill = size - array.shape[dim]
    fill_format = [[0, 0]] * dim + [[0, num_fill]] + [[0, 0]] * (array.ndim - dim - 1)
    return numpy.pad(array, fill_format, mode='constant',
                     constant_values=constant_values)


def append_different_size(matrix1, matrix2, axis=0):
    if matrix1.shape[1] > matrix2.shape[1]:
        matrix2 = fill_array(matrix2, matrix1.shape[1], dim=1,
                             constant_values=MISSING_VALUES[matrix1.dtype])
    else:
        matrix1 = fill_array(matrix1, matrix2.shape[1], dim=1,
                             constant_values=MISSING_VALUES[matrix1.dtype])
    return numpy.append(matrix1, matrix2, axis=axis)


def extend_matrix(matrix, arrays):
    arrays = iter(arrays)
    if is_dataset(matrix):
        _extend_dset(matrix, arrays)
    else:
        _extend_array_with_iter(matrix, arrays)


def _extend_array(array, arrays):
    arrays = list(filter(lambda x: x is not None, arrays))
    if not arrays:
        return
    n_snps = sum([array_.shape[0] for array_ in arrays])
    n_snps += array.shape[0]

    for array_ in arrays:
        if array.shape[1:] != array_.shape[1:]:
            msg = 'The arrays to extend do not match the shape'
            raise ValueError(msg)

    i0 = array.shape[0]
    shape = list(arrays[0].shape)
    shape[0] = n_snps
    array.resize(shape, refcheck=False)

    for array_to_append in arrays:
        if array_to_append is array:
            msg = 'You cannot append an array onto itself'
            raise ValueError(msg)
        i1 = i0 + array_to_append.shape[0]
        array[i0:i1] = array_to_append
        i0 = i1


def _extend_array_with_iter(array, matrices):
    try:
        matrix = first(matrices)
    except ValueError:
        return

    matrices = chain([matrix], matrices)

    matrix_size = sys.getsizeof(matrix)
    mats_in_group = math.floor(AVAILABLE_MEM / matrix_size)
    if not mats_in_group:
        mats_in_group = 1
    for mats_in_mem in group_items(matrices, mats_in_group):
        _extend_array(array, mats_in_mem)


def num_variations(matrix):
    return matrix.shape[0]


def iterate_matrix_chunks(matrix, chunk_size=SNPS_PER_CHUNK, sample_idx=None):
    nsnps = num_variations(matrix)
    for start in range(0, nsnps, chunk_size):
        stop = start + chunk_size
        if stop > nsnps:
            stop = nsnps
        if sample_idx is not None:
            mat = matrix[start:stop, sample_idx]
        else:
            mat = matrix[start:stop]
        yield mat


def calc_min_max(matrix, chunk_size=SNPS_PER_CHUNK, sample_idx=None):
    if matrix.size == 0:
        return numpy.inf, -numpy.inf
    if is_dataset(matrix):
        max_ = None
        min_ = None
        if chunk_size:
            chunks = iterate_matrix_chunks(matrix, chunk_size=chunk_size,
                                           sample_idx=sample_idx)
        else:
            if sample_idx:
                chunks = [matrix[:, sample_idx]]
            else:
                chunks = [matrix]
        for chunk in chunks:
            chunk_max = numpy.nanmax(chunk)
            chunk_min = numpy.nanmin(chunk)
            if max_ is None or max_ < chunk_max:
                max_ = chunk_max
            if min_ is None or min_ > chunk_min:
                min_ = chunk_min
    else:
        if sample_idx:
            matrix = matrix[:, sample_idx]
        max_ = numpy.nanmax(matrix)
        min_ = numpy.nanmin(matrix)
    return min_, max_


def is_missing(matrix, axis=1):
    if is_dataset(matrix):
        matrix = matrix[:]
    if axis is None:
        return matrix == MISSING_VALUES[matrix.dtype]
    else:
        return numpy.any(matrix == MISSING_VALUES[matrix.dtype], axis=axis)


def _set_matrix_by_chunks(mat1, slice1, mat2, chunk_size=SNPS_PER_CHUNK):

    if isinstance(slice1, tuple):
        first_slice = slice1[0]
        other_slices = list(slice1[1:])
    else:
        first_slice = slice1
        other_slices = None

    start = first_slice.start
    if start is None:
        start = 0
    stop = first_slice.stop
    if stop is None:
        stop = mat1.shape[0]

    if start < 0:
        start = mat1.shape[0] + start
    if stop < 0:
        stop = mat1.shape[0] + stop

    slice_len = abs(stop - start)
    assert slice_len == mat2.shape[0]

    stop2 = slice_len

    for subslice1_start, subslice2_start in zip(range(start, stop, chunk_size),
                                                range(0, slice_len, chunk_size)):
        subslice1_stop = subslice1_start + chunk_size
        subslice2_stop = subslice2_start + chunk_size
        if subslice1_stop > stop:
            subslice1_stop = stop
            subslice2_stop = stop2
        if other_slices is None:
            mat1[subslice1_start: subslice1_stop, ...] = mat2[subslice2_start: subslice2_stop, ...]
        else:
            subslice1 = tuple([slice(subslice1_start, subslice1_stop)] + other_slices)
            subslice2 = tuple([slice(subslice2_start, subslice2_stop)] + other_slices)
            mat1[subslice1] = mat2[subslice2]


def _get_longest_byte_dtype(mat1, mat2):
    if mat1.dtype.type == numpy.bytes_:
        itemsize = max(mat1.dtype.itemsize, mat2.dtype.itemsize)
        new_dtype = numpy.dtype(('S', itemsize))
    else:
        new_dtype = None
    return new_dtype


def _reshape_filling(mat, new_shape, fill_value, dtype=None):
    if dtype is None:
        dtype = mat.dtype

    new_mat = numpy.full(new_shape, fill_value, dtype=dtype)
    shape = mat.shape
    slice_ = [shape[0]] + [shape[1]] + list(shape[2:])
    slice_ = tuple([slice(0, dim_len) for dim_len in slice_])
    new_mat[slice_] = mat
    return new_mat


def _copy_dset(dset, shape=None, dtype=None):
    group = dset.parent

    old_shape = dset.shape
    if shape is None:
        shape = old_shape

    maxshape = []
    for max_, shape_ in zip(dset.maxshape, shape):
        if max_ is None or shape_ is None:
            maxshape.append(None)
        else:
            maxshape.append(max(max_, shape_))
    maxshape = tuple(maxshape)

    if dtype is None:
        dtype = dset.dtype

    chunks = dset.chunks
    compression = dset.compression
    compression_opts = dset.compression_opts
    shuffle = dset.shuffle
    fletcher32 = dset.fletcher32
    fillvalue = dset.fillvalue

    annon_dset = group.create_dataset(None, shape=shape, dtype=dtype,
                                      chunks=chunks, maxshape=maxshape,
                                      compression=compression,
                                      compression_opts=compression_opts,
                                      shuffle=shuffle, fletcher32=fletcher32,
                                      fillvalue=fillvalue)

    slice_ = tuple([slice(None, dim_size) for dim_size in old_shape])
    _set_matrix_by_chunks(annon_dset, slice_, dset[:])

    return annon_dset


def _reshape_filling_dset(dset, new_shape=None, dtype=None):

    if dtype is None:
        new_dtype = dset.dtype
    else:
        new_dtype = dtype

    try:
        dset.resize(new_shape)
    except (TypeError, ValueError):
        dset = _copy_dset(dset, shape=new_shape, dtype=new_dtype)
    return dset


def _concat_array_vector(vectors, missing_value=None):
    concat_vector = None
    for vector in vectors:
        if concat_vector is None:
            concat_vector = vector.reshape((vector.size, 1))
        else:
            vector = vector.reshape((vector.size, 1))
            concat_vector = vstack([concat_vector, vector], missing_value)
    return concat_vector.reshape((concat_vector.shape[0],))


def _concat_dset_vector(vectors, missing_value=None):
    concat_vector = None
    for vector in vectors:
        if concat_vector is None:
            concat_vector = vector
        else:
            old_concat_size = concat_vector.shape[0]
            new_shape = (old_concat_size + vector.shape[0],)
            new_dtype = _get_longest_byte_dtype(concat_vector, vector)
            concat_vector = _reshape_filling_dset(concat_vector,
                                                  new_shape=new_shape,
                                                  dtype=new_dtype)
            concat_vector[old_concat_size:None] = vector
    return concat_vector


def _replace_dset_if_required(if_first_matrix_is_dataset_replace_it,
                              dset_in_h5, annon_dset):
    if if_first_matrix_is_dataset_replace_it and dset_in_h5 is not annon_dset:
        h5 = dset_in_h5.file
        path = dset_in_h5.name
        del h5[path]
        h5[path] = annon_dset


def vstack(matrices, missing_value,
           if_first_matrix_is_dataset_replace_it=False):

    matrices = iter(matrices)

    mat1 = None
    for mat2 in matrices:
        if mat1 is None:
            mat1 = mat2
            first_mat = mat1
            continue

        shape1 = mat1.shape
        shape2 = mat2.shape
        if len(shape1) != len(shape2):
            raise ValueError('All matrices should have the same number of dimensions')
        if len(shape1) < 2:
            raise ValueError('At least two dimensional matrices required')
        if len(shape1) > 3:
            raise NotImplementedError('Not more than 3 dimensions allowed')

        new_dtype = _get_longest_byte_dtype(mat1, mat2)

        max_dim_1_len = max(shape1[1], shape2[1])
        if len(shape1) == 2:
            new_shape = (shape1[0] + shape2[0], max_dim_1_len)
        elif len(shape1) == 3:
            max_dim_2_len = max(shape1[2], shape2[2])
            new_shape = (shape1[0] + shape2[0], max_dim_1_len, max_dim_2_len)
        else:
            RuntimeError('Fixme: we should not be here')

        if isinstance(mat1, numpy.ndarray):
            mat1 = _reshape_filling(mat1, new_shape, missing_value,
                                    dtype=new_dtype)
        else:
            mat1 = _reshape_filling_dset(mat1, new_shape, dtype=new_dtype)

        if len(shape1) == 2:
            slice_to_copy_mat2 = (slice(shape1[0], None), slice(0, shape2[1]))
        elif len(shape1) == 3:
            slice_to_copy_mat2 = (slice(shape1[0], None),
                                  slice(0, shape2[1]),
                                  slice(0, shape2[2]))

        slice_to_copy_mat2 = tuple(slice_to_copy_mat2)
        mat1[slice_to_copy_mat2] = mat2

    if mat1 is None:
        raise ValueError('At least one matrix required')

    if isinstance(mat1, numpy.ndarray):
        _replace_dset_if_required(if_first_matrix_is_dataset_replace_it,
                                  first_mat, mat1)

    return mat1


def concat_vector(vectors, missing_value=None,
                  if_first_matrix_is_dataset_replace_it=False):
    vectors = iter(vectors)

    try:
        first_mat = next(vectors)
    except StopIteration:
        raise ValueError('At least one vector required')

    vectors = chain([first_mat], vectors)

    if isinstance(first_mat, numpy.ndarray):
        return _concat_array_vector(vectors, missing_value=missing_value)
    else:
        annon_dset = _concat_dset_vector(vectors, missing_value=missing_value)
        _replace_dset_if_required(if_first_matrix_is_dataset_replace_it,
                                  first_mat, annon_dset)
        return annon_dset


def concat_matrices(matrices, missing_value=None,
                    if_first_matrix_is_dataset_replace_it=False):
    matrices = iter(matrices)
    try:
        first_mat = next(matrices)
    except StopIteration:
        raise ValueError('At least one matrix required')
    matrices = chain([first_mat], matrices)

    if first_mat.ndim == 1:
        return concat_vector(matrices, missing_value=missing_value,
                             if_first_matrix_is_dataset_replace_it=if_first_matrix_is_dataset_replace_it)
    else:
        return vstack(matrices, missing_value=missing_value,
                      if_first_matrix_is_dataset_replace_it=if_first_matrix_is_dataset_replace_it)


def resize_array(array, shape, missing_value):
    assert len(array.shape) == len(shape)
    slice_ = tuple(slice(0, dim_len) for dim_len in array.shape)
    mat = numpy.full(shape, missing_value, dtype=array.dtype)
    mat[slice_] = array
    return mat
