
from itertools import chain
import tempfile

import numpy
import h5py

from variation import SNPS_PER_CHUNK, DEF_DSET_PARAMS

def _first_item(iterable):
    for item in iterable:
        return item
    raise ValueError('The iterable was empty')


def concat_chunks_into_dset(matrices, group, dset_name,
                                 rows_in_chunk=SNPS_PER_CHUNK):
    matrices = iter(matrices)
    fst_mat = _first_item(matrices)
    if matrices is None:
        raise ValueError('There were no matrices to concatenate')
    mats = chain([fst_mat], matrices)

    size = fst_mat.shape
    kwargs = DEF_DSET_PARAMS.copy()
    kwargs['dtype'] = fst_mat.dtype
    kwargs['maxshape'] = (None,) + size[1:]
    kwargs['chunks'] = (SNPS_PER_CHUNK,) + size[1:]
    dset = group.create_dataset(dset_name, size, **kwargs)

    current_snp_index = 0
    for mat in mats:
        num_snps = mat.shape[0]
        start = current_snp_index
        stop = current_snp_index + num_snps

        current_snps_in_dset = dset.shape[0]
        if current_snps_in_dset < stop:
            dset.resize((stop,) + size[1:])
        dset[start:stop] = mat
        current_snp_index += num_snps

    return dset


def concat_chunks_into_array(matrices, concat_in_memory=True):
    '''concat_in_memory=False will require to use the double memory during
    the process.
    concat_in_memory=True will use a compressed hdf5 dset in disk, so it will
    require less memory, but it will be slower
    '''

    if concat_in_memory:
        matrices = list(matrices)

    try:
        return numpy.concatenate(matrices, axis=0)
    except TypeError as error:
        error = str(error)
        if 'argument' in error and 'sequence' in error:
            # the matrices are not in a list, but in an iterator
            pass
        else:
            raise

    # we will create an hdf5 dataset and uset it to store the matrices
    # in the iterator. Once all of them are stored we will create from
    # them the numpy.array
    # This will save memory, even if we create the hdf5 in memory, because
    # it will be compressed
    fhand = tempfile.NamedTemporaryFile(suffix='.hdf5')

    hdf5 = h5py.File(fhand)
    group = hdf5.create_group('concat')
    dset = concat_chunks_into_dset(matrices, group, 'concat')
    return dset[:]
