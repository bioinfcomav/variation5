import numpy
from variation.matrix.methods import iterate_matrix_chunks


def _get_chunks_gt(chunks):
    gts = chunks['/calls/GT']
    return gts


def is_variant(chunks):
    gts = _get_chunks_gt(chunks)
    is_var = numpy.ndarray(gts.shape[0])
    start = 0
    for chunk in iterate_matrix_chunks(gts):
        stop = start + chunk.shape[0]
        is_var[start:stop] = numpy.sum(chunk > 0, axis=(1, 2)) >= 1
        start = stop
    return is_var


def is_non_variant(chunks):
    gts = _get_chunks_gt(chunks)
    is_non_var = numpy.ndarray(gts.shape[0])
    start = 0
    for chunk in iterate_matrix_chunks(gts):
        stop = start + chunk.shape[0]
        is_non_var[start:stop] = numpy.all(chunk <= 0, axis=(1, 2))
        start = stop
    return is_non_var


def count_variant(chunks):
    """Cuenta variantes con al menos una observación allelica no referenciada
    Count variants with at least one non-reference allele observation."""
    count_var = 0
    count_var += numpy.count_nonzero(is_variant(chunks))
    return count_var


def count_non_variant(chunks):
    """Cuenta variantes con ninguna una observación allelica no referenciada
    Count variants with no non-reference allele observation."""
    count_var = 0
    count_var += numpy.count_nonzero(is_non_variant(chunks))
    return count_var


def is_singleton(chunks, allele):
    """Find variants with only a single instance of `allele` observed.
    Encuentra variantes con solo una instancia del alelo observado"""
    gts = _get_chunks_gt(chunks)
    is_single = numpy.ndarray(gts.shape[0])
    start = 0
    for chunk in iterate_matrix_chunks(gts):
        stop = start + chunk.shape[0]
        is_single[start:stop] = numpy.sum(chunk == allele, axis=(1, 2)) == 1
        start = stop
    return is_single


def count_singleton(chunks, allele=1):
    return numpy.count_nonzero(is_singleton(chunks, allele))


def is_doubleton(chunks, allele=1):
    """Find variants with only two instance of `allele` observed.
    Encuentra variantes con dos instancias del alelo observado"""
    gts = _get_chunks_gt(chunks)
    start = 0
    is_double = numpy.ndarray(gts.shape[0])
    for chunk in iterate_matrix_chunks(gts):
        stop = start + chunk.shape[0]
        is_double[start:stop] = numpy.sum(chunk == allele, axis=(1, 2)) == 2
        start = stop
    return is_double


def count_doubleton(chunks, allele=1):
    return numpy.count_nonzero(is_doubleton(chunks, allele))


def allele_number(chunks):
    """Count the number of non-missing allele calls per variant.
    Cuenta el numero de allelos no-missing por snp"""
    gts = _get_chunks_gt(chunks)
    allele_num = numpy.ndarray(gts.shape[0])
    start = 0
    for chunk in iterate_matrix_chunks(gts):
        stop = start + chunk.shape[0]
        allele_num[start:stop] = numpy.sum(chunk >= 0, axis=(1, 2))
        start = stop
    return allele_num


def allele_count(chunks, allele=1):
    """Calculate number of observations of the given allele per variant.
    Cuenta el numero de observaciones de un allelo dado por snp"""
    gts = _get_chunks_gt(chunks)
    a_count = numpy.ndarray(gts.shape[0])
    start = 0
    for chunk in iterate_matrix_chunks(gts):
        stop = start + chunk.shape[0]
        a_count[start:stop] = numpy.sum(chunk == allele, axis=(1, 2))
        start = stop
    return a_count


def allele_frequency(chunks, allele=1):
    allele_num = allele_number(chunks)
    a_count = allele_count(chunks, allele=allele)
    allele_freq = numpy.where(allele_num > 0, a_count / allele_num, 0)
    return allele_freq
