# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

from os.path import abspath, dirname, join
import inspect
from tempfile import NamedTemporaryFile
import os
import gzip
from timeit import timeit

from test.test_utils import TEST_DATA_DIR
from variation.vars_matrices.vars_matrices import VariationsH5, VariationsArrays
from variation.vars_matrices.filters import (mafs_filter_fact,
                                             missing_rate_filter_fact)

from variation.vars_matrices.stats import (MafCalculator,
                                           MissingGTCalculator,
    calc_stat_by_chunk)
from variation.matrix.stats import histogram, _low_mem_histogram

from variation.vcf import VCFParser
from variation.matrix.methods import calc_min_max


# def from_vcf_to_hdf5():
#     fpath = join(TEST_DATA_DIR, 'performance',
#                  'fb_genome_inca_aethi_torvum_beren_annotated.vcf.gz')
#     fhand = gzip.open(fpath, 'rb')
#
#     kwargs = {'max_field_lens': {"alt":0}, 'max_n_vars':5000,
#               'kept_fields':['/calls/GT', '/calls/GQ'], 'ignore_alt':True}
#     vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=5,
#                            **kwargs)
#     out_fhand = NamedTemporaryFile(suffix='.h5', prefix='tomato_all')
#     os.remove(out_fhand.name)
#     var_mat = VariationsH5(out_fhand.name, mode='w')
#     log = var_mat.put_vars_from_vcf(vcf_parser)
#     print('log_alt_max_detected: ', log['alt_max_detected'])
#     print('log_items_alt_descarted: ', log['num_alt_item_descarted'])

def from_vcf_to_hdf5():
    fpath = join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.vcf.gz')
    fhand = gzip.open(fpath, 'rb')
    kwargs = {'max_field_lens': {"alt":4}}
    vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=10000,
                           **kwargs)
    h5 = VariationsH5(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.h5'), mode='w')
    h5.put_vars_from_vcf(vcf_parser)


def filter_mafs_from_hdf5():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')

    filter_chunk = mafs_filter_fact(max_=0.8)
    chunks = h5.iterate_chunks(kept_fields=['/calls/GT'])
    filtered_chunks = map(filter_chunk, chunks)

    out_fpath = NamedTemporaryFile(suffix='.h5')
    os.remove(out_fpath.name)
    h5_2 = VariationsH5(out_fpath.name, mode='w')
    h5_2.put_chunks(filtered_chunks)
    h5_2.close()

def filter_missing_rates_from_hdf5():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')

    filter_chunk = missing_rate_filter_fact(min_=0.8)
    chunks = h5.iterate_chunks(kept_fields=['/calls/GT'])
    filtered_chunks = map(filter_chunk, chunks)

    out_fpath = NamedTemporaryFile(suffix='.h5')
    os.remove(out_fpath.name)
    h5_2 = VariationsH5(out_fpath.name, mode='w')
    h5_2.put_chunks(filtered_chunks)
    h5_2.close()


def stats_missing_rate_from_hdf5():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')
    calc_stat_by_chunk(h5, MissingGTCalculator())

def stats_missing_rate_from_hdf5_memory():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    var_mat = VariationsH5(fpath, mode='r')
    array = VariationsArrays()
    array.put_chunks(var_mat.iterate_chunks(kept_fields=['/calls/GT']))
    calc_stat_by_chunk(array, MissingGTCalculator())


def stats_mafs_from_hdf5():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')
    calc_stat_by_chunk(h5, MafCalculator())


def histograma_from_hdf5():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')
    histogram(h5['/calls/GT'], 10, False)


def histograma_low_from_hdf5():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')
    histogram(h5['/calls/GT'], 10, True)

def histograma_min_max_prueba():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')
    min_,max_ = calc_min_max(h5['/calls/GT'])

def histograma_prueba():
    fpath = join(TEST_DATA_DIR, 'performance', 'inca_torvum_all_snps.h5')
    h5 = VariationsH5(fpath, mode='r')
    _low_mem_histogram(h5['/calls/GT'], 30, -1, 3)

def _prepare_smt_for_timeit(funct):
    per_path = (join(dirname(inspect.getfile(inspect.currentframe()))))
    var_package_path = abspath(join(per_path, '..', '..'))
    pre_smt = '''import sys
sys.path.append("{}")
import variation
'''
    pre_smt = pre_smt.format(var_package_path)
    smt = pre_smt + 'from __main__ import ' + funct + '\n'
    smt += funct + '()'
    return smt


def check_performance():
#     smt = _prepare_smt_for_timeit('from_vcf_to_hdf5')
#     print(timeit(smt, number=1))
#     smt = _prepare_smt_for_timeit('filter_mafs_from_hdf5')
#     print(timeit(smt, number=50))
#     smt = _prepare_smt_for_timeit('filter_missing_rates_from_hdf5')
#     print(timeit(smt, number=50))
#     smt = _prepare_smt_for_timeit('stats_mafs_from_hdf5')
#     print(timeit(smt, number=50))
    smt = _prepare_smt_for_timeit('stats_missing_rate_from_hdf5')
    print(timeit(smt, number=50))
    smt = _prepare_smt_for_timeit('stats_missing_rate_from_hdf5_memory')
    print(timeit(smt, number=50))
#     smt = _prepare_smt_for_timeit('histograma_from_hdf5')
#     print(timeit(smt, number=1))
#     smt = _prepare_smt_for_timeit('histograma_low_from_hdf5')
#     print( timeit(smt, number=1))
#     smt = _prepare_smt_for_timeit('histograma_min_max_prueba')
#     print('Min_max', timeit(smt, number=50))
#     smt = _prepare_smt_for_timeit('histograma_prueba')
#     print('Histogram', timeit(smt, number=50))

if __name__ == "__main__":
    check_performance()
    pass
