# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from tempfile import NamedTemporaryFile
import os
from os.path import join
import gzip
from time import time

from test.test_utils import TEST_DATA_DIR
from variation.vars_matrices.vars_matrices import (VariationsArrays,
                                                   VariationsH5,
                                                   VariationsBcolz)
from variation.vcf import VCFParser

import numpy


def _create_var_mat_objs_from_h5(h5_fpath):
    in_snps = VariationsH5(h5_fpath, mode='r')
    for klass in VAR_MAT_CLASSES:
        out_snps = _init_var_mat(klass)
        out_snps.put_chunks(in_snps.iterate_chunks())
        yield out_snps

def _create_var_mat_objs_from_vcf(vcf_fpath, kwargs, kept_fields=None,
                                  ignored_fields=None):
    for klass in VAR_MAT_CLASSES:
        if vcf_fpath.endswith('.gz'):
            fhand = gzip.open(vcf_fpath, 'rb')
        else:
            fhand = open(vcf_fpath, 'rb')
        vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=100000, **kwargs)
        out_snps = _init_var_mat(klass)
        out_snps.put_vars_from_vcf(vcf_parser)
        fhand.close()
        yield out_snps


class VcfH5Test(unittest.TestCase):
    def test_create_empty(self):
        with NamedTemporaryFile(suffix='.h5') as fhand:
            os.remove(fhand.name)
            h5f = VariationsH5(fhand.name, 'w')
            assert h5f._h5file.filename

    def test_put_vars_hdf5_from_vcf(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            os.remove(fhand.name)
            h5f = VariationsH5(fhand.name, 'w')
            h5f.put_vars_from_vcf(vcf)
            dset = h5f['/calls/GT']
            vcf_fhand.close()

    def test_put_vars_arrays_from_vcf(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VariationsArrays()
        snps.put_vars_from_vcf(vcf)
        assert snps['/calls/GT'].shape == (5, 3, 2)
        assert numpy.all(snps['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])
        vcf_fhand.close()

    def xtest_put_vars_barrays_from_vcf(self):
        print('dentro de test bcolz')
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VariationsBcolz()
        snps.put_vars_from_vcf(vcf)
        assert snps['/calls/GT'].shape == (5, 3, 2)
        assert numpy.all(snps['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])
        vcf_fhand.close()

    def test_create_hdf5_with_chunks(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5_2 = VariationsH5(out_fhand.name, 'w')
        try:
            hdf5_2.put_chunks(hdf5.iterate_chunks())
            assert sorted(hdf5_2['calls'].keys()) == ['DP', 'GQ', 'GT', 'HQ']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()

        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5_2 = VariationsH5(out_fhand.name, 'w')
        try:
            hdf5_2.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
            assert list(hdf5_2['calls'].keys()) == ['GT']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()


VAR_MAT_CLASSES = (VariationsH5, VariationsArrays)


def _init_var_mat(klass):
    if klass is VariationsH5:
        fhand = NamedTemporaryFile(suffix='.h5')
        os.remove(fhand.name)
        var_mat = klass(fhand.name, mode='w')
    else:
        var_mat = klass()
    return var_mat


class VarMatsTests(unittest.TestCase):
    def test_create_arrays_with_chunks(self):

        for klass in VAR_MAT_CLASSES:
            in_snps = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
            var_mat = _init_var_mat(klass)
            try:
                var_mat.put_chunks(in_snps.iterate_chunks())
                assert numpy.all(in_snps['/calls/GT'][:] == var_mat['/calls/GT'][:])
                in_snps.close()
            finally:
                pass

    def test_count_alleles(self):
        for klass in VAR_MAT_CLASSES:
            in_snps = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
            var_mat = _init_var_mat(klass)
            try:
                var_mat.put_chunks(in_snps.iterate_chunks(kept_fields=['/calls/GT']))
                assert numpy.any(var_mat.allele_count)
                in_snps.close()
            finally:
                pass

        expected = [[3, 3, 0], [5, 1, 0], [0, 2, 4], [6, 0, 0], [2, 3, 1]]
        for klass in VAR_MAT_CLASSES :
            fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
            vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=1000)
            var_mat = _init_var_mat(klass)
            var_mat.put_vars_from_vcf(vcf_parser)
            assert numpy.all(var_mat.allele_count == expected)
            fhand.close()

    def test_create_matrix(self):
        for klass in VAR_MAT_CLASSES:
            var_mat = _init_var_mat(klass)
            matrix = var_mat._create_matrix('/calls/HQ', shape = (200, 1), dtype=float,
                                           fillvalue=1.5)
            assert matrix.shape == (200, 1)
            assert matrix.dtype == float
            assert matrix[0, 0] == 1.5

    def test_create_with_chunks(self):
        in_snps = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        for klass in VAR_MAT_CLASSES:
            out_snps = _init_var_mat(klass)
            out_snps.put_chunks(in_snps.iterate_chunks())
            assert '/calls/GQ' in out_snps.keys()
            assert out_snps['/calls/GT'].shape == (5, 3, 2)
            assert numpy.all(out_snps['/calls/GT'][0] == [[0, 0], [1, 0], [1, 1]])

        for klass in VAR_MAT_CLASSES:
            out_snps = _init_var_mat(klass)
            out_snps.put_chunks(in_snps.iterate_chunks(kept_fields=['/calls/GT']))
            assert '/calls/GQ' not in out_snps.keys()
            assert out_snps['/calls/GT'].shape == (5, 3, 2)
            assert numpy.all(out_snps['/calls/GT'][:] == in_snps['/calls/GT'])

    def test_iterate_chunks(self):

        fpath = join(TEST_DATA_DIR, 'ril.vcf.gz')
        kwargs = {'max_field_lens': {"alt":3}, 'ignored_fields': {'GL'}}
        for var_mats in _create_var_mat_objs_from_vcf(fpath, kwargs=kwargs):
            start = time()
            chunks = list(var_mats.iterate_chunks())
            chunk = chunks[0]
            fin = time()
            elapsed = fin-start
            assert chunk['/calls/GT'].shape == (200, 153, 2)

        fpath = join(TEST_DATA_DIR, 'format_def.vcf')
        #check GT
        for var_mats in _create_var_mat_objs_from_vcf(fpath, {}):
            start = time()
            chunks = list(var_mats.iterate_chunks())
            chunk = chunks[0]
            fin = time()
            elapsed = fin-start
            assert chunk['/calls/GT'].shape == (5, 3, 2)
            assert numpy.all(chunk['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])

    def test_delete_item_from_variationArray(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VariationsArrays()
        snps.put_vars_from_vcf(vcf)
        del snps['/calls/GT']
        assert '/calls/GT' not in snps.keys()
        vcf_fhand.close()

    def test_metadata(self):
        for klass in VAR_MAT_CLASSES :
            fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
            vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=1000)
            var_mat = _init_var_mat(klass)
            var_mat.put_vars_from_vcf(vcf_parser)
            metadata = var_mat.metadata
            assert '/variations/filter/q10' in metadata.keys()

            for klass in VAR_MAT_CLASSES:
                out_snps = _init_var_mat(klass)
                out_snps.put_chunks(var_mat.iterate_chunks())
                assert '/variations/filter/q10' in out_snps.keys()
            fhand.close()

    def test_expand_list(self):
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'VarMatsTests.test_dset_id']
    unittest.main()
