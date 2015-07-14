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

from test.test_utils import TEST_DATA_DIR
from variation.vcfh5 import VcfH5
from variation.vcf import VCFParser

import numpy
from variation.vcfh5.vcfh5 import VcfArrays


class VcfH5Test(unittest.TestCase):
    def test_create_empty(self):
        with NamedTemporaryFile(suffix='.h5') as fhand:
            os.remove(fhand.name)
            h5f = VcfH5(fhand.name, 'w')
            assert h5f.h5file.filename

    def test_write_vars_hdf5_from_vcf(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            os.remove(fhand.name)
            h5f = VcfH5(fhand.name, 'w')
            h5f.write_vars_from_vcf(vcf)
            vcf_fhand.close()

    def test_write_vars_arrays_from_vcf(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VcfArrays()
        snps.write_vars_from_vcf(vcf)
        assert snps[b'/calls/GT'].shape == (5, 3, 2)
        assert numpy.all(snps[b'/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])
        vcf_fhand.close()

    def test_create_hdf5_with_chunks(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5_2 = VcfH5(out_fhand.name, 'w')
        try:
            hdf5_2.write_chunks(hdf5.iterate_chunks())
            assert sorted(hdf5_2['calls'].keys()) == ['DP', 'GQ', 'GT', 'HQ']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()

        hdf5 = VcfH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5_2 = VcfH5(out_fhand.name, 'w')
        try:
            hdf5_2.write_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
            assert list(hdf5_2['calls'].keys()) == ['GT']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()


    def test_create_arrays_with_chunks(self):
        hdf5 = VcfArrays(join(TEST_DATA_DIR, '1000snps.hdf5'))
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5_2 = VcfH5(out_fhand.name, 'w')
        try:
            hdf5_2.write_chunks(hdf5.iterate_chunks())
            assert sorted(hdf5_2['calls'].keys()) == ['DP', 'GQ', 'GT', 'HQ']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()

        hdf5 = VcfH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5_2 = VcfH5(out_fhand.name, 'w')
        try:
            hdf5_2.write_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
            assert list(hdf5_2['calls'].keys()) == ['GT']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()


    def test_count_alleles(self):
        hdf5 = VcfH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        assert numpy.any(hdf5.allele_count)

    def test_create_matrix(self):
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        os.remove(out_fhand.name)
        hdf5 = VcfH5(out_fhand.name, 'w')
        try:
            hdf5.create_matrix('/group/')
            self.fail('Value error expected')
        except ValueError:
            pass
        dset = hdf5.create_matrix('/group/dset', shape = (200, 1))
        assert dset.shape == (200, 1)

if __name__ == "__main__":
    import sys;sys.argv = ['', 'VcfH5Test.test_write_vars_arrays_from_vcf']
    unittest.main()
