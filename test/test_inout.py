import unittest
import inspect
from os.path import dirname, abspath, join
from tempfile import NamedTemporaryFile
import gzip
import warnings

import numpy
import h5py

from variation.inout import (VCFParser, vcf_to_hdf5, dsets_chunks_iter,
                             write_hdf5_from_chunks)

# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

TEST_DATA_DIR = abspath(join(dirname(inspect.getfile(inspect.currentframe())),
                        'test_data'))


TEST_VCF2 = join(TEST_DATA_DIR, 'format_def.vcf')
TEST_VCF = join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.vcf.gz')


class IOTest(unittest.TestCase):

    def test_vcf_parsing(self):
        fhand = open(TEST_VCF2, 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=1000)
        snps = list(vcf.variations)
        assert len(snps) == 5
        snp = snps[0]
        assert snp[0] == b'20'
        assert snp[1] == 14370
        assert snp[6] == []
        assert snp[7] == {b'AF': 0.5, b'DP': 14, b'NS': 3, b'H2': True,
                          b'DB': True}
        gts = dict(snp[-1])
        assert gts[b'GT'] == [[0, 0], [1, 0], [1, 1]]
        assert gts[b'GQ'] == [48, 48, 43]
        assert gts[b'DP'] == [1, 8, 5]
        fhand.close()

        fhand = gzip.open(TEST_VCF, 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=100, max_field_lens={'alt': 4})
        assert vcf.max_field_lens['alt'] == 4
        assert vcf.max_field_str_lens == {'FILTER': 0,
                                          'INFO': {b'TYPE': 3, b'CIGAR': 2},
                                          'alt': 1, 'chrom': 10}
        fhand = gzip.open(TEST_VCF, 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=1000)
        assert vcf.max_field_lens['alt'] == 2
        assert vcf.max_field_str_lens == {'FILTER': 0,
                                          'INFO': {b'TYPE': 7, b'CIGAR': 6},
                                          'alt': 4, 'chrom': 10}

        fhand = gzip.open(join(TEST_DATA_DIR, 'ril.vcf.gz'), 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=10000)
        assert vcf.max_field_lens['FORMAT'] == {b'QA': 1, b'AO': 1, b'GL': 0}

    def test_write_hdf5(self):
        fhand = open(TEST_VCF2, 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=1000)
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        try:
            log = vcf_to_hdf5(vcf, out_fhand.name)
        finally:
            out_fhand.close()
            fhand.close()
        assert log == {'data_no_fit': {}, 'variations_processed': 5}

        fhand = gzip.open(TEST_VCF, 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=1000,
                        max_field_lens={'alt': 4}, kept_fields=['GT', 'QA'],
                        max_n_vars=2500)
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        try:
            log = vcf_to_hdf5(vcf, out_fhand.name)
        finally:
            out_fhand.close()
            fhand.close()
        assert log == {'variations_processed': 2500,
                       'data_no_fit': {b'QA': 12}}

        fhand = gzip.open(join(TEST_DATA_DIR, 'ril.vcf.gz'), 'rb')
        vcf = VCFParser(fhand)
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                log = vcf_to_hdf5(vcf, out_fhand.name)
            self.fail('RuntimeError expected')
        except RuntimeError:
            pass
        finally:
            out_fhand.close()
            fhand.close()
        return

        fhand = gzip.open(join(TEST_DATA_DIR, 'ril.vcf.gz'), 'rb')
        vcf = VCFParser(fhand, pre_read_max_size=10000,
                        max_field_lens={'alt':3})
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        try:
            log = vcf_to_hdf5(vcf, out_fhand.name)
        finally:
            out_fhand.close()
            fhand.close()

        assert log['variations_processed'] == 943
        assert log['data_no_fit'][b'PAIRED'] == 16


class ChunkIterTest(unittest.TestCase):
    def test_chunk_iter(self):
        hdf5 = h5py.File(join(TEST_DATA_DIR, '1000snps.hdf5'))
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        hdf5_2 = h5py.File(out_fhand.name, 'w')
        try:
            write_hdf5_from_chunks(hdf5_2, dsets_chunks_iter(hdf5))
            assert list(hdf5_2['calls'].keys()) == ['DP', 'GQ', 'GT', 'HQ']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()

        hdf5 = h5py.File(join(TEST_DATA_DIR, '1000snps.hdf5'))
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        hdf5_2 = h5py.File(out_fhand.name, 'w')
        try:
            write_hdf5_from_chunks(hdf5_2, dsets_chunks_iter(hdf5,
                                                           kept_fields=['GT']))
            assert list(hdf5_2['calls'].keys()) == ['GT']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            out_fhand.close()

if __name__ == "__main__":
    import sys;sys.argv = ['', 'IOTest.test_write_hdf5']
    unittest.main()
