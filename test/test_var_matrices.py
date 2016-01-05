# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import os
import unittest
import gzip
from tempfile import NamedTemporaryFile
from os.path import join

import h5py
import numpy

from variation.variations.vars_matrices import (VariationsArrays,
                                                VariationsH5,
                                                _prepare_vcf_header, _to_vcf)
from variation.gt_parsers.vcf import VCFParser
from test.test_utils import TEST_DATA_DIR
from variation.utils.misc import remove_nans
from variation.variations.stats import GT_FIELD

VAR_MAT_CLASSES = (VariationsH5, VariationsArrays)


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
        out_snps.put_vars(vcf_parser)
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
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000,
                        max_field_lens={'alt': 3})
        with NamedTemporaryFile(suffix='.hdf5') as fhand:
            os.remove(fhand.name)
            h5f = VariationsH5(fhand.name, 'w', ignore_undefined_fields=True)
            h5f.put_vars(vcf)
            assert numpy.all(h5f['/variations/alt'][:] == [[b'A', b'', b''],
                                                           [b'A', b'', b''],
                                                           [b'G', b'T', b''],
                                                           [b'', b'', b''],
                                                           [b'G', b'GTACT',
                                                            b'']])
            assert h5f['/calls/GT'].shape == (5, 3, 2)
            assert numpy.all(h5f['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])
            expected = numpy.array([48, 48, 43], dtype=numpy.int16)
            assert numpy.all(h5f['/calls/GQ'][0, :] == expected)
            vcf_fhand.close()

    def test_put_vars_arrays_from_vcf(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VariationsArrays(ignore_undefined_fields=True)
        snps.put_vars(vcf)
        assert snps['/calls/GT'].shape == (5, 3, 2)
        assert numpy.all(snps['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])
        expected = numpy.array([48, 48, 43], dtype=numpy.int16)
        assert numpy.all(snps['/calls/GQ'][0, :] == expected)
        vcf_fhand.close()

    def test_create_hdf5_with_chunks(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        out_fpath = out_fhand.name
        out_fhand.close()
        hdf5_2 = VariationsH5(out_fpath, 'w')
        try:
            hdf5_2.put_chunks(hdf5.iterate_chunks())
            assert sorted(hdf5_2['calls'].keys()) == ['DP', 'GQ', 'GT', 'HQ']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            os.remove(out_fpath)

        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        out_fhand = NamedTemporaryFile(suffix='.hdf5')
        out_fpath = out_fhand.name
        out_fhand.close()
        hdf5_2 = VariationsH5(out_fpath, 'w')
        try:
            hdf5_2.put_chunks(hdf5.iterate_chunks(kept_fields=['/calls/GT']))
            assert list(hdf5_2['calls'].keys()) == ['GT']
            assert numpy.all(hdf5['/calls/GT'][:] == hdf5_2['/calls/GT'][:])
        finally:
            os.remove(out_fpath)

        hdf5 = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'), mode='r')
        hdf5_2 = VariationsArrays()
        hdf5_2.put_chunks(hdf5.iterate_chunks(random_sample_rate=0.5))
        assert hdf5.num_variations >= hdf5_2.num_variations


def _init_var_mat(klass):
    if klass is VariationsH5:
        fhand = NamedTemporaryFile(suffix='.h5')
        fpath = fhand.name
        fhand.close()
        var_mat = klass(fpath, mode='w', ignore_undefined_fields=True)
    else:
        var_mat = klass(ignore_undefined_fields=True)
    return var_mat


class VarMatsTests(unittest.TestCase):
    def test_create_arrays_with_chunks(self):

        for klass in VAR_MAT_CLASSES:
            in_snps = VariationsH5(join(TEST_DATA_DIR, '1000snps.hdf5'),
                                   mode='r')
            var_mat = _init_var_mat(klass)
            try:
                var_mat.put_chunks(in_snps.iterate_chunks())
                result = var_mat['/calls/GT'][:]
                assert numpy.all(in_snps['/calls/GT'][:] == result)
                in_snps.close()
            finally:
                pass

    def test_count_alleles(self):
        for klass in VAR_MAT_CLASSES:
            in_snps = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
            var_mat = _init_var_mat(klass)
            try:
                chunks = in_snps.iterate_chunks(kept_fields=['/calls/GT'])
                var_mat.put_chunks(chunks)
                assert numpy.any(var_mat.allele_count)
                in_snps.close()
            finally:
                pass

        expected = [[3, 3, 0], [5, 1, 0], [0, 2, 4], [6, 0, 0], [2, 3, 1]]
        for klass in VAR_MAT_CLASSES:
            fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
            vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=1000)
            var_mat = _init_var_mat(klass)
            var_mat.put_vars(vcf_parser)
            assert numpy.all(var_mat.allele_count == expected)
            fhand.close()

    def test_create_matrix(self):
        for klass in VAR_MAT_CLASSES:
            var_mat = _init_var_mat(klass)
            matrix = var_mat._create_matrix('/calls/HQ', shape=(200, 1),
                                            dtype=float, fillvalue=1.5)
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
            assert numpy.all(out_snps['/calls/GT'][0] == [[0, 0], [1, 0],
                                                          [1, 1]])

        for klass in VAR_MAT_CLASSES:
            out_snps = _init_var_mat(klass)
            chunks = in_snps.iterate_chunks(kept_fields=['/calls/GT'])
            out_snps.put_chunks(chunks)
            assert '/calls/GQ' not in out_snps.keys()
            assert out_snps['/calls/GT'].shape == (5, 3, 2)
            assert numpy.all(out_snps['/calls/GT'][:] == in_snps['/calls/GT'])

    def test_iterate_chunks(self):

        fpath = join(TEST_DATA_DIR, 'ril.vcf.gz')
        kwargs = {'max_field_lens': {"alt": 3},
                  'ignored_fields': {'/calls/GL'}}
        for var_mats in _create_var_mat_objs_from_vcf(fpath, kwargs=kwargs):
            chunks = list(var_mats.iterate_chunks())
            chunk = chunks[0]
            assert chunk['/calls/GT'].shape == (200, 153, 2)

        fpath = join(TEST_DATA_DIR, 'format_def.vcf')
        # check GT
        for var_mats in _create_var_mat_objs_from_vcf(fpath, {}):
            chunks = list(var_mats.iterate_chunks())
            chunk = chunks[0]
            assert chunk['/calls/GT'].shape == (5, 3, 2)
            assert numpy.all(chunk['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])

    def test_delete_item_from_variationArray(self):
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, pre_read_max_size=1000)
        snps = VariationsArrays(ignore_undefined_fields=True)
        snps.put_vars(vcf)
        del snps['/calls/GT']
        assert '/calls/GT' not in snps.keys()
        vcf_fhand.close()

    def test_metadata(self):
        for klass in VAR_MAT_CLASSES:
            fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
            vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=1000,
                                   kept_fields=['/calls/GT'])
            var_mat = _init_var_mat(klass)
            var_mat.put_vars(vcf_parser)
            metadata = var_mat.metadata
            assert '/variations/filter/q10' in metadata.keys()

            for klass in VAR_MAT_CLASSES:
                out_snps = _init_var_mat(klass)
                out_snps.put_chunks(var_mat.iterate_chunks())
                assert '/variations/filter/q10' in out_snps.keys()
                out_snps.put_chunks(var_mat.iterate_chunks())

            fhand.close()

    def test_max_field_str_len(self):
        fhand = gzip.open(join(TEST_DATA_DIR, 'tomato.apeki_gbs.calmd.vcf.gz'),
                          'rb')
        vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=1000,
                               max_field_lens={'alt': 5})
        expected = {'INFO': {b'TYPE': 7, b'CIGAR': 6}, 'alt': 4, 'chrom': 10,
                    'id': 10, 'ref': 0, 'FILTER': 0}
        assert vcf_parser.max_field_str_lens == expected

    def test_vcf_to_hdf5(self):
        tmp_fhand = NamedTemporaryFile()
        path = tmp_fhand.name
        tmp_fhand.close()

        fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf_parser = VCFParser(fhand=fhand, max_field_lens={'alt': 5},
                               n_threads=None, pre_read_max_size=1000)
        h5 = VariationsH5(path, mode='w', ignore_undefined_fields=True)
        h5.put_vars(vcf_parser)
        fhand.close()

        h5 = VariationsH5(path, 'r')
        assert h5['/calls/GT'].shape == (5, 3, 2)
        assert numpy.all(h5['/calls/GT'][1] == [[0, 0], [0, 1], [0, 0]])

        expected = numpy.array([[[51, 51], [51, 51], [-1, -1]],
                                [[58, 50], [65, 3], [-1, -1]],
                                [[23, 27], [18, 2], [-1, -1]],
                                [[56, 60], [51, 51], [-1, -1]],
                                [[-1, -1], [-1, -1], [-1, -1]]],
                               dtype=numpy.int16)
        assert numpy.all(h5['/calls/HQ'][:] == expected)
        expected = numpy.array([48, 48, 43], dtype=numpy.int16)
        assert numpy.all(h5['/calls/GQ'][0, :] == expected)

        # Variations filters fields
        expected = numpy.array([False, True, False, False, False])
        assert numpy.all(h5['/variations/filter/q10'][:] == expected)
        expected = numpy.array([False, False, False, False, False])
        assert numpy.all(h5['/variations/filter/s50'][:] == expected)
        expected = [True, False, True, True, True]
        assert numpy.all(h5['/variations/filter/PASS'][:] == expected)

        # Variations info fields
        expected = remove_nans(numpy.array([[0.5, numpy.nan],
                                            [0.01699829, numpy.nan],
                                            [0.33300781, 0.66699219],
                                            [numpy.nan, numpy.nan],
                                            [numpy.nan, numpy.nan]],
                                           dtype=numpy.float16))

        af = remove_nans(h5['/variations/info/AF'][:])
        assert numpy.all(af == expected)
        expected = numpy.array([3, 3, 2, 3, 3])
        assert numpy.all(h5['/variations/info/NS'][:] == expected)
        expected = numpy.array([14, 11, 10, 13, 9])
        assert numpy.all(h5['/variations/info/DP'][:] == expected)
        expected = numpy.array([True, False, True, False, False])
        assert numpy.all(h5['/variations/info/DB'][:] == expected)
        expected = numpy.array([True, False, False, False, False])
        assert numpy.all(h5['/variations/info/H2'][:] == expected)

        os.remove(path)
        # With another file
        tmp_fhand = NamedTemporaryFile()
        path = tmp_fhand.name
        tmp_fhand.close()

        fhand = open(join(TEST_DATA_DIR, 'phylome.sample.vcf'), 'rb')
        vcf_parser = VCFParser(fhand=fhand, pre_read_max_size=1000)
        h5 = VariationsH5(path, mode='w', ignore_undefined_fields=True)
        h5.put_vars(vcf_parser)
        fhand.close()
        h5 = h5py.File(path, 'r')

        assert numpy.all(h5['/calls/GT'].shape == (2, 42, 2))
        assert numpy.all(h5['/calls/GT'][1, 12] == [1, 1])
        assert numpy.all(h5['/calls/GL'][0, 0, 0] == 0)
        os.remove(path)


class VcfWrittenTest(unittest.TestCase):

    def test_write_vcf(self):
        # With missing info in variations
        tmp_fhand = NamedTemporaryFile()
        out_fpath = tmp_fhand.name
        tmp_fhand.close()
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def_without_info.vcf'),
                         'rb')
        vcf = VCFParser(vcf_fhand)

        max_field_lens = {'INFO': {}, 'CALLS': {b'GQ': 1, b'GT': 1, b'HQ': 2, b'DP': 1}, 'FILTER': 1, 'alt': 2}
        max_field_str_lens = {'ref': 4, 'INFO': {}, 'id': 10, 'FILTER': 0, 'alt': 5, 'chrom': 2}

        h5_without_info = VariationsH5(fpath=out_fpath, mode='w',
                                       ignore_undefined_fields=True)
        h5_without_info.put_vars(vcf, max_field_lens=max_field_lens,
                                 max_field_str_lens=max_field_str_lens)
        vcf_fhand.close()

        lines = _to_vcf(h5_without_info, vcf_format='VCFv4.0')
        exp_fhand = open(join(TEST_DATA_DIR,
                              'format_def_without_info_exp.vcf'), 'r')
        exp_lines = list(exp_fhand)
        exp_lines = [line.strip() for line in exp_lines]
        for line in lines:
            assert line in exp_lines

        exp_fhand.close()
        # With all fields available
        tmp_fhand = NamedTemporaryFile()
        tmp_fhand.close()
        vcf_fhand = open(join(TEST_DATA_DIR, 'format_def.vcf'), 'rb')
        vcf = VCFParser(vcf_fhand, max_field_lens={'alt': 2},
                        pre_read_max_size=10000)

        max_field_lens = {'CALLS': {b'GT': 1, b'HQ': 2, b'DP': 1, b'GQ': 1}, 'FILTER': 1, 'INFO': {b'AA': 1, b'AF': 2, b'DP': 1, b'DB': 1, b'NS': 1, b'H2': 1}, 'alt': 2}
        max_field_str_lens = {'INFO': {}, 'alt': 5, 'chrom': 2, 'ref': 4, 'id': 10, 'FILTER': 0}

        variations = VariationsArrays(ignore_undefined_fields=True)
        variations.put_vars(vcf, max_field_lens=max_field_lens,
                            max_field_str_lens=max_field_str_lens)
        vcf_fhand.close()

        lines = _to_vcf(variations, vcf_format='VCFv4.0')
        exp_fhand = open(join(TEST_DATA_DIR,
                              'format_def_exp.vcf'), 'r')
        exp_lines = list(exp_fhand)
        exp_lines = [line.strip() for line in exp_lines]
        for line in lines:
            assert line in exp_lines
        exp_fhand.close()
        # With hdf5 file
        tomato_h5 = VariationsH5(join(TEST_DATA_DIR,
                                      'tomato.apeki_gbs.calmd.h5'), "r")
        exp_fhand = open(join(TEST_DATA_DIR, "tomato.apeki_100_exp.vcf"), "r")
        lines = _to_vcf(tomato_h5)

        # only checking snps
        exp_lines = list(exp_fhand)
        exp_lines = [line.strip() for line in exp_lines if not line.startswith('#')]
        i = 0
        for line in lines:
            if line.startswith('#'):
                continue
            assert line == exp_lines[i]
            i += 1
            if i > 42:
                break
        exp_fhand.close()

    def test_write_header(self):
        files = ['format_def_without_info.vcf',
                 'format_def_without_filter.vcf',
                 'format_without_flt_info_qual.vcf']
        for file in files:
            vcf_fhand = open(join(TEST_DATA_DIR, file), 'rb')
            header_lines = [line.strip() for line in vcf_fhand
                            if line.startswith(b'##')]
            vcf_fhand.close()
            vcf_fhand = open(join(TEST_DATA_DIR, file), 'rb')
            vcf = VCFParser(vcf_fhand, max_field_lens={'alt': 2},
                            pre_read_max_size=10000)
            var_array = VariationsArrays(ignore_undefined_fields=True)
            var_array.put_vars(vcf)
            for line in _prepare_vcf_header(var_array, vcf_format='VCFv4.0'):
                assert line.encode('utf-8') in header_lines
            vcf_fhand.close()


class Mat012Test(unittest.TestCase):
    def test_mat012(self):
        gts = numpy.array([[[0, 0], [0, 1], [2, 2], [-1, 3]],
                           [[0, 0], [0, 0], [1, 1], [2, 2]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        varis = VariationsArrays()
        varis[GT_FIELD] = gts
        gts012 = varis.gts_as_mat012
        expected = [[0, 1, 2, -1], [0, 0, 2, 2], [-1, -1, -1, -1]]
        assert numpy.allclose(gts012, expected, equal_nan=True)


if __name__ == "__main__":
    # import sys; sys.argv = ['', 'VcfH5Test.test_create_hdf5_with_chunks']
    unittest.main()
