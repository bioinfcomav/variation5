import unittest
from os.path import join
from tempfile import NamedTemporaryFile
import io
import re

import numpy

from variation import GT_FIELD, POS_FIELD, CHROM_FIELD, REF_FIELD, ALT_FIELD
from variation.variations.vars_matrices import VariationsArrays, VariationsH5
from variation.gt_parsers.vcf import VCFParser
from variation.gt_writers.vcf import (_write_vcf_header, _write_vcf_meta,
                                      write_vcf, write_vcf_parallel)
from variation.gt_writers.excel import write_excel
from variation.gt_writers.fasta import write_fasta

from test.test_utils import TEST_DATA_DIR


class VcfWrittenTest(unittest.TestCase):

    def test_write_meta_header(self):
        files = ['format_def_without_info.vcf',
                 'format_def_without_filter.vcf',
                 'format_without_flt_info_qual.vcf']
        for file in files:
            vcf_fhand = open(join(TEST_DATA_DIR, file), 'rb')
            header_lines = [line for line in vcf_fhand if line.startswith(b'#')]
            vcf_fhand.close()
            with open(join(TEST_DATA_DIR, file), 'rb') as vcf_fhand:
                vcf = VCFParser(vcf_fhand)
                var_array = VariationsArrays(ignore_undefined_fields=True)
                var_array.put_vars(vcf)
                with NamedTemporaryFile(suffix='.h5') as tmp_fhand:
                    _write_vcf_meta(var_array, tmp_fhand, vcf_format='VCFv4.0')
                    _write_vcf_header(var_array, tmp_fhand)
                    tmp_fhand.flush()
                    with open(tmp_fhand.name, 'rb') as retmp_fhand:
                        for line in retmp_fhand:
                            assert line in header_lines

    def test_generated_vcf_feed_outputs_equal_vcfs(self):
        h5_vars = VariationsH5(join(TEST_DATA_DIR,
                                    'tomato.apeki_gbs.calmd.1stchunk.h5'), "r")
        with NamedTemporaryFile(mode='wb') as vcf_vars_from_h5:
            write_vcf(h5_vars, vcf_vars_from_h5)
            vcf_vars_from_h5.flush()
            vcf_fhand = open(vcf_vars_from_h5.name, 'rb')
            vcf = VCFParser(vcf_fhand)
            vcf_vars_parsed = VariationsArrays()
            vcf_vars_parsed.put_vars(vcf)
            with NamedTemporaryFile(mode='wb') as vcf_vars_from_vcf:
                vcf_vars_parsed.write_vcf(vcf_vars_from_vcf)
                vcf_vars_from_vcf.flush()
                vcf_from_h5_fhand = open(vcf_vars_from_h5.name, 'rb')
                vcf_from_vcf_fhand = open(vcf_vars_from_vcf.name, 'rb')
                for line_parsed_from_h5, line_parsed_from_vcf in zip(vcf_from_h5_fhand, vcf_from_vcf_fhand):
                    assert line_parsed_from_h5 == line_parsed_from_vcf, "when importing from a generated VCF and exporting to a new VCF both files must be the same"

    def test_parallel_vcf_write(self):
        tomato_h5 = VariationsH5(join(TEST_DATA_DIR,
                                      'tomato.apeki_gbs.calmd.1stchunk.h5'),
                                 "r")
        with NamedTemporaryFile(mode='wb') as tmp_fhand:
            write_vcf_parallel(tomato_h5, tmp_fhand, n_threads=4,
                               tmp_dir='/tmp')
            vcf_line_regex = re.compile('(.*\t){9}.*\n')
            tmp_fhand.flush()
            vcf_from_tmp_fhand = open(tmp_fhand.name, 'rb')
            i = 0
            for line in vcf_from_tmp_fhand:
                if line.startswith(b'#'):
                    continue
                assert vcf_line_regex.match(line.decode())
                i += 1
                if i > 42:
                    break
            tmp_fhand.close()
            vcf_from_tmp_fhand.close()

    def xtest_real_file(self):
        fpath = '/home/peio/work_in/test_variation5/write_vcf/original.h5'
        vcf_fpath = '/home/peio/work_in/test_variation5/write_vcf/traditom_tier1.vcf'
        out_fhand = open(vcf_fpath, 'w')
#         kept_fields = ['/variations/chrom', '/variations/pos', '/variations/ref',
#                        '/variations/alt', '/variations/qual', '/calls/GT',
#                        '/calls/GQ', '/calls/DP', '/calls/AO', '/calls/RO']
#         vcfparser = VCFParser(open(vcf_fpath, 'rb'), pre_read_max_size=10000)
        h5 = VariationsH5(fpath=fpath, mode='r')
        # h5.put_vars(vcfparser)
        h5.write_vcf(out_fhand)


class ExcelTest(unittest.TestCase):

    def test_excel(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 1], [1, 1], [0, 0], [0, 0]],
                           [[2, 2], [2, 0], [2, 1], [0, 0], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1]],
                           [[0, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        variations[GT_FIELD] = gts
        variations.samples = list(range(gts.shape[1]))

        fhand = NamedTemporaryFile(suffix='.xlsx')
        write_excel(variations, fhand)

        # chrom pos
        variations[CHROM_FIELD] = numpy.array([1, 1, 2, 2])
        variations[POS_FIELD] = numpy.array([10, 20, 10, 20])
        fhand = NamedTemporaryFile(suffix='.xlsx')
        write_excel(variations, fhand)

        # REF, ALT
        variations[REF_FIELD] = numpy.array([b'A', b'A', b'A', b'A'])
        variations[ALT_FIELD] = numpy.array([[b'T'], [b'T'], [b'T'], [b'T']])
        write_excel(variations, fhand)

        # with classifications
        classes = [1, 1, 1, 2, 2]
        write_excel(variations, fhand, classes)


class FastaWriterTest(unittest.TestCase):

    def test_fasta_writer(self):
        variations = VariationsArrays()
        gts = numpy.array([[[0, 0], [0, 1], [1, 1], [0, 0], [0, 0]],
                           [[2, 2], [1, 1], [-1, 2], [0, 0], [-1, -1]],
                           [[0, 1], [0, 0], [0, 0], [1, 1], [0, 0]],
                           [[0, 0], [1, -1], [1, 1], [1, -1], [1, 1]],
                           [[0, 1], [0, 1], [-1, -1], [-1, 1], [1, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                           [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                           ])
        ref = numpy.array(['C', 'G', 'A', 'T', 'T', 'C', 'C'])
        alt = numpy.array([['A', 'TT'], ['A', 'T'], ['C', ''], ['G', ''],
                           ['A', 'T'], ['G', ''], ['G', '']])
        variations[GT_FIELD] = gts
        variations[ALT_FIELD] = alt
        variations[REF_FIELD] = ref
        variations[CHROM_FIELD] = numpy.array(['ch1', 'ch2', 'ch2', 'ch2',
                                               'ch2', 'ch3', 'ch3'])
        variations[POS_FIELD] = numpy.array([10, 20, 30, 40, 50, 10, 15])
        variations.samples = list(map(str, range(gts.shape[1])))

        fhand = io.BytesIO()
        write_fasta(variations, fhand, remove_sites_all_N=True,
                    remove_invariant_snps=True)
        # SNPS
        # C A TT
        # G A T
        # A C
        # T G
        # T A T
        # C G
        # N
        # indi1> TNT
        # indi2> AAN
        # indi3> NAG
        # indi4> GCN
        # indi5> NAG
        result = fhand.getvalue().splitlines()
        assert b'>0' in result[0]
        assert result[1] == b'TNT'
        assert b'>1' in result[2]
        assert result[3] == b'AAN'
        assert b'>2' in result[4]
        assert result[5] == b'NAG'
        assert b'>3' in result[6]
        assert result[7] == b'GCN'
        assert b'>4' in result[8]
        assert result[9] == b'NAG'

        fhand = io.BytesIO()
        write_fasta(variations, fhand, remove_invariant_snps=True)
        result = fhand.getvalue().splitlines()
        assert b'>0' in result[0]
        assert result[1] == b'TNT'

        fhand = io.BytesIO()
        write_fasta(variations, fhand, remove_sites_all_N=True)
        result = fhand.getvalue().splitlines()
        assert b'>0' in result[0]
        assert result[1] == b'TNTC'


if __name__ == "__main__":
    # import sys; sys.argv = ['', 'VcfWrittenTest.test_write_vcf_from_h5']
    unittest.main()
