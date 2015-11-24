import unittest
from os.path import join
from tempfile import NamedTemporaryFile

import numpy

from variation.variations.strand import count_compatible_snps, change_strand
from variation.variations.vars_matrices import VariationsH5

from test.test_utils import TEST_DATA_DIR


class StrandTest(unittest.TestCase):
    def test_count_compatible_snsp_in_strands(self):
        fpath = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        h5 = VariationsH5(fpath, "r")

        custom_alleles = numpy.array([[b'G', b'T'],
                                     [b'G', b'T'],
                                     [b'G', b'T']])
        array_spec_matrix = numpy.array([[True, False, True],
                                         [True, True, False],
                                         [True, True, False]])
        snps_check, counts = count_compatible_snps(h5, array_spec_matrix,
                                                   custom_alleles)
        assert counts == [1, 0, 2]
        assert snps_check == 3

    def test_change_gts_chain(self):
        fp = join(TEST_DATA_DIR, 'csv', 'iupac_ex.h5')
        with NamedTemporaryFile(suffix='.h5') as fhand, open(fp, 'rb') as fh5:
            fhand.write(fh5.read())

            h5 = VariationsH5(fhand.name, "r+")
            original = numpy.array([True, False, True])
            final = numpy.array([True, True, True])

            change_strand(h5, original, final)
            exp_ref = numpy.array([b'T', b'G', b'T'])
            exp_alt = numpy.array([[b'G', b''], [b'', b''], [b'A', b'']])

            assert numpy.all(exp_ref == h5['/variations/ref'][:])
            assert numpy.all(exp_alt == h5['/variations/alt'][:])

if __name__ == "__main__":
    unittest.main()
