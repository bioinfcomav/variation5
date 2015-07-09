
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from tempfile import NamedTemporaryFile

import numpy

from variation.plot import plot_histogram


class PlotTest(unittest.TestCase):
    def test_histogram(self):
        numbers = numpy.random.normal(size=(10000,))
        with NamedTemporaryFile(suffix='.png') as fhand:
            plot_histogram(numbers, bins=40, fhand=fhand)
            fhand.flush()
            read_fhand = open(fhand.name, 'rb')
            assert b'\x89PNG\r\n' in read_fhand.readline()
            fhand.close()
            read_fhand.close()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_vcf_parsing']
    unittest.main()
