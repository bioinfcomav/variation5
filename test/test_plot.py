
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from tempfile import NamedTemporaryFile

import numpy

from variation.plot import plot_histogram, calc_boxplot_stats, plot_boxplot
from os.path import join
from test.test_utils import TEST_DATA_DIR
from variation.variations.vars_matrices import VariationsH5
from variation.variations.stats import calc_quality_by_depth


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

    def test_calc_boxplot_stats(self):
        data = numpy.ones((1, 100))
        stats = calc_boxplot_stats(data)[0]
        assert stats['med'] == 50
        assert stats['q1'] == 25
        assert stats['q3'] == 75
        assert stats['mean'] == 49.5

    def xtest_plot_boxplot(self):
        data = numpy.ones((40, 100))
        fhand = open(join(TEST_DATA_DIR, 'prueba.png'), mode='w')
        bp = plot_boxplot(data, fhand=fhand, mpl_params={'set_xlim':
                                                         {'args': [None, 60],
                                                          'kwargs': {}}},
                          figsize=[16,12])
        fhand.close()

    def test_plot_boxplot(self):
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        dist, cum = calc_quality_by_depth(hdf5, depths=range(1, 30))
        fhand = open(join(TEST_DATA_DIR, 'pruebadeverdadverdadera.png'),
                     mode='w')
        mpl_params = {'set_xlabel':{'args':['Depth'], 'kwargs':{}},
                      'set_ylabel':{'args':['Genotype quality'], 'kwargs':{}},
                      'set_title':{'args':['Depth dependent GQ distribution'], 'kwargs':{}}}
        bp = plot_boxplot(dist, fhand=fhand, mpl_params=mpl_params)

if __name__ == "__main__":
    import sys;sys.argv = ['', 'PlotTest.test_plot_boxplot']
    unittest.main()
