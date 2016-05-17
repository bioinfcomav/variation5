
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join

import numpy

from variation.variations.pipeline import Pipeline
from variation.variations.filters import MinCalledGTsFilter
from variation.variations.vars_matrices import VariationsH5, VariationsArrays
from test.test_utils import TEST_DATA_DIR


class PipelineTest(unittest.TestCase):
    def test_pipeline(self):
        pipeline = Pipeline()
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')

        flt = MinCalledGTsFilter(min_called=0.1, range_=(0, 1))
        pipeline.append(flt, id_='filter1')
    
        vars_out = VariationsArrays()
        result = pipeline.run(hdf5, vars_out)

        # check same result with no pipeline
        result2  = flt(hdf5)
        assert numpy.allclose(result['filter1']['counts'], result2['counts'])
        assert numpy.allclose(result['filter1']['edges'], result2['edges'])
        assert numpy.allclose(vars_out['/calls/GT'],
                              result2['flt_vars']['/calls/GT'])

        # check with no range set
        pipeline = Pipeline()
        flt = MinCalledGTsFilter(min_called=0.1)
        pipeline.append(flt, id_='filter1')
    
        vars_out = VariationsArrays()
        result = pipeline.run(hdf5, vars_out)

        result2  = flt(hdf5)
        assert numpy.allclose(result['filter1']['counts'], result2['counts'])
        assert numpy.allclose(result['filter1']['edges'], result2['edges'])
        assert numpy.allclose(vars_out['/calls/GT'],
                              result2['flt_vars']['/calls/GT'])

        # With rates False
        pipeline = Pipeline()
        flt = MinCalledGTsFilter(min_called=20, rates=False)
        pipeline.append(flt, id_='filter1')
    
        vars_out = VariationsArrays()
        result = pipeline.run(hdf5, vars_out)

        result2  = flt(hdf5)
        assert numpy.allclose(result['filter1']['counts'], result2['counts'])
        assert numpy.allclose(result['filter1']['edges'], result2['edges'])
        assert numpy.allclose(vars_out['/calls/GT'],
                              result2['flt_vars']['/calls/GT'])
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'FilterTest']
    unittest.main()
