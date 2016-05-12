
# Method could be a function
# pylint: disable=R0201
# Too many public methods
# pylint: disable=R0904
# Missing docstring
# pylint: disable=C0111

import unittest
from os.path import join


from variation.variations.pipeline import Pipeline
from variation.variations.filters import flt_hist_min_called_gts
from variation.variations.vars_matrices import VariationsH5, VariationsArrays
from test.test_utils import TEST_DATA_DIR


class PipelineTest(unittest.TestCase):
    def test_pipeline(self):
        pipeline = Pipeline()
        hdf5 = VariationsH5(join(TEST_DATA_DIR, 'ril.hdf5'), mode='r')
        pipeline.append(flt_hist_min_called_gts, kwargs={'min_': 0.1,
                                                         'range_': (0, 1)},
                        id_='filter1')
        vars_out = VariationsArrays()
        result = pipeline.run(hdf5, vars_out)
        print(result)

        # TODO check same result with no pipeline
        # TODO check with no range set

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'FilterTest']
    unittest.main()
