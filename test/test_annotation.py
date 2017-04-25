
import unittest
import numpy

from variation.variations.vars_matrices import VariationsArrays
from variation import GT_FIELD, MISSING_INT, TRUE_INT, FALSE_INT
from variation.variations.annotation import is_variable


class IsVariableTest(unittest.TestCase):

    def test_is_variable_func(self):
        variations = VariationsArrays()
        gts = numpy.array([[[-1, -1], [1, 1], [0, 1], [1, 1], [-1, -1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]]])
        variations[GT_FIELD] = gts
        variations.samples = [1, 2, 3, 4, 5]

        expected_variable = [MISSING_INT, TRUE_INT, TRUE_INT, FALSE_INT]
        variable = is_variable(variations, samples=[1, 5])
        assert numpy.all(variable == expected_variable)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
