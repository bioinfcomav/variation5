from distutils.core import setup
import numpy

from Cython.Build import cythonize

CYTHON_MODULES = ["variation/gt_parsers/vcf_field_parsers.pyx"]
# "variation/gt_writers/vcf_field_writer.pyx"

setup(
    packages=[
        "variation.gt_parsers",
        "variation.variations",
        "variation.matrix",
        "variation.utils",
        "variation.gt_writers",
    ],
    ext_modules=cythonize(CYTHON_MODULES),
    include_dirs=[numpy.get_include()],
)
