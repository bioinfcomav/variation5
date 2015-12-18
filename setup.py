from distutils.core import setup
from Cython.Build import cythonize


setup(
    packages=['variation.gt_parsers'],
    ext_modules=cythonize("variation/gt_parsers/vcf_field_parsers.pyx"),
)
