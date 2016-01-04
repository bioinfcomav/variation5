from distutils.core import setup
from Cython.Build import cythonize


setup(
    packages=['variation.gt_parsers', 'variation.variations',
              'variation.matrix', 'variation.utils'],
    ext_modules=cythonize("variation/gt_parsers/vcf_field_parsers.pyx"),
)
