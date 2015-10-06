from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [ ]

ext_modules += cythonize([Extension("variation.matrix.count_alleles",
                                    sources=["variation/matrix/count_alleles.pyx"],
                                    include_dirs=[numpy.get_include(), '.'])])

cmdclass = { 'build_ext': build_ext }
setup(
    #packages=['hola'],
    #package_dir=[]
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
