from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize('base.pyx', annotate=True),
    include_dirs=[numpy.get_include()],
    python_requires='>=3.11',
)