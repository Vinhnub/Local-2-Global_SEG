from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Add numpy include dir explicitly
extensions = [
    Extension(
        "hamming",
        ["hamming.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
