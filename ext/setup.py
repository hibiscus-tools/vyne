from setuptools import setup
from torch.utils import cpp_extension

setup(name='vyne',
    ext_modules=[
        cpp_extension.CppExtension('vyne', [
            'source/vyne.cpp'
        ])
    ],
    include_dirs=[
        '..',
        '../dependencies',
        '../dependencies/oak/include',
        '../dependencies/oak/dependencies/glm',
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension,
    },
    version='1.0.0')