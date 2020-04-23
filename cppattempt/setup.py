from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension,CUDAExtension

setup(name='point',
      ext_modules=[
            CUDAExtension('point', [
            'point_api.cpp',
            'Point.cpp',
            'Point_cuda.cu',
            ]),
       ],
      cmdclass={'build_ext': BuildExtension})
