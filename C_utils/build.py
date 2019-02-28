import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = [osp.join(abs_path, 'build/libsift.so')]
extra_objects += glob.glob('/usr/local/cuda-9.0/lib64/*.a')

ffi = create_extension(
    'libsift',
    headers=['include/PointSift.h'],
    sources=['src/PointSift.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[osp.join(abs_path, 'include'),"/opt/cuda/include"]
)


if __name__ == '__main__':
    print("COMPILING C ")
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
