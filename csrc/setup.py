from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name='fake_fp8quant_tools',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'fake_fp8quant_tools',
            ['quanttools.cpp', 'fake_quant.cu'],
            extra_compile_args={
                'cxx': ['-g', '-lineinfo', '-std=c++17'],
                'nvcc': ['-O2', '-g', '-Xcompiler', '-rdynamic', '-lineinfo','--std=c++17' ,'-DENABLE_FP8_E5M2']
            })
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
