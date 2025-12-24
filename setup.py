"""ActionFormer package setup with C++ extensions."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='actionformer',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='nms_1d_cpu',
            sources=['libs/utils/csrc/nms_cpu.cpp'],
            extra_compile_args=['-fopenmp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'pyyaml>=6.0',
        'tensorboard>=2.10.0',
        'tqdm>=4.65.0',
        'pandas>=1.5.0',
        'scipy>=1.10.0',
    ],
)
