#!/usr/bin/env python3
import importlib.util
from distutils.command.build import build as DistutilsBuild
from os.path import abspath, join, dirname, realpath
from setuptools import find_packages, setup, Extension
import numpy as np
import os
from Cython.Build import cythonize


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dirname(realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


cpu_extension = Extension(
    'mujoco_py.cymj',
    sources=["mujoco_py/cymj.pyx", "mujoco_py/gl/osmesashim.c"],
    include_dirs=[np.get_include()],
    libraries=['mujoco150', 'glewosmesa', 'OSMesa'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language='c'
)

gpu_extension = Extension(
    'mujoco_py.cymj',
    sources=["mujoco_py/cymj.pyx", "mujoco_py/gl/eglshim.c"],
    include_dirs=[np.get_include(), "mujoco_py/vendor/egl"],
    libraries=['mujoco150', 'glewegl'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language='c'
)

# read environment variables
extension = gpu_extension if os.getenv('MUJOCO_BUILD_GPU') else cpu_extension

setup(
    name='mujoco-py',
    version='1.50.1.1',
    author='OpenAI Robotics Team',
    author_email='robotics@openai.com',
    url='https://github.com/openai/mujoco-py',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements_file('requirements.txt'),
    setup_requires=read_requirements_file('requirements.txt'),
    tests_require=read_requirements_file('requirements.dev.txt'),
    ext_modules=cythonize(extension, build_dir="mujoco_py/generated"),
)
