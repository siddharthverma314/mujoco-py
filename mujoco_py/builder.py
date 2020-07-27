import distutils
import imp
import os
import shutil
import subprocess
import sys
from distutils.core import Extension
from distutils.dist import Distribution
from distutils.sysconfig import customize_compiler
from os.path import abspath, dirname, exists, join, getmtime
from os import getenv
from shutil import move

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils.old_build_ext import old_build_ext as build_ext

from mujoco_py.utils import discover_mujoco


def load_cython_ext(mjpro_path):
    """
    Loads the cymj Cython extension. This is safe to be called from
    multiple processes running on the same machine.

    Cython only gives us back the raw path, regardless of whether
    it found a cached version or actually compiled. Since we do
    non-idempotent postprocessing of the DLL, be extra careful
    to only do that once and then atomically move to the final
    location.
    """
    if ('glfw' in sys.modules and
            'mujoco' in abspath(sys.modules["glfw"].__file__)):
        print('''
WARNING: Existing glfw python module detected!

MuJoCo comes with its own version of GLFW, so it's preferable to use that one.

The easy solution is to `import mujoco_py` _before_ `import glfw`.
''')

    if getenv('MUJOCO_BUILD_GPU'):
        Builder = LinuxGPUExtensionBuilder
    else:
        Builder = LinuxCPUExtensionBuilder

    builder = Builder(mjpro_path)
    cext_so_path = builder.get_so_file_path()
    #if not exists(cext_so_path):
    #    cext_so_path = builder.build()
    mod = imp.load_dynamic("cymj", cext_so_path)
    return mod


class custom_build_ext(build_ext):
    """
    Custom build_ext to suppress the "-Wstrict-prototypes" warning.
    It arises from the fact that we're using C++. This seems to be
    the cleanest way to get rid of the extra flag.

    See http://stackoverflow.com/a/36293331/248400
    """

    def build_extensions(self):
        customize_compiler(self.compiler)

        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


def fix_shared_library(so_file, name, library_path):
    ldd_output = subprocess.check_output(
        ['ldd', so_file]).decode('utf-8')

    if name in ldd_output:
        subprocess.check_call(['patchelf',
                               '--remove-needed',
                               name,
                               so_file])
    subprocess.check_call(
        ['patchelf', '--add-needed',
         library_path,
         so_file])


class MujocoExtensionBuilder():

    CYMJ_DIR_PATH = abspath(dirname(__file__))

    def __init__(self, mjpro_path):
        self.mjpro_path = mjpro_path
        self.extension = Extension(
            'mujoco_py.cymj',
            sources=[join(self.CYMJ_DIR_PATH, "cymj.pyx")],
            include_dirs=[
                self.CYMJ_DIR_PATH,
                join(mjpro_path, 'include'),
                np.get_include(),
            ],
            libraries=['mujoco150'],
            library_dirs=[join(mjpro_path, 'bin')],
            extra_compile_args=[
                '-fopenmp',  # needed for OpenMP
                '-w',  # suppress numpy compilation warnings
            ],
            extra_link_args=['-fopenmp'],
            language='c')

    def build(self):
        built_so_file_path = self._build_impl()
        new_so_file_path = self.get_so_file_path()
        move(built_so_file_path, new_so_file_path)
        return new_so_file_path

    def _build_impl(self):
        dist = Distribution({
            "script_name": None,
            "script_args": ["build_ext"]
        })
        dist.ext_modules = cythonize([self.extension])
        dist.include_dirs = []
        dist.cmdclass = {'build_ext': custom_build_ext}
        build = dist.get_command_obj('build')
        # following the convention of cython's pyxbuild and naming
        # base directory "_pyxbld"
        build.build_base = join(self.CYMJ_DIR_PATH, 'generated',
                                '_pyxbld_%s' % self.__class__.__name__)
        dist.parse_command_line()
        obj_build_ext = dist.get_command_obj("build_ext")
        dist.run_commands()
        built_so_file_path, = obj_build_ext.get_outputs()
        return built_so_file_path

    def get_so_file_path(self):
        dir_path = abspath(dirname(__file__))
        return join(dir_path, "generated", "cymj_%s.so" % (
            self.__class__.__name__.lower()))


class LinuxCPUExtensionBuilder(MujocoExtensionBuilder):

    def __init__(self, mjpro_path):
        super().__init__(mjpro_path)

        self.extension.sources.append(
            join(self.CYMJ_DIR_PATH, "gl", "osmesashim.c"))
        self.extension.libraries.extend(['glewosmesa', 'OSMesa'])
        self.extension.runtime_library_dirs = [join(mjpro_path, 'bin')]


class LinuxGPUExtensionBuilder(MujocoExtensionBuilder):

    def __init__(self, mjpro_path):
        super().__init__(mjpro_path)

        self.extension.sources.append(self.CYMJ_DIR_PATH + "/gl/eglshim.c")
        self.extension.include_dirs.append(self.CYMJ_DIR_PATH + '/vendor/egl')
        self.extension.libraries.extend(['glewegl'])
        self.extension.runtime_library_dirs = [join(mjpro_path, 'bin')]

    def _build_impl(self):
        so_file_path = super()._build_impl()
        nvidia_lib_dir = '/usr/local/nvidia/lib64/'
        fix_shared_library(so_file_path, 'libOpenGL.so',
                           join(nvidia_lib_dir, 'libOpenGL.so.0'))
        fix_shared_library(so_file_path, 'libEGL.so',
                           join(nvidia_lib_dir, 'libEGL.so.1'))
        return so_file_path


class MujocoException(Exception):
    pass


def user_warning_raise_exception(warn_bytes):
    '''
    User-defined warning callback, which is called by mujoco on warnings.
    Here we have two primary jobs:
        - Detect known warnings and suggest fixes (with code)
        - Decide whether to raise an Exception and raise if needed
    More cases should be added as we find new failures.
    '''
    # TODO: look through test output to see MuJoCo warnings to catch
    # and recommend. Also fix those tests
    warn = warn_bytes.decode()  # Convert bytes to string
    if 'Pre-allocated constraint buffer is full' in warn:
        raise MujocoException(warn + 'Increase njmax in mujoco XML')
    if 'Pre-allocated contact buffer is full' in warn:
        raise MujocoException(warn + 'Increase njconmax in mujoco XML')
    raise MujocoException('Got MuJoCo Warning: {}'.format(warn))


def user_warning_ignore_exception(warn_bytes):
    pass


class ignore_mujoco_warnings:
    """
    Class to turn off mujoco warning exceptions within a scope. Useful for
    large, vectorized rollouts.
    """

    def __enter__(self):
        self.prev_user_warning = cymj.get_warning_callback()
        cymj.set_warning_callback(user_warning_ignore_exception)
        return self

    def __exit__(self, type, value, traceback):
        cymj.set_warning_callback(self.prev_user_warning)


key_path = discover_mujoco()
from mujoco_py import cymj


# Trick to expose all mj* functions from mujoco in mujoco_py.*
class dict2(object):
    pass


functions = dict2()
for func_name in dir(cymj):
    if func_name.startswith("_mj"):
        setattr(functions, func_name[1:], getattr(cymj, func_name))

functions.mj_activate(key_path)

# Set user-defined callbacks that raise assertion with message
cymj.set_warning_callback(user_warning_raise_exception)
