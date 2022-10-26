# Based on https://github.com/Wells-Group/asimov-custom-assemblers/blob/main/python/setup.py

import os
import subprocess
import platform
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION = "0.1.0"
REQUIREMENTS = []


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the"
                               + "following extensions:"
                               + ", ".join(e.name for e in self.extensions))
        if platform.system() == "Windows":
            raise RuntimeError("Windows is not supported")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j3']

        env = os.environ.copy()

        # default to 3 build threads
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
            env["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"

        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp, env=env)


setup(name='dolfinx_hdg',
      version=VERSION,
      description='Custom assemblers for HDG',
      author='Joseph P. Dean',
      author_email='jpd62@cam.ac.uk',
      python_requires='>3.6.0',
      packages=['dolfinx_hdg'],
      ext_modules=[CMakeExtension('dolfinx_hdg.cpp')],
      cmdclass=dict(build_ext=CMakeBuild),
      install_requires=REQUIREMENTS,
      setup_requires=["pybind11"],
      zip_safe=False)
