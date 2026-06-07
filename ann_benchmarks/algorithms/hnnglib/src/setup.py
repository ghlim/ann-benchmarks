import os
import sys
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __str__(self):
        import pybind11
        return pybind11.get_include()


ext_modules = [
    Extension(
        'hnnglib',
        sources=['python_bindings/bindings.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            # Path to hnnglib headers
            'hnnglib',
        ],
        language='c++',
        extra_compile_args=[],
    ),
]


# Compiler-specific options
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except Exception:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++11 compiler flag."""
    flags = ['-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/fp:fast'],
        'unix': ['-O3', '-ffast-math', '-march=native'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        build_ext.build_extensions(self)


# Read README for long description
def read_file(filename):
    try:
        with open(filename, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''


setup(
    name='hnnglib',
    version='0.1.0',
    author='Gi Hyun Lim',
    author_email='your.email@example.com',
    description='Hierarchical Nearest Neighbor Graph library for approximate nearest neighbor search',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/hnnglib',
    license='MIT',
    
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['numpy>=1.14.0'],
    
    python_requires='>=3.6',
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    
    keywords='nearest-neighbor-search ann nng hierarchy',
    
    zip_safe=False,
)
