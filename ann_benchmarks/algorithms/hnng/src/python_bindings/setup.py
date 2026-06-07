"""Build the hnnglib Python extension.

Build in place:
    python setup.py build_ext --inplace

Test:
    pytest tests/
"""
import os
import sys
import sysconfig
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = "0.1"

ext_modules = [
    Extension(
        "hnnglib",
        sources=["bindings.cpp"],
        language="c++",
    ),
]


def has_flag(compiler, flag):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main(){return 0;}\n")
        name = f.name
    try:
        compiler.compile([name], extra_postargs=[flag])
        return True
    except Exception:
        return False
    finally:
        try:
            os.remove(name)
        except OSError:
            pass


def cpp_std_flag(compiler):
    for flag in ("-std=c++17", "-std=c++14"):
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError("Need at least C++14 support")


class BuildExt(build_ext):
    c_opts = {
        "unix": ["-O3", "-march=native", "-fpermissive"],
        "msvc": ["/EHsc", "/O2"],
    }
    link_opts = {"unix": ["-pthread"], "msvc": []}

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = list(self.c_opts.get(ct, []))
        link_opts = list(self.link_opts.get(ct, []))
        if ct == "unix":
            opts.append(cpp_std_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        import pybind11
        import numpy as np
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)
            ext.include_dirs.extend([
                pybind11.get_include(),
                np.get_include(),
            ])
        super().build_extensions()


setup(
    name="hnnglib",
    version=__version__,
    description="Hierarchical Nearest Neighbor Graph (HNNG) Python bindings",
    long_description="HNNG: exact k-NN via cluster-radius pruning over a hierarchy of 1-NNGs.",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6", "numpy"],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
