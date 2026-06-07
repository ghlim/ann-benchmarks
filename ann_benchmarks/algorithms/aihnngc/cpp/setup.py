"""setup.py -- build the `hnng` pybind11 extension.

Compiles and is tested: built with g++ 13.3 / pybind11 3.0.4; the full suite
(incl. C++/Python parity) runs in CI. This build recipe mirrors the hnswlib
ann-benchmarks packaging (pybind11.setup_helpers). See README.md.

`pip install .` (or `python setup.py build_ext --inplace`) compiles
bindings/hnng_py.cpp against the header-only core in include/ and produces the
importable `hnng` module.
"""

from __future__ import annotations

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Release optimization flags. -O3 on POSIX, /O2 on MSVC. -march=native is left
# OFF by default for portable wheels; enable via the HNNG_NATIVE env var when
# building on the run host (e.g. inside the ann-benchmarks Docker image).
import os

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args += ["/O2", "/std:c++17"]
else:
    extra_compile_args += ["-O3", "-funroll-loops", "-std=c++17"]
    if os.environ.get("HNNG_NATIVE", "0") == "1":
        extra_compile_args += ["-march=native"]

ext_modules = [
    Pybind11Extension(
        "hnng",
        sources=["bindings/hnng_py.cpp"],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="hnng",
    version="0.1.0",
    description=(
        "Exact hierarchical Nearest Neighbor Graph (hNNG) index — C++/pybind11 "
        "core mirroring the hnng_ref reference."
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    # pybind11 floor matches the CMakeLists.txt FetchContent pin (v2.12.0).
    setup_requires=["pybind11>=2.12.0"],
    install_requires=["numpy>=1.21,<3"],
    zip_safe=False,
)
