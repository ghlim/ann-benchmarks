# CMakeLists.txt -- builds the pybind11 module `hnng`.
#
# Compiles and is tested: the module builds with g++ 13.3 / pybind11 3.0.4 and
# the full suite (incl. C++/Python parity) runs in CI. See README.md.
#
# Usage:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release -j
#
# The resulting module (hnng.<ext>) is importable from the build dir, or install
# via the Python packaging path (pyproject.toml / setup.py) which is preferred
# for `pip install .` and the ann-benchmarks Dockerfile.

cmake_minimum_required(VERSION 3.15)
project(hnng LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -O3 for release; -march=native is opt-in (portability/CI safety) via the
# HNNG_NATIVE option. ann-benchmarks images are usually built on the run host,
# so enabling native there is fine; default OFF to keep wheels portable.
option(HNNG_NATIVE "Enable -march=native (non-portable, faster)" OFF)

# ---------------------------------------------------------------- pybind11
# Prefer a system/pip-installed pybind11; fall back to FetchContent.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via find_package; fetching via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ---------------------------------------------------------------- module
pybind11_add_module(hnng MODULE bindings/hnng_py.cpp)

target_include_directories(hnng PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Release optimization flags (the core is header-only; all in this TU).
if(MSVC)
  target_compile_options(hnng PRIVATE /O2)
else()
  target_compile_options(hnng PRIVATE -O3 -funroll-loops)
  if(HNNG_NATIVE)
    target_compile_options(hnng PRIVATE -march=native)
  endif()
endif()

# Optional: a tiny C++ smoke executable (no Python) for local sanity once a
# compiler exists. Enable with -DHNNG_BUILD_SMOKE=ON.
option(HNNG_BUILD_SMOKE "Build the standalone C++ smoke test" OFF)
if(HNNG_BUILD_SMOKE)
  add_executable(hnng_smoke tests/smoke_main.cpp)
  target_include_directories(hnng_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(hnng_smoke PROPERTIES CXX_STANDARD 17)
endif()
