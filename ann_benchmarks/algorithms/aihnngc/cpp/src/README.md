# src/ — intentionally empty (the core is header-only)

The entire hNNG core lives in `include/hnng/` as header-only code so the hot
distance loop stays in a single translation unit (`bindings/hnng_py.cpp`) where
the compiler can inline it. There are no `.cpp` implementation files to build
here.

If a future change wants out-of-line definitions (e.g. to cut compile time or
expose a non-template C ABI), add them here and list them in BOTH:
  * `CMakeLists.txt`  -> `pybind11_add_module(hnng MODULE bindings/hnng_py.cpp src/<new>.cpp)`
  * `setup.py`        -> the `sources=[...]` list of the Pybind11Extension.
