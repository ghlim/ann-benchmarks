FROM ann-benchmarks

RUN pip install pybind11 numpy setuptools

# Vendored hnng C++ source (branch Code20260528, d:/hnng/hnng)
COPY ann_benchmarks/algorithms/hnng/src /tmp/hnng_src
RUN cd /tmp/hnng_src/python_bindings && python setup.py install

RUN python -c 'import hnnglib; print("hnng import OK")'
