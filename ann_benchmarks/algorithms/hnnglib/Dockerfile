FROM ann-benchmarks

RUN pip install pybind11 numpy setuptools

# Vendored hnnglib C++ source (top-level d:/hnng/hnnglib)
COPY ann_benchmarks/algorithms/hnnglib/src /tmp/hnnglib_src
RUN cd /tmp/hnnglib_src && pip install .

RUN python -c 'import hnnglib; print("hnnglib import OK")'
