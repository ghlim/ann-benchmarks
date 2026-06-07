# Dockerfile for the AIhNNGc algorithm (exact hierarchical NNG, C++ core).
# Installed under the name `aihnngc` so it coexists with the other local
# `hnng`/`hnnglib` implementations for side-by-side benchmarking.
FROM ann-benchmarks

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy setuptools wheel

# Compiled C++/pybind11 core (module name `hnng`), built with -march=native SIMD.
COPY ann_benchmarks/algorithms/aihnngc/cpp /tmp/hnng_cpp
ENV HNNG_NATIVE=1
RUN pip install --no-cache-dir /tmp/hnng_cpp

RUN python -c "import hnng; print('AIhNNGc (C++) import OK', hnng.Index('l2'))"
