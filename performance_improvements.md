# Performance Improvements since 0.5

(temp file will delete)

**NOTES:**

1. CPy refers to CPython 3.14.0, PyPy refers to PyPy (version 7.3.20) 3.11.13**
2. All values are in seconds.
3. All values are taken on an Intel Core i7-14700KF with 32GB RAM on Arch Linux (CachyOS kernel), exact results may vary

| **Benchmark**                 | **0.5.3 (CPy)** | **0.6.0 (CPy)** | **Gains (CPy)** | **0.5.3 (PyPy)** | **0.6.0 (PyPy)** | **Gains (PyPy)** |
|-------------------------------|-----------------|-----------------|-----------------|------------------|------------------|------------------|
| BsortTorture 500 nums         | 4.051           | 2.344           | 1.73x           | 1.166            | 0.698            | 1.67x            |
| QsortTorture 1000 nums        | 3.378           | 3.25            | 1.04x           | 1.434            | 1.283            | 1.12x            |
| PrimeTorture 30000 max        | 2.429           | 1.558           | 1.56x           | 0.528            | 0.382            | 1.38x            |
| raylib_random_rects 400 rects | 3.463           | 1.981           | 1.75x           | 1.406            | 0.737            | 1.91x            |
