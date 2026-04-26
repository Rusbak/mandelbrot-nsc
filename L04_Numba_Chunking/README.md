# L04 - Numba Parallel (Chunking)
L04 introduces core parallel‑performance reasoning. It explains why adding cores does not yield proportional speedup through Amdahl’s Law, Gustafson’s Law, and the Work‑Span model (Brent’s theorem). Serial fractions limit scalability and how critical‑path length bounds achievable speedup. The topic then moves to Python multiprocessing, showing how to bypass the GIL using Pool.map() and dynamic scheduling. Monte Carlo Pi and Mandelbrot are parallelised by splitting work into independent chunks, measuring speedup across core counts, and relating observed scaling back to Amdahl’s implied serial fraction.

## Exercises
### Exercise 1
- [x] Implement Monte Carlo Pi Estimation

### Exercise 2
- [x] Implement Monte Carlo Pi Estimation in Parallel using ´Pool.map´

### Exercise 3
- [x] Analyze Differences in Serial and Parallel Implementations
- [x] Backsolve Implied Serial Fraction
---

## Milestones
### Milestone 1
- [x] Refactor Numba to Chunk-Based Numba (Three Functions)
- [x] Compare to Numba from Last Week

### Milestone 2
- [x] Implement ´Pool.map´ Wrapper
- [x] Verify Results Visually

### Milestone 3
- [x] Perform Benchmarks
    - [x] Sweep of Processes
    - [x] Speedup
    - [x] Efficiency
---

## Results
This Implementation Compared to L03's Implementation, and L01's Implementation:

|Implementation|Compute Time|Speed-Up|
|:--|--:|--:|
|Naive (Baseline)          |3.56867 s|  1.000x|
|Numba Serial (Last Week)  |0.04954 s| 72.036x|
|Numba Parallel (This Week)|0.02190 s|162.953x|
