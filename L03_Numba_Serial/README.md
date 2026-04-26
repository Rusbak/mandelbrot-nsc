# L03 - Numba (Naive + @njit)
L03 introduces algorithmic intensity to distinguish compute‑ vs memory‑bound workloads. Profiling tools (cProfile, line profiler) reveal real bottlenecks in naive code. Numba JIT is applied to remove interpreter overhead and generate native machine code, achieving speedups of 72.036x. The topic also explores float32 vs float64 tradeoffs and mentions a systematic workflow: measure, optimize, verify. Mandelbrot’s high intensity makes it ideal for JIT acceleration.

## Milestones
### Milestone 1
- [x] Use cProfile on Both Versions (Naive & Numpy)
- [x] Inspect cProfile Output

### Milestone 2
- [x] Add @profile Decorator and Run line_profiler
- [x] Compare cProfile vs. line_profiler

### Milestone 3
- [x] Implement & Compare Numba Approaches (Hybrid & Naive)
- [x] Benchmark All Versions

### Milestone 4
- [x] Test Different Precisions (float 16/32/64/128) (16 + 128 are not supported...)
- [x] Visualize Comparison
- [x] Inspect Comparison
---

## Results
This Implementation Compared to L02's Implementation, and L01's Implementation:

|Implementation|Compute Time|Speed-Up|
|:--|--:|--:|
|Naive (Baseline) |3.56867 s| 1.000x|
|Numpy (Last Week)|1.44573 s| 2.468x|
|Numba (This Week)|0.04954 s|72.036x|
