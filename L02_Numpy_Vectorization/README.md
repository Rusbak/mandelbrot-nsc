# L02 - Numpy Vectorization
L02 explains why memory, not CPU speed, often limits performance. It introduces latency, bandwidth, cache hierarchy, cache lines, spatial locality, and memory layout. It explains how access patterns affect effective bandwidth and why contiguous arrays matter. SIMD and multicore concepts show how modern CPUs exploit parallelism. The topic concludes with NumPy vectorization of Mandelbrot, showing a big speedup (2.468x) by reducing Python overhead and leveraging cache‑friendly, compiled operations.

## Milestones
### Milestone 1
- [x] Create 1D Arrays with ´np.linspace´
- [x] Create 2D Grid with ´np.meshgrid´
- [x] Combine into a Complex Array
- [x] Verify Shape and dType

### Milestone 2
- [x] Eliminate Loops 1 & 2
- [x] Measure Compute Time

### Milestone 3
- [x] Create a Large Square Array
- [x] Write a Function that computes Row Sums
- [x] Write a Function that computes Column Sums
- [x] Inspect Differences
- [x] Do the same for a Fortran Array

### Milestone 4
- [x] Run Vectorized Mandelbrot Implementation for Different Grid Sizes (256, 512, 1024, 2048, 4096)
- [x] Compare Compute Times
---

## Results
This Implementation Compared to L01's Implementation:

|Implementation|Compute Time|Speed-Up|
|:--|--:|--:|
|Naive (Baseline) |3.56867 s|1.000x|
|Numpy (This Week)|1.44573 s|2.468x|
