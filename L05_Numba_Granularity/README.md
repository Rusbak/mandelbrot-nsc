# L05 - Numba Parallel (Granularity)
L05 focuses on granularity and load balancing. Equal‑sized chunks do not imply equal work for Mandelbrot because iteration counts vary heavily across rows. The topic goes over Graham’s Bound, the Load Imbalance Factor, and how over‑decomposition reduces the longest‑task penalty. The topic describes static vs. dynamic scheduling, showing why Pool.map() benefits from many moderately sized tasks. The map‑filter‑reduce pattern is introduced. Lastly n_chunks is tuned to find the hardware‑specific sweet spot and produces a complete multiprocessing speedup table and Amdahl fit.

## Exercises
### Exercise 1
- [x] Compare Chunk Sizes for Monte Carli Pi Estimation

### Exercise 2
- [x] Compare Serial and Parallel(´Pool.map()´) for a Simple Task
---

## Milestones
### Milestone 1
- [x] Add ´(cache=True)´ to Decorator
- [x] Add ´nun_chunks´ Parameter to Mandelbrot Function
- [x] Pass Open Pools as Parameter in Mandelbrot Function

### Milestone 2
- [x] Find the Best Amount of Chunks per Worker

### Milestone 3
- [x] Perform Benchmark
    - [x] Sweep of Chunks
    - [x] Speedup
    - [x] LIF
---

## Results
This Implementation Compared to L03 & L04's Implementations, and L01's Implementation:

|Implementation|Compute Time|Speed-Up|
|:--|--:|--:|
|Naive (Baseline)          |3.56867 s|  1.000x|
|Numba Serial (Week Before)|0.04954 s| 72.036x|
|Numba Parallel (Last Week)|0.02190 s|162.953x|
|Numba Parallel (This Week)|0.01294 s|275.786x|
