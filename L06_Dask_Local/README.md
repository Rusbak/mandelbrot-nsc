# L06 - Dask (Local)
L06 introduces distributed computing and how moving beyond shared memory changes everything: separate RAM, network latency, and scheduler overhead. The α + βn model reappears at network scale, where Ethernet latency is hundreds of times higher than RAM. Dask’s lazy evaluation is introduced via dask.delayed, building task graphs and executing them with dask.compute(). A LocalCluster runs multiple worker processes, visualised in the Dask dashboard. Mandelbrot is ported to Dask, chunk sizes are swept to find the overhead‑balanced region, and a full benchmark table is collected for multiprocessing vs. Dask local.

## Exercises
### Exercise 1
- [x] Visualize Dask Graph
- [x] Sweep Chunks

### Exercise 2
- [x] View Task Execution Live in Dashboard
- [x] Try Different Amounts of Workers

### Exercise 3
- [x] Two-pass Normalization for Dependent Task Graphs
---

## Milestones
### Milestone 1
- [x] Wrap existing Numba Mandlebrot Algorithm with ´dask.delayed´
- [x] Verify Output
- [x] Record Median

### Milestone 2
- [x] Sweep Amount of Chunks, guided by the 'Three-way Trade-off'
- [x] Print: Chunks, Time, Speedup, LIF
- [x] Add Early Exit to Dask Implementation

### Milestone 3
- [x] Add Dask Implementation to Performance Notebook
---

## Results
This Implementation Compared to L05's Implementation, and L01's Implementation:

|Implementation|Compute Time|Speed-Up|
|:--|--:|--:|
|Naive (Baseline)          |3.56867 s|  1.000x|
|Numba Parallel (Last Week)|0.01294 s|275.786x|
|Dask Local (This Week)    |0.06986 s| 51.083x|
