import numpy as np
import time, statistics
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit

@njit
def compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res):
    result = np.zeros((res,res), dtype=np.int32)

    for i in range(res):
        for j in range(res):
            c = x_region[j] + y_region[i] * 1j
            z = 0j
            n = 0

            while n < max_iterations and z.real*z.real + z.imag*z.imag <= bound*bound:
                z = z*z + c
                n += 1
            result[i, j] = n

    return result

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iterations, bound):
    z_real = 0.0
    z_imag = 0.0
    bound_sq = bound * bound
    
    for iteration in range(max_iterations):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag

        if z_real_sq + z_imag_sq > bound_sq:
            return iteration
        else:
            z_imag = 2.0 * z_real * z_imag + c_imag
            z_real = z_real_sq - z_imag_sq + c_real

    return max_iterations

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound):
    chunk_output = np.empty((row_end - row_start, res), dtype=np.int32)
    
    dx = (x_max - x_min) / res
    dy = (y_max - y_min) / res

    for row in range(row_end - row_start):
        c_imag = y_min + (row + row_start) * dy

        for col in range(res):
            c_real = x_min + col * dx
            chunk_output[row, col] = mandelbrot_pixel(c_real, c_imag, max_iterations, bound)

    return chunk_output

def mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, n_chunks, bound):
    chunk_size = max(1, res // n_chunks)
    tasks = []
    row = 0

    while row < res:
        row_end = min(row + chunk_size, res)
        task = delayed(mandelbrot_chunk)(row, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound)
        tasks.append(task)
        row = row_end

    parts = dask.compute(*tasks)
    mandelbrot_stack = np.vstack(parts)

    return mandelbrot_stack

res = 1024
max_iterations = 100
bound = 2
power = 2

x_min, x_max = -2.5, 1.0
y_min, y_max = -1.5, 1.5
x_region = np.linspace(x_min, x_max, res)
y_region = np.linspace(y_min, y_max, res)

if __name__ == '__main__':
    print(f"Numba Barebones:")
    
    barebones_times = []
    for _ in range(5): # 5 tests
        warm_up = compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res)
        barebones_start = time.perf_counter()
        result = compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res)
        barebones_time = time.perf_counter() - barebones_start
        barebones_times.append(barebones_time)

    barebones_median = statistics.median(barebones_times)
    print(f'{barebones_median:.5f}s\n')

    # Milestone 1
    num_workers = 12 # found from previous lecture
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    client = Client(cluster)
    client.run(lambda:mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10, bound))
    
    num_chunks = num_workers # one chunk per worker
    print(f"Dask Local Baseline ({num_workers} Workers | {num_chunks} Chunks):")
    
    baseline_times = []
    for _ in range(5): # 5 tests
        warm_up = mandelbrot_dask(128, x_min, x_max, y_min, y_max, 10, num_workers, bound)
        baseline_start = time.perf_counter()
        result = mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, num_chunks, bound)
        baseline_time = time.perf_counter() - baseline_start
        baseline_times.append(baseline_time)

    baseline_median = statistics.median(baseline_times)
    print(f'{baseline_median:.5f}s')
    barebones_speedup = barebones_median / baseline_median
    print(f'Speed Up from Barebones: {barebones_speedup:.3f}\n')

    # Milestone 2
    # Open Cluster before Benchmarking  
    client.run(lambda:mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10, bound))
    
    chunk_multiples = [1, 2, 4, 8, 16, 32]
    wall_times = []
    lifs = []

    print(f"Dask Local Chunk Sweeping ({num_workers} Workers):")
    print(f"{'Total Chunks':>12} | {'Compute Time':>12} | {'x1':>8} | {'Speed Up':>8} | {'LIF':>6}")

    for chunk_multiple in chunk_multiples:
        num_chunks = chunk_multiple * num_workers

        chunk_times = []
        for _ in range(5): # 5 tests
            warm_up = mandelbrot_dask(128, x_min, x_max, y_min, y_max, 10, num_workers, bound)
            chunk_start = time.perf_counter()
            result = mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, num_chunks, bound)
            chunk_time = time.perf_counter() - chunk_start
            chunk_times.append(chunk_time)

        chunk_median = statistics.median(chunk_times)
        numba_speedup = barebones_median / chunk_median
        speed_up = baseline_median / chunk_median
        lif = num_workers * chunk_median / baseline_median - 1

        print(f'{num_chunks:12d} | {chunk_median:11.5f}s | {numba_speedup:7.3f}x |{speed_up:7.3f}x | {lif:6.2f}')
        
        wall_times.append(chunk_median)
        lifs.append(lif)

    # fastest run
    min_wall_time = min(wall_times)
    min_wall_idx = wall_times.index(min_wall_time)
    min_wall_chunks = chunk_multiples[min_wall_idx] * num_workers
    print(f'\nFastest Run: {min_wall_chunks} Chunks -> {min_wall_time:.5f}s')
    
    # lowest LIF
    min_lif = min(lifs)
    min_lif_idx = lifs.index(min_lif)
    min_lif_chunks = chunk_multiples[min_lif_idx] * num_workers
    print(f'Lowest LIF: {min_lif_chunks} Chunks -> {min_lif:.3f}')

    client.close()
    cluster.close()

'''Copy + paste from terminal window
Numba Barebones: 
0.04358s

Dask Local Baseline (12 Workers | 12 Chunks):
0.09779s
Speed Up from Barebones: 0.446x

Dask Local Chunk Sweeping (12 Workers):
Total Chunks | Compute Time |       x1 | Speed Up |    LIF
          12 |     0.09476s |   0.460x |   1.032x |  10.63
          24 |     0.06986s |   0.624x |   1.400x |   7.57
          48 |     0.09321s |   0.468x |   1.049x |  10.44
          96 |     0.17286s |   0.252x |   0.566x |  20.21
         192 |     0.34085s |   0.128x |   0.287x |  40.83
         384 |     1.03624s |   0.042x |   0.094x | 126.16

Fastest Run: 24 Chunks -> 0.06986s
Lowest LIF: 24 Chunks -> 7.57
'''