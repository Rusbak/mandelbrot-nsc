import numpy as np
from matplotlib import pyplot as plt
import time
from numba import njit
from multiprocessing import Pool
import os
import statistics

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

def mandelbrot_serial(res, x_min, x_max, y_min, y_max, max_iterations, bound):
    mandelbrot_grid = mandelbrot_chunk(0, res, res, x_min, x_max, y_min, y_max, max_iterations, bound)

    return mandelbrot_grid

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(res, x_min, x_max, y_min, y_max, max_iterations, bound, num_workers, num_chunks, pool):
    chunk_size = max(1, res // num_chunks)
    chunks = []

    row = 0
    while row < res:
        row_end = min(row + chunk_size, res)
        chunks.append((row, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound))
        row = row_end
    
    if pool:
        parts = pool.map(_worker, chunks)
        mandelbrot_stack = np.vstack(parts)

        return mandelbrot_stack

    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iterations, bound)]

    with Pool(processes=num_workers) as pool: # remove benchmark from here
        parallel_times = []
        for _ in range(5): # 5 tests
            warm_up = pool.map(_worker, tiny)

            test_start = time.perf_counter()
            parts = pool.map(_worker, chunks)
            mandelbrot_stack = np.vstack(parts)
            test_time = time.perf_counter() - test_start

            parallel_times.append(test_time)

    parallel_median = statistics.median(parallel_times)
    print(f"Median Compute Time: {parallel_median:.5f}s")

    return mandelbrot_stack

def benchmark(func, tests, *args):
    times = []

    for test in range(tests):
        warm_up = func(*args)
        start_time = time.perf_counter()
        mandelbrot = func(*args)
        test_time = time.perf_counter() - start_time
        times.append(test_time)

    median = statistics.median(times)
    print(f"Median Compute Time: {median:.5f}s")

    return mandelbrot, median

# parameters
max_iterations = 100
bound = 2.0
power = 2
res = 1024

# regions
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5
x_region = np.linspace(x_min, x_max, res)
y_region = np.linspace(y_min, y_max, res)

if __name__ == '__main__':
    # Milestone 1
    print('Mandelbrot Serial:')
    mandelbrot_array, serial_median = benchmark(mandelbrot_serial, 5, res, x_min, x_max, y_min, y_max, max_iterations, bound)
    print()

    # ax = plt.axes()
    # ax.set_aspect('equal')
    # graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap='twilight_shifted')
    # plt.colorbar(graph)
    # plt.xlabel("Real")
    # plt.ylabel("Imaginary")
    # plt.title('Mandelbrot set for $z_n$ = $z^2$ + c')
    # plt.show()

    # Milestone 2
    num_workers = 12 # best amount found from L4
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iterations, bound)]
    chunks_per_worker = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f'Mandelbrot Parallel ({num_workers} workers):')
    print(f"{'Chunks':>6} | {'Compute Time':>12} | {'Speed Up':>8} | {'LIF':>4}")
    for mult in chunks_per_worker:
        num_chunks = num_workers * mult

        with Pool(processes=num_workers) as pool:
            warm_up = pool.map(_worker, tiny)

            test_times = []
            for test in range(5):
                test_start = time.perf_counter()
                mandelbrot_array = mandelbrot_parallel(res, x_min, x_max, y_min, y_max, max_iterations, bound, num_workers, num_chunks, pool)
                test_time = time.perf_counter() - test_start
                test_times.append(test_time)

        parallel_median = statistics.median(test_times)
        lif = num_workers * parallel_median / serial_median - 1
        speed_up = serial_median / parallel_median
        print(f'{num_chunks:6d} | {parallel_median:11.5f}s | {speed_up:7.3f}x | {lif:3.2f}')

    '''Copy + paste from terminal window:
    Chunks | Compute Time | Speed Up |  LIF
        12 |     0.01859s |   2.504x | 3.79
        24 |     0.01401s |   3.321x | 2.61
        48 |     0.01294s |   3.597x | 2.34
        96 |     0.02147s |   2.168x | 4.54
       192 |     0.01873s |   2.485x | 3.83
       384 |     0.02098s |   2.219x | 4.41
       768 |     0.01907s |   2.440x | 3.92
      1536 |     0.02073s |   2.245x | 4.34
    '''
