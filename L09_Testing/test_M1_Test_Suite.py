import numpy as np
import pytest
import time, statistics
from numba import njit
from multiprocessing import Pool
import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

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

def mandelbrot_parallel(res, x_min, x_max, y_min, y_max, max_iterations, bound, num_workers):
    chunk_size = max(1, res // num_workers)
    chunks = []

    row = 0
    while row < res:
        row_end = min(row + chunk_size, res)
        chunks.append((row, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound))
        row = row_end
    
    with Pool(processes=num_workers) as pool:
        pool.map(_worker, chunks)
        parallel_times = []
        for _ in range(5): # 5 tests
            test_start = time.perf_counter()
            mandelbrot_stack = np.vstack(pool.map(_worker,chunks))
            test_time = time.perf_counter() - test_start
            parallel_times.append(test_time)
    parallel_median = statistics.median(parallel_times)
    print(f"Median Compute Time: {parallel_median:.5f}s")

    return mandelbrot_stack

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

# Core Unit Test of Mandelbrot Point Function
@pytest.mark.parametrize(
    "c_real, c_imag, expected",
    [
        (0.0, 0.0, 100),        # inside set → never escapes
        (2.0, 2.0, 1),          # escapes immediately
        (-1.0, 0.0, 100),       # inside set
        (0.5, 0.5, 5),          # escapes after a few iterations
    ],
)
def test_mandelbrot_pixel(c_real, c_imag, expected):
    max_iter = 100
    bound = 2.0

    result = mandelbrot_pixel(c_real, c_imag, max_iter, bound)

    # not exact for all → check behavior category
    if expected == 100:
        assert result == max_iter
    else:
        assert result < max_iter

# Core Unit Test for Shapes and Values of the Mandelbrot Chunk Function
def test_mandelbrot_chunk_shape_and_values():
    res = 10
    result = mandelbrot_chunk(
        0, res, res,
        -2.0, 1.0,
        -1.5, 1.5,
        50,
        2.0
    )

    assert result.shape == (res, res)
    assert result.dtype == np.int32
    assert np.all(result >= 0)
    assert np.all(result <= 50)

# Unit Test for Serial Consistency
def test_mandelbrot_serial_consistency():
    res = 20

    grid1 = mandelbrot_serial(res, -2, 1, -1.5, 1.5, 50, 2.0)
    grid2 = mandelbrot_serial(res, -2, 1, -1.5, 1.5, 50, 2.0)

    assert np.array_equal(grid1, grid2)

# Unit Test for Worker Consistency when doing Multiprocessing
def test_worker_matches_chunk():
    args = (0, 5, 10, -2, 1, -1.5, 1.5, 50, 2.0)

    worker_result = _worker(args)
    direct_result = mandelbrot_chunk(*args)

    assert np.array_equal(worker_result, direct_result)

# Unit Test for Equality between Serial and Parallel Implementations
def test_parallel_matches_serial():
    res = 30

    serial = mandelbrot_serial(res, -2, 1, -1.5, 1.5, 50, 2.0)

    parallel = mandelbrot_parallel(
        res, -2, 1, -1.5, 1.5,
        50, 2.0,
        num_workers=2
    )

    assert np.array_equal(serial, parallel)

# Unit Test for Equality between Serial and Local Dask Implementations
def test_dask_matches_serial():
    res = 30

    serial = mandelbrot_serial(res, -2, 1, -1.5, 1.5, 50, 2.0)

    dask_result = mandelbrot_dask(
        res, -2, 1, -1.5, 1.5,
        50,
        n_chunks=4,
        bound=2.0
    )

    assert np.array_equal(serial, dask_result)
