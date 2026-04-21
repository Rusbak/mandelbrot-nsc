# Copy of code from L04
import numpy as np
from matplotlib import pyplot as plt
import time
from numba import njit
from multiprocessing import Pool
import os
import statistics
from typing import Tuple, List

@njit
def mandelbrot_pixel(
    c_real: float,
    c_imag: float,
    max_iterations: int,
    bound: float
) -> int:
    '''
    Compute escape iteration count for a single complex point in the Mandelbrot set.

    Parameters
    ----------
    c_real : float
        Real part of the complex number \( c \).
    c_imag : float
        Imaginary part of the complex number \( c \).
    max_iterations : int
        Maximum number of iterations to test for divergence.
    bound : float
        Escape radius; iteration stops when \( |z_n| > \\text{bound} \).

    Returns
    -------
    int
        Iteration index at which divergence occurs, or `max_iterations`
        if the point does not escape.

    Examples
    --------
    >>> mandelbrot_pixel(0.0, 0.0, 100, 2.0)
    100
    >>> mandelbrot_pixel(2.0, 2.0, 100, 2.0)
    1
    '''
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

@njit
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    res: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iterations: int,
    bound: float
) -> np.ndarray:
    '''
    Compute a horizontal chunk of the Mandelbrot set grid.

    Parameters
    ----------
    row_start : int
        Starting row index (inclusive).
    row_end : int
        Ending row index (exclusive).
    res : int
        Resolution of the full grid (number of columns).
    x_min, x_max : float
        Range of real axis.
    y_min, y_max : float
        Range of imaginary axis.
    max_iterations : int
        Maximum iterations per pixel.
    bound : float
        Escape radius.

    Returns
    -------
    np.ndarray
        2D array of shape `(row_end - row_start, res)` containing iteration counts.

    Examples
    --------
    >>> mandelbrot_chunk(0, 10, 100, -2, 1, -1.5, 1.5, 50, 2.0).shape
    (10, 100)
    '''
    chunk_output = np.empty((row_end - row_start, res), dtype=np.int32)

    dx = (x_max - x_min) / res
    dy = (y_max - y_min) / res

    for row in range(row_end - row_start):
        c_imag = y_min + (row + row_start) * dy

        for col in range(res):
            c_real = x_min + col * dx
            chunk_output[row, col] = mandelbrot_pixel(
                c_real, c_imag, max_iterations, bound
            )

    return chunk_output

def mandelbrot_serial(
    res: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iterations: int,
    bound: float
) -> np.ndarray:
    '''
    Compute the full Mandelbrot grid using a single process.

    Parameters
    ----------
    res : int
        Grid resolution (number of rows and columns).
    x_min, x_max : float
        Real axis bounds.
    y_min, y_max : float
        Imaginary axis bounds.
    max_iterations : int
        Maximum iterations per pixel.
    bound : float
        Escape radius.

    Returns
    -------
    np.ndarray
        2D array of shape `(res, res)` with iteration counts.

    Examples
    --------
    >>> grid = mandelbrot_serial(100, -2, 1, -1.5, 1.5, 50, 2.0)
    >>> grid.shape
    (100, 100)
    '''
    return mandelbrot_chunk(
        0, res, res, x_min, x_max, y_min, y_max, max_iterations, bound
    )

def _worker(args: tuple) -> np.ndarray:
    '''
    Worker wrapper for multiprocessing execution.

    Parameters
    ----------
    args : tuple
        Arguments to pass to `mandelbrot_chunk`.

    Returns
    -------
    np.ndarray
        Computed chunk of the Mandelbrot grid.

    Examples
    --------
    >>> _worker((0, 5, 10, -2, 1, -1.5, 1.5, 20, 2.0)).shape
    (5, 10)
    '''
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    res: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iterations: int,
    bound: float,
    num_workers: int
) -> np.ndarray:
    '''
    Compute the Mandelbrot grid using multiprocessing.

    Parameters
    ----------
    res : int
        Grid resolution.
    x_min, x_max : float
        Real axis bounds.
    y_min, y_max : float
        Imaginary axis bounds.
    max_iterations : int
        Maximum iterations per pixel.
    bound : float
        Escape radius.
    num_workers : int
        Number of parallel worker processes.

    Returns
    -------
    np.ndarray
        2D array of shape `(res, res)` with iteration counts.

    Examples
    --------
    >>> mandelbrot_parallel(100, -2, 1, -1.5, 1.5, 50, 2.0, 2).shape
    (100, 100)
    '''
    chunk_size = max(1, res // num_workers)
    chunks: List[tuple] = []

    row = 0
    while row < res:
        row_end = min(row + chunk_size, res)
        chunks.append(
            (row, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound)
        )
        row = row_end

    with Pool(processes=num_workers) as pool:
        pool.map(_worker, chunks)
        parallel_times = []

        for _ in range(5):
            test_start = time.perf_counter()
            mandelbrot_stack = np.vstack(pool.map(_worker, chunks))
            test_time = time.perf_counter() - test_start
            parallel_times.append(test_time)

    parallel_median = statistics.median(parallel_times)
    print(f"Median Compute Time: {parallel_median:.5f}s")

    return mandelbrot_stack

def benchmark(
    func, 
    tests: int, 
    *args
) -> Tuple[np.ndarray, float]:
    '''
    Benchmark execution time of a Mandelbrot computation function.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    tests : int
        Number of repeated runs.
    *args
        Arguments passed to `func`.

    Returns
    -------
    tuple[np.ndarray, float]
        Output of the function and median execution time (seconds).

    Examples
    --------
    >>> grid, median = benchmark(mandelbrot_serial, 3, 100, -2, 1, -1.5, 1.5, 50, 2.0)
    >>> isinstance(median, float)
    True
    '''
    times = []

    for _ in range(tests):
        func(*args)  # warm-up
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

    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap='twilight_shifted')
    plt.colorbar(graph)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title('Mandelbrot set for $z_n$ = $z^2$ + c')
    plt.show()

    # Milestone 2
    num_workers = 4
    print(f'Mandelbrot Parallel ({num_workers}):')
    mandelbrot_array = mandelbrot_parallel(res, x_min, x_max, y_min, y_max, max_iterations, bound, num_workers)
    print()

    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap='twilight_shifted')
    plt.colorbar(graph)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title('Mandelbrot set for $z_n$ = $z^2$ + c')
    plt.show()

    # Milestone 3
    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, res // n_workers)
        chunks = []

        row = 0
        while row < res:
            end = min(row + chunk_size, res)
            chunks.append((row, end, res, x_min, x_max, y_min, y_max, max_iterations, bound))
            row = end

        with Pool(processes=n_workers) as pool:
            warm_up = pool.map(_worker, chunks)
            parallel_times = []

            for _ in range(5): # 5 tests
                test_start = time.perf_counter()
                np.vstack(pool.map(_worker, chunks)) # visualization already happened in Milestone 2 :)
                test_time = time.perf_counter() - test_start
                parallel_times.append(test_time)
        parallel_median = statistics.median(parallel_times)

        speedup = serial_median / parallel_median
        print(f"{n_workers:2d} workers: {parallel_median:.5f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")

        '''Copy + paste from terminal window:
        Mandelbrot Serial:
        Median Compute Time: 0.05315s

        Mandelbrot Parallel (4):
        Median Compute Time: 0.03935s

        1  workers: 0.05431s, speedup=0.98x, eff=98%
        2  workers: 0.03165s, speedup=1.68x, eff=84%
        3  workers: 0.04602s, speedup=1.15x, eff=38%
        4  workers: 0.03238s, speedup=1.64x, eff=41%
        5  workers: 0.03438s, speedup=1.55x, eff=31%
        6  workers: 0.02723s, speedup=1.95x, eff=33%
        7  workers: 0.02723s, speedup=1.95x, eff=28%
        8  workers: 0.02537s, speedup=2.10x, eff=26%
        9  workers: 0.02442s, speedup=2.18x, eff=24%
        10 workers: 0.02523s, speedup=2.11x, eff=21%
        11 workers: 0.02190s, speedup=2.43x, eff=22%
        12 workers: 0.02258s, speedup=2.35x, eff=20%
        '''
