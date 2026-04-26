import numpy as np
import time
from numba import njit
from matplotlib import pyplot as plt

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

# parameters
max_iterations = 100
bound = 2
power = 2
res = 1024

# regions
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5
x_region = np.linspace(x_min, x_max, res)
y_region = np.linspace(y_min, y_max, res)

if __name__ == '__main__':
    # test time of computation
    warm_up = compute_numba_naive_mandelbrot(x_region, y_region, 1, bound, power, res)
    start_time = time.perf_counter()
    mandelbrot_array = compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res)
    test_time = time.perf_counter() - start_time
    print(f'Computation took {test_time:.5f} seconds!')

    # plot mandelbrot
    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap='twilight_shifted')
    plt.colorbar(graph)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title('Mandelbrot set for $z_n$ = $z^2$ + c')
    plt.show()
