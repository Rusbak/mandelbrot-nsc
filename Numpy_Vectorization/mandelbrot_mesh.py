import numpy as np
from matplotlib import pyplot as plt
import time

def compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power):
    complex_number = 1j
    C = x_region + y_region * complex_number
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for iteration in range(max_iterations):
        mask = np.abs(Z) <= bound
        Z[mask] = Z[mask]**power + C[mask]
        M[mask] += 1

    return M

# regions
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

x_res, y_res = 1024, 1024

x_region = np.linspace(x_min, x_max, x_res)
y_region = np.linspace(y_min, y_max, y_res)
x_region, y_region = np.meshgrid(x_region, y_region)

max_iterations = 100
bound = 2
power = 2

# test time of computation
start_time = time.perf_counter()
mandelbrot_array = compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power)
test_time = time.perf_counter() - start_time
print(f'Computation took {test_time:.3f} seconds!')

ax = plt.axes()
ax.set_aspect('equal')
graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap = 'twilight_shifted')
plt.colorbar(graph)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title('Mandelbrot set for $z_n$ = $z^2$ + c')
plt.show()
