import numpy as np
from matplotlib import pyplot as plt
import time

def compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power):
    complex_number = 1j
    C = x_region + y_region * complex_number
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for _ in range(max_iterations):
        mask = np.abs(Z) <= bound
        Z[mask] = Z[mask]**power + C[mask]
        M[mask] += 1

    return M

max_iterations = 100
bound = 2
power = 2

# regions
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

scales = [256, 512, 1024, 2048, 4096]

for scale in scales:
    print(f'Scale: {str(scale)}')

    x_res, y_res = scale, scale

    x_values = np.linspace(x_min, x_max, x_res)
    y_values = np.linspace(y_min, y_max, y_res)
    x_region, y_region = np.meshgrid(x_values, y_values)

    # test time of computation
    start_time = time.perf_counter()
    mandelbrot_array = compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power)
    test_time = time.perf_counter() - start_time
    print(f'Computation took {test_time:.3f} seconds!\n')

'''
When scaling the dimension by a factor of 2, we are increasing the total number 
of pixel by a factor of 4, so we should expect an increase in compute time of ~4x 
(+/- due to some pixel shifting between number of iterations at each scale)

Scale: 256
Computation took 0.063 seconds!

Scale: 512
Computation took 0.338 seconds!

Scale: 1024
Computation took 1.520 seconds!

Scale: 2048
Computation took 5.676 seconds!

Scale: 4096
Computation took 25.630 seconds!
'''