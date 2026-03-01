import numpy as np
from matplotlib import pyplot as plt
import time
import os
import line_profiler

os.environ['LINE_PROFILE'] = '1' # makes sure that files are actually saved

# run this function using: 'kernprof -l -v mandelbrot_profiler.py' in the terminal
@line_profiler.profile
def compute_naive_mandelbrot(x_region, y_region, max_iterations, bound, power):
    mandelbrot_array = []

    for y_value in y_region:
        row = []

        for x_value in x_region:
            c = complex(x_value, y_value)
            z = 0

            for iteration in range(max_iterations):
                if(abs(z) >= bound):
                    row.append(iteration)
                    break
                else: 
                    z = z**power + c
            else: # this is only called if the for loop never 'breaks'
                row.append(max_iterations)
        mandelbrot_array.append(row)

    return mandelbrot_array

@line_profiler.profile
def compute_numpy_mandelbrot(x_mesh, y_mesh, max_iterations, bound, power):
    complex_number = 1j
    C = x_mesh + y_mesh * complex_number
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)

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
x_mesh, y_mesh = np.meshgrid(x_region, y_region)

max_iterations = 100
bound = 2
power = 2

# run the functions to get the line profiles
naive_mandelbrot = compute_naive_mandelbrot(x_region, y_region, max_iterations, bound, power)
numpy_mandelbrot = compute_numpy_mandelbrot(x_mesh, y_mesh, max_iterations, bound, power)
