# Naive Python Implementation
import numpy as np
from matplotlib import pyplot as plt
import time

def mandelbrot_point(x, y, max_iterations, bound, power):
    c = complex(x, y)
    z = 0
    result = None

    for iteration in range(max_iterations):
        if(abs(z) >= bound):
            result = iteration
            break
        else:
            z = z**power + c
    else: # this is only called if the for loop never 'breaks'
        result = 0

    return result

def compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power):
    mandelbrot_array = []

    for y_value in y_region:
        row = []
        for x_value in x_region:
            point_iteration = mandelbrot_point(x_value, y_value, max_iterations, bound, power)
            row.append(point_iteration)
        mandelbrot_array.append(row)

    return mandelbrot_array

# regions
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

x_res, y_res = 1024, 1024

x_region = np.linspace(x_min, x_max, x_res)
y_region = np.linspace(y_min, y_max, y_res)

max_iterations = 100
bound = 2
power = 2

# test time of computation
start_time = time.time()
mandelbrot_array = compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power)
test_time = time.time() - start_time
print(f'Computation took {test_time:.3f} seconds!')

ax = plt.axes()
ax.set_aspect('equal')
graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap = 'prism')
plt.colorbar(graph)
plt.xlabel("Real-Axis")
plt.ylabel("Imaginary-Axis")
plt.title('Multibrot set for $z_n$ = $z^2$ + c')
plt.show()
