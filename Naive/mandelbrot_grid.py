import numpy as np
from matplotlib import pyplot as plt
import time

def compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power):
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

# regions
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

res = 1024

x_region = np.linspace(x_min, x_max, res)
y_region = np.linspace(y_min, y_max, res)

max_iterations = 100
bound = 2
power = 2

if __name__ == '__main__':
    # test time of computation
    warm_up = compute_mandelbrot_grid(x_region, y_region, 1, bound, power)
    start_time = time.perf_counter()
    mandelbrot_array = compute_mandelbrot_grid(x_region, y_region, max_iterations, bound, power)
    test_time = time.perf_counter() - start_time
    print(f'Computation took {test_time:.4f} seconds!')

    # plot mandelbrot
    ax = plt.axes()
    ax.set_aspect('equal')
    graph = ax.pcolormesh(x_region, y_region, mandelbrot_array, cmap = 'twilight_shifted')
    plt.colorbar(graph)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title('Mandelbrot set for $z_n$ = $z^2$ + c')
    plt.show()
