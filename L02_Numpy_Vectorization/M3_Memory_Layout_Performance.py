import numpy as np
from matplotlib import pyplot as plt
import time

size = 10000
numpy_square = np.random.rand(size, size)
fortran_square = np.asfortranarray(numpy_square)

squares = [
    ("numpy_square", numpy_square),
    ("fortran_square", fortran_square)
]

def compute_row_sums(size, square):
    total_sum = 0
    for i in range(size):
        row_sum = np.sum(square[i, :])
        total_sum += row_sum
    return total_sum

def compute_col_sums(size, square):
    total_sum = 0
    for j in range(size):
        col_sum = np.sum(square[:, j])
        total_sum += col_sum
    return total_sum

test_functions = [compute_row_sums, compute_col_sums]

def test_layout_performance(squares, size, test_functions):
    for square_type, square in squares:
        print(f'Type of Square: {square_type}')

        for test_function in test_functions:
            print(f'Used Function: {test_function.__name__}')

            start_time = time.perf_counter()
            total_sum = test_function(size, square)
            test_time = time.perf_counter() - start_time
            print(f'Compute Time: {test_time:.4f}\n')

test_layout_performance(squares, size, test_functions)

'''
It is expected that the difference in looping through rows vs cols will give 
an increase of ~8x, due to rows looping through all 8 entries in a cacheline, 
while cols loops through one entry per cacheline. 
It is then extected that the opposite can be said about a fortran style array, 
due to its structure of being column-major.

Type of Square: numpy_square
Used Function: compute_row_sums
Compute Time: 0.171

Used Function: compute_col_sums
Compute Time: 1.277

Type of Square: fortran_square
Used Function: compute_row_sums
Compute Time: 1.183

Used Function: compute_col_sums
Compute Time: 0.148
'''