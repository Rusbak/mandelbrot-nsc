# Naive Python Implementation

def mandelbrot_point(x, y, max_iterations, bound):
    complex_number = 1j
    c = x + y * complex_number
    z = 0
    result = None

    for iteration in range(max_iterations):
        if(abs(z) >= bound):
            result = iteration
            break
        else:
            z = z**2 + c
    else: # this is only called if the for loop never 'breaks'
        result = max_iterations

    return result

x = 0
y = 0
max_iterations = 100
bound = 2

point = mandelbrot_point(x, y, max_iterations, bound)

print(f'{point=}')