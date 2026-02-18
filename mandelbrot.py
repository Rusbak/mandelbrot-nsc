# Naive Python Implementation

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

x = 0
y = 0
max_iterations = 100
bound = 2
power = 2

point = mandelbrot_point(x, y, max_iterations, bound, power)

print(f'{point=}')