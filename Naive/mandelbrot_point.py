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

# Testing the Mandelbrot Point function
# Known points are extracted from image on Wikipedia: 
# https://upload.wikimedia.org/wikipedia/commons/d/dc/Mandelbrot_set%2C_plotted_with_Matplotlib.svg
# The mandelbrot set is visualized using in pyplot, with the colormap parameter set to prism. 
# The prism colormap, has the property of the rainbow pattern that consist of only 10 colors, repeatedly.

max_iterations = 100
bound = 2
power = 2

known_points = [
    {'x':0, 'y':0, 'expected':0},
    {'x':-1.95, 'y':-0.6, 'expected':1},
    {'x':-1.85, 'y':-0.6, 'expected':2},
    {'x':-1.6, 'y':-0.5, 'expected':3},
    {'x':-1.0, 'y':-0.65, 'expected':4},
    {'x':-1.0, 'y':-0.5, 'expected':5},
    {'x':-0.85, 'y':-0.475, 'expected':6},
    {'x':-0.85, 'y':-0.4, 'expected':7},
    {'x':-0.775, 'y':-0.375, 'expected':8},
    {'x':-0.8, 'y':-0.325, 'expected':9}
]

print(f'{max_iterations=} | {power=}')
for point in known_points:
    x = point['x']
    y = point['y']
    expected = point['expected']

    result = mandelbrot_point(x, y, max_iterations, bound, power)

    print(f"{x=}, {y=} | {expected=} <-> {result=}")