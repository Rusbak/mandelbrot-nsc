import random, time, os
from functools import reduce
from multiprocessing import Pool

samples = 1000000
data = [random.randint(10, 100) for sample in range(samples)]

def subtract_seven(x):
    return x - 7

if __name__ == '__main__':
    # serial
    serial_start = time.perf_counter()
    result_serial = reduce(lambda a, b: a + b,
                        filter(lambda x: x % 2 == 1,
                            map(subtract_seven, data)))
    serial_time = time.perf_counter() - serial_start
    print(f'Serial: {serial_time:.5f}s | Result: {result_serial}')

    # parallel
    with Pool(processes=os.cpu_count() // 2) as pool:
        warm_up = pool.map(subtract_seven, data)
        para_start = time.perf_counter()
        mapped = pool.map(subtract_seven, data)
    result_para = reduce(lambda a, b: a + b,
                    filter(lambda x: x % 2 == 1,
                        mapped))
    para_time = time.perf_counter() - para_start
    print(f'Parallel: {para_time:.5f}s | Result: {result_para}')


    speedup = serial_time / para_time
    print(f'Speedup: {(speedup):.3f}x')

'''Copy+paste from terminal window

Serial: 0.09715s | Result: 24212027
Parallel: 0.12318s | Result: 24212027
Speedup: 0.789x
'''