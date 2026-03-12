from multiprocessing import Pool
import time

def square(x):
    time.sleep(0.1)
    result = x * x
    return result

if __name__=='__main__':
    numbers = list(range(100))
    
    start_serial = time.time()
    results_serial = [square(x) for x in numbers]
    time_serial = time.time() - start_serial
    print(f"Serial: {time_serial:.2f}s")

    with Pool(processes=4) as pool:
        parallel_start = time.time()
        results_parallel = pool.map(square,numbers)
        print('hello ;)')

    time_parallel = time.time() - parallel_start
    print(f"Parallel: {time_parallel:.2f}s")

    speedup = time_serial / time_parallel
    print(f"Speedup: {speedup:.2f}x")