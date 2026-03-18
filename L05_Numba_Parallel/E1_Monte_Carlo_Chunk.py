from multiprocessing import Pool
import random
import time
import os
import math

def monte_carlo_chunk(num_samples):
    inside = 0

    for sample in range(num_samples):
        x, y = random.random(), random.random()
    
        if x*x + y*y <= 1:
            inside += 1
    
    return inside

def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks

    test_start = time.perf_counter()
    if n_proc == 1:
        results = [monte_carlo_chunk(s) for s in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(monte_carlo_chunk,tasks)
    test_time = time.perf_counter() - test_start
    
    pi_approx = 4 * sum(results) / total_work

    return test_time, pi_approx

if __name__ == '__main__':
    total_work = 1000000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1000, 10000, 100000, 1000000]

    print(f"{'Chunk Size':>10} | {'Serial(s)':>9} | {'Parallel(s)':>11} | {'Approx':>7} | {'Error(abs)':>10}")
    for chunk_size in chunk_sizes:
        serial_time, _ = test_granularity(total_work, chunk_size, n_proc=1)
        parallel_time, pi = test_granularity(total_work, chunk_size, n_proc=n_proc)
        abs_error = abs(pi - math.pi)

        print(f"{chunk_size:10d} | {serial_time:9.5f} | {parallel_time:11.5f} | {pi:6.5f} | {abs_error:10.5f}")

'''Copy+paste from terminal window

Chunk Size | Serial(s) | Parallel(s) |  Approx | Error(abs)
        10 |   0.13098 |     0.20062 | 3.14118 |    0.00041
       100 |   0.11297 |     0.16966 | 3.14239 |    0.00080
      1000 |   0.12165 |     0.17868 | 3.14221 |    0.00062
     10000 |   0.12732 |     0.17501 | 3.14208 |    0.00049
    100000 |   0.12601 |     0.17479 | 3.14056 |    0.00103
   1000000 |   0.13161 |     0.26944 | 3.14113 |    0.00046
'''