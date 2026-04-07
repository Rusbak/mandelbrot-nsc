import os, math, random, time, statistics
from multiprocessing import Pool

#region serial
def estimate_pi_serial(num_samples, _):
    inside_circ = 0

    for sample in range(num_samples):
        x, y = random.random(), random.random()

        if x*x + y*y <= 1:
            inside_circ += 1

    pi_estimate = 4 * inside_circ / num_samples

    return pi_estimate

#region parallel
def estimate_pi_chunk(num_samples):
    inside_circ = 0

    for sample in range(num_samples):
        x, y = random.random(), random.random()

        if x*x + y*y <= 1:
            inside_circ += 1
        
    return inside_circ

def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes

    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    
    pi_estimate = 4 * sum(results) / num_samples

    return pi_estimate

#region benchmark
def benchmark(func, num_samples, num_proc):
    times = []

    for test in range(3):
        warm_up = func(1000, num_proc)
        start_time = time.perf_counter()
        pi_estimate = func(num_samples, num_proc)
        test_time = time.perf_counter() - start_time
        times.append(test_time)

    median = statistics.median(times)

    print(f"Pi Estimate:{pi_estimate:.6f} (error: {abs(pi_estimate - math.pi):.6f})")
    print(f"Serial time:{median:.3f}s")

    return median

if __name__ == '__main__':
    num_samples = 10000000

    print('Serial')
    serial_median = benchmark(estimate_pi_serial, num_samples, 1)
    print()

    print('Parallel')
    parallel_medians = []
    for num_proc in range(1, os.cpu_count() + 1):
        print(f'Number of workers: {num_proc}')
        parallel_median = benchmark(estimate_pi_parallel, num_samples, num_proc)
        parallel_medians.append(parallel_median)
        print()

'''
Copy + Paste from terminal window:

Serial
Pi Estimate:3.141530 (error: 0.000062)
Serial time:1.311s

Parallel
Number of workers: 1
Pi Estimate:3.141017 (error: 0.000576)
Serial time:1.343s

Number of workers: 2
Pi Estimate:3.141832 (error: 0.000239)
Serial time:0.751s

Number of workers: 3
Pi Estimate:3.141982 (error: 0.000390)
Serial time:0.641s

Number of workers: 4
Pi Estimate:3.142470 (error: 0.000878)
Serial time:0.571s

Number of workers: 5
Pi Estimate:3.141190 (error: 0.000403)
Serial time:0.519s

Number of workers: 6
Pi Estimate:3.141756 (error: 0.000164)
Serial time:0.471s

Number of workers: 7
Pi Estimate:3.141115 (error: 0.000477)
Serial time:0.457s

Number of workers: 8
Pi Estimate:3.140858 (error: 0.000734)
Serial time:0.435s

Number of workers: 9
Pi Estimate:3.141344 (error: 0.000248)
Serial time:0.436s

Number of workers: 10
Pi Estimate:3.141854 (error: 0.000261)
Serial time:0.454s

Number of workers: 11
Pi Estimate:3.141530 (error: 0.000063)
Serial time:0.484s

Number of workers: 12
Pi Estimate:3.141691 (error: 0.000099)
Serial time:0.471s
'''