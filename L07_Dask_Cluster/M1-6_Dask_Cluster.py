import dask, numpy as np, time, statistics
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit

@njit
def compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res):
    result = np.zeros((res,res), dtype=np.int32)

    for i in range(res):
        for j in range(res):
            c = x_region[j] + y_region[i] * 1j
            z = 0j
            n = 0

            while n < max_iterations and z.real*z.real + z.imag*z.imag <= bound*bound:
                z = z*z + c
                n += 1
            result[i, j] = n

    return result

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iterations, bound):
    z_real = 0.0
    z_imag = 0.0
    bound_sq = bound * bound
    
    for iteration in range(max_iterations):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag

        if z_real_sq + z_imag_sq > bound_sq:
            return iteration
        else:
            z_imag = 2.0 * z_real * z_imag + c_imag
            z_real = z_real_sq - z_imag_sq + c_real

    return max_iterations

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound):
    row_start = int(row_start)
    row_end = int(row_end)
    res = int(res)

    chunk_output = np.empty((row_end - row_start, res), dtype=np.int32)
    
    dx = (x_max - x_min) / res
    dy = (y_max - y_min) / res

    for row in range(row_end - row_start):
        c_imag = y_min + (row + row_start) * dy

        for col in range(res):
            c_real = x_min + col * dx
            chunk_output[row, col] = mandelbrot_pixel(c_real, c_imag, max_iterations, bound)

    return chunk_output

def mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, n_chunks, bound):
    chunk_size = max(1, res // n_chunks)
    tasks = []
    row = 0

    while row < res:
        row_end = min(row + chunk_size, res)
        task = delayed(mandelbrot_chunk)(row, row_end, res, x_min, x_max, y_min, y_max, max_iterations, bound)
        tasks.append(task)
        row = row_end

    parts = dask.compute(*tasks)
    mandelbrot_stack = np.vstack(parts)

    return mandelbrot_stack

res = 1024
max_iterations = 100
bound = 2.0
power = 2

x_min, x_max = -2.5, 1.0
y_min, y_max = -1.5, 1.5
x_region = np.linspace(x_min, x_max, res)
y_region = np.linspace(y_min, y_max, res)

if __name__ == '__main__':
    resolutions = [1024, 2048, 4096, 8192, 16384]
    barebones_res_times = []
    baseline_res_times = []
    best_res_chunks = []
    best_chunk_times = []
    best_res_workers = []
    best_worker_times = []

    for res in resolutions:
        print(f'-=-=-=-=-=-=-=-=-=-\n RESOLUTION: {res} \n-=-=-=-=-=-=-=-=-=-')

        # barebones numba (single-core)
        print(f"Numba Barebones:")
        
        barebones_times = []
        for _ in range(5): # 5 tests
            warm_up = compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res)
            barebones_start = time.perf_counter()
            result = compute_numba_naive_mandelbrot(x_region, y_region, max_iterations, bound, power, res)
            barebones_time = time.perf_counter() - barebones_start
            barebones_times.append(barebones_time)

        barebones_median = statistics.median(barebones_times)
        barebones_res_times.append(barebones_median)
        print(f'Barebones Median: {barebones_median:.5f}s')

        print(' - - - - - \n')

        # baseline numba
        client = Client('tcp://10.92.0.67:8786') # replace with the actual address
        client.run(lambda:mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10, bound))
        num_workers = len(client.scheduler_info()['workers'])
        num_chunks = num_workers # one chunk per worker
        print(f"Dask Cluster Baseline ({num_workers} Workers | {num_chunks} Chunks):")
        
        baseline_times = []
        for _ in range(5): # 5 tests
            warm_up = mandelbrot_dask(128, x_min, x_max, y_min, y_max, 10, num_workers, bound)
            baseline_start = time.perf_counter()
            result = mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, num_chunks, bound)
            baseline_time = time.perf_counter() - baseline_start
            baseline_times.append(baseline_time)

        baseline_median = statistics.median(baseline_times)
        baseline_res_times.append(baseline_median)
        print(f"Baseline Median: {baseline_median:.5f}s")
        barebones_speedup = barebones_median / baseline_median
        print(f'Speed Up from Barebones: {barebones_speedup:.3f}x')

        print(' - - - - - \n')

        print('-=- Experiment 1 -=-')
        # Open Cluster before Benchmarking  
        client.run(lambda:mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10, bound))
        
        chunk_multiples = [1, 2, 4, 8, 16, 32]
        chunk_wall_times = []

        print(f"Dask Cluster Chunk Sweeping ({num_workers} Workers):")
        print(f"{'Total Chunks':>12} | {'Compute Time':>12} | {'x1':>8} | {'Speed Up':>8}")

        for chunk_multiple in chunk_multiples:
            num_chunks = chunk_multiple * num_workers

            chunk_times = []
            for _ in range(5): # 5 tests
                warm_up = mandelbrot_dask(128, x_min, x_max, y_min, y_max, 10, num_workers, bound)
                chunk_start = time.perf_counter()
                result = mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, num_chunks, bound)
                chunk_time = time.perf_counter() - chunk_start
                chunk_times.append(chunk_time)

            chunk_median = statistics.median(chunk_times)
            numba_speedup = barebones_median / chunk_median
            chunk_speed_up = baseline_median / chunk_median

            print(f'{num_chunks:12d} | {chunk_median:11.5f}s | {numba_speedup:7.3f}x | {chunk_speed_up:7.3f}x')
            
            chunk_wall_times.append(chunk_median)

        # fastest run
        min_wall_time = min(chunk_wall_times)
        best_chunk_times.append(min_wall_time)
        min_wall_idx = chunk_wall_times.index(min_wall_time)
        min_wall_chunks = chunk_multiples[min_wall_idx] * num_workers
        print(f'\nFastest Run: {min_wall_chunks} Chunks -> {min_wall_time:.5f}s')

        print(' - - - - - \n')

        print('-=- Experiment 2 -=-')
        # cluster should already be open from last experiment
        client.run(lambda:mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, 10, bound))
        
        chunks_per_worker = min_wall_chunks // num_workers # chunks per worker
        best_res_chunks.append(chunks_per_worker) # this finishes off experiment 1
        # max_workers = 12
        worker_wall_times = []

        print(f"Dask Cluster Worker Sweeping ({chunks_per_worker} Chunks per Worker):")
        print(f"{'Total Workers':>13} | {'Compute Time':>12} | {'x1':>8} | {'Speed Up':>8}")

        num_chunks = chunks_per_worker * num_workers

        worker_times = []
        for _ in range(5): # 5 tests
            warm_up = mandelbrot_dask(128, x_min, x_max, y_min, y_max, 10, num_workers, bound)
            worker_start = time.perf_counter()
            result = mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, num_chunks, bound)
            worker_time = time.perf_counter() - worker_start
            worker_times.append(worker_time)

        worker_median = statistics.median(worker_times)
        numba_speedup = barebones_median / worker_median
        worker_speed_up = baseline_median / worker_median

        print(f'{num_workers:13d} | {worker_median:11.5f}s | {numba_speedup:7.3f}x | {worker_speed_up:7.3f}x')
        
        worker_wall_times.append(worker_median)

        # this is able to scale/sweep worker amount programmaticaly, but does not work on clusters, which is what we are testing on... 
        # for num_worker in range(1, max_workers+1):
        #     num_chunks = chunks_per_worker * num_workers

        #     client.cluster.scale(num_workers) # changes the worker amount, rather than adding VMs/worker manually through terminal
        #     client.wait_for_workers(num_workers)

        #     worker_times = []
        #     for _ in range(5): # 5 tests
        #         warm_up = mandelbrot_dask(128, x_min, x_max, y_min, y_max, 10, num_workers, bound)
        #         worker_start = time.perf_counter()
        #         result = mandelbrot_dask(res, x_min, x_max, y_min, y_max, max_iterations, num_chunks, bound)
        #         worker_time = time.perf_counter() - worker_start
        #         worker_times.append(worker_time)

        #     worker_median = statistics.median(worker_times)
        #     numba_speedup = barebones_median / worker_median
        #     worker_speed_up = baseline_median / worker_median

        #     print(f'{num_workers:13d} | {worker_median:11.5f}s | {numba_speedup:7.3f}x | {worker_speed_up:7.3f}x')
            
        #     worker_wall_times.append(worker_median)

        # fastest run
        min_wall_time = min(worker_wall_times)
        best_worker_times.append(min_wall_time)
        min_wall_idx = worker_wall_times.index(min_wall_time)
        workers_per_vm = 4
        min_wall_workers = (min_wall_idx + 1) * workers_per_vm
        best_res_workers.append(min_wall_workers)
        print(f'\nFastest Run: {min_wall_workers} Workers -> {min_wall_time:.5f}s')

        print(' - - - - - \n')

        # summary of findings at each resolution
        print(f' -=- SUMMARY FOR RESOLUTION: {res} -=- ')
        print(f'Barebones: {barebones_median:.5f}s\n')

        print(f'Baseline: {baseline_median:.5f}s ({barebones_speedup:3f}x)\n')

        print(f'Best Chunk Amount per Worker: {chunks_per_worker} (All Workers Active)')
        intermediate_speed_up = baseline_median / chunk_median
        total_speed_up = barebones_median / chunk_median
        print(f"{'Time':>8} | {'Spd. Up':>7} | {'Tot. Spd. Up':>12}")
        print(f'{min_wall_time:7.5f}s | {intermediate_speed_up:6.3f}x | {total_speed_up:11.5f}x\n')

        print(f'Best Amount of Workers: {min_wall_workers} ({chunks_per_worker} Chunks)')
        intermediate_speed_up = baseline_median / worker_median
        total_speed_up = barebones_median / worker_median
        print(f"{'Time':>8} | {'Spd. Up':>7} | {'Tot. Spd. Up':>12}")
        print(f'{min_wall_time:7.5f}s | {intermediate_speed_up:6.3f}x | {total_speed_up:11.5f}x\n')

    client.close()

    print(f' -=- FINAL SUMMARY -=- ')
    print(f"{'Test Type':<12} | {resolutions[0]:>11} | {resolutions[1]:>11} | {resolutions[2]:>11} | {resolutions[3]:>11} | {resolutions[4]:>11}")
    print(f"{'Barebones':<12} | {barebones_res_times[0]:10.5f}s | {barebones_res_times[1]:10.5f}s | {barebones_res_times[2]:10.5f}s | {barebones_res_times[3]:10.5f}s | {barebones_res_times[4]:10.5f}s")
    print(f"{'Baseline ':<12} | {baseline_res_times[0]:10.5f}s | {baseline_res_times[1]:10.5f}s | {baseline_res_times[2]:10.5f}s | {baseline_res_times[3]:10.5f}s | {baseline_res_times[4]:10.5f}s")
    print(f"{'Chunk Sweep':<12} | {best_chunk_times[0]:.5f}({best_res_chunks[0]:>2d}) | {best_chunk_times[1]:.5f}({best_res_chunks[1]:>2d}) | {best_chunk_times[2]:.5f}({best_res_chunks[2]:>2d}) | {best_chunk_times[3]:.5f}({best_res_chunks[3]:>2d}) | {best_chunk_times[4]:.5f}({best_res_chunks[4]:>2d})")
    print(f"{'Worker Sweep':<12} | {best_worker_times[0]:.5f}({best_res_workers[0]:>2d}) | {best_worker_times[1]:.5f}({best_res_workers[1]:>2d} | {best_worker_times[2]:.5f}({best_res_workers[2]:>2d} | {best_worker_times[3]:.5f}({best_res_workers[3]:>2d} | {best_worker_times[4]:.5f}({best_res_workers[4]:>2d}")

'''
Copy + Paste from Strato Terminal (1 VM - 4 Workers)

-=-=-=-=-=-=-=-=-=-
 RESOLUTION: 1024
-=-=-=-=-=-=-=-=-=-
Numba Barebones:
Barebones Median: 0.07279s
 - - - - -

Dask Cluster Baseline (4 Workers | 4 Chunks):
Baseline Median: 0.06725s
Speed Up from Barebones: 1.082x
 - - - - -

-=- Experiment 1 -=-
Dask Cluster Chunk Sweeping (4 Workers):
Total Chunks | Compute Time |       x1 | Speed Up
           4 |     0.06926s |   1.051x |   0.971x
           8 |     0.06874s |   1.059x |   0.978x
          16 |     0.06899s |   1.055x |   0.975x
          32 |     0.09995s |   0.728x |   0.673x
          64 |     0.18653s |   0.390x |   0.361x
         128 |     0.33138s |   0.220x |   0.203x

Fastest Run: 8 Chunks -> 0.06874s
 - - - - -

-=- Experiment 2 -=-
Dask Cluster Worker Sweeping (2 Chunks per Worker):
Total Workers | Compute Time |       x1 | Speed Up
            4 |     0.07562s |   0.963x |   0.889x

Fastest Run: 4 Workers -> 0.07562s
 - - - - -

 -=- SUMMARY FOR RESOLUTION: 1024 -=-
Barebones: 0.07279s

Baseline: 0.06725s (1.082417x)

Best Chunk Amount per Worker: 2 (All Workers Active)
    Time | Spd. Up | Tot. Spd. Up
0.07562s |  0.203x |     0.21967x

Best Amount of Workers: 4 (2 Chunks)
    Time | Spd. Up | Tot. Spd. Up
0.07562s |  0.889x |     0.96267x

-=-=-=-=-=-=-=-=-=-
 RESOLUTION: 2048
-=-=-=-=-=-=-=-=-=-
Numba Barebones:
Barebones Median: 1.03823s
 - - - - -

Dask Cluster Baseline (4 Workers | 4 Chunks):
Baseline Median: 0.19891s
Speed Up from Barebones: 5.220x
 - - - - -

-=- Experiment 1 -=-
Dask Cluster Chunk Sweeping (4 Workers):
Total Chunks | Compute Time |       x1 | Speed Up
           4 |     0.20397s |   5.090x |   0.975x
           8 |     0.19505s |   5.323x |   1.020x
          16 |     0.16900s |   6.143x |   1.177x
          32 |     0.16947s |   6.126x |   1.174x
          64 |     0.21664s |   4.793x |   0.918x
         128 |     0.29627s |   3.504x |   0.671x

Fastest Run: 16 Chunks -> 0.16900s
 - - - - -

-=- Experiment 2 -=-
Dask Cluster Worker Sweeping (4 Chunks per Worker):
Total Workers | Compute Time |       x1 | Speed Up
            4 |     0.16827s |   6.170x |   1.182x

Fastest Run: 4 Workers -> 0.16827s
 - - - - -

 -=- SUMMARY FOR RESOLUTION: 2048 -=-
Barebones: 1.03823s

Baseline: 0.19891s (5.219636x)

Best Chunk Amount per Worker: 4 (All Workers Active)
    Time | Spd. Up | Tot. Spd. Up
0.16827s |  0.671x |     3.50436x

Best Amount of Workers: 4 (4 Chunks)
    Time | Spd. Up | Tot. Spd. Up
0.16827s |  1.182x |     6.17005x

-=-=-=-=-=-=-=-=-=-
 RESOLUTION: 4096
-=-=-=-=-=-=-=-=-=-
Numba Barebones:
Barebones Median: 3.69837s
 - - - - -

Dask Cluster Baseline (4 Workers | 4 Chunks):
Baseline Median: 0.69447s
Speed Up from Barebones: 5.325x
 - - - - -

-=- Experiment 1 -=-
Dask Cluster Chunk Sweeping (4 Workers):
Total Chunks | Compute Time |       x1 | Speed Up
           4 |     0.67309s |   5.495x |   1.032x
           8 |     0.64903s |   5.698x |   1.070x
          16 |     0.54360s |   6.804x |   1.278x
          32 |     0.52549s |   7.038x |   1.322x
          64 |     0.51774s |   7.143x |   1.341x
         128 |     0.59146s |   6.253x |   1.174x

Fastest Run: 64 Chunks -> 0.51774s
 - - - - -

-=- Experiment 2 -=-
Dask Cluster Worker Sweeping (16 Chunks per Worker):
Total Workers | Compute Time |       x1 | Speed Up
            4 |     0.50509s |   7.322x |   1.375x

Fastest Run: 4 Workers -> 0.50509s
 - - - - -

 -=- SUMMARY FOR RESOLUTION: 4096 -=-
Barebones: 3.69837s

Baseline: 0.69447s (5.325474x)

Best Chunk Amount per Worker: 16 (All Workers Active)
    Time | Spd. Up | Tot. Spd. Up
0.50509s |  1.174x |     6.25294x

Best Amount of Workers: 4 (16 Chunks)
    Time | Spd. Up | Tot. Spd. Up
0.50509s |  1.375x |     7.32216x

-=-=-=-=-=-=-=-=-=-
 RESOLUTION: 8192
-=-=-=-=-=-=-=-=-=-
Numba Barebones:
Barebones Median: 8.41157s
 - - - - -

Dask Cluster Baseline (4 Workers | 4 Chunks):
Baseline Median: 2.65049s
Speed Up from Barebones: 3.174x
 - - - - -

-=- Experiment 1 -=-
Dask Cluster Chunk Sweeping (4 Workers):
Total Chunks | Compute Time |       x1 | Speed Up
           4 |     2.58346s |   3.256x |   1.026x
           8 |     2.65542s |   3.168x |   0.998x
          16 |     2.12263s |   3.963x |   1.249x
          32 |     1.92736s |   4.364x |   1.375x
          64 |     1.85637s |   4.531x |   1.428x
         128 |     1.82216s |   4.616x |   1.455x

Fastest Run: 128 Chunks -> 1.82216s
 - - - - -

-=- Experiment 2 -=-
Dask Cluster Worker Sweeping (32 Chunks per Worker):
Total Workers | Compute Time |       x1 | Speed Up
            4 |     1.90592s |   4.413x |   1.391x

Fastest Run: 4 Workers -> 1.90592s
 - - - - -

 -=- SUMMARY FOR RESOLUTION: 8192 -=-
Barebones: 8.41157s

Baseline: 2.65049s (3.173592x)

Best Chunk Amount per Worker: 32 (All Workers Active)
    Time | Spd. Up | Tot. Spd. Up
1.90592s |  1.455x |     4.61626x

Best Amount of Workers: 4 (32 Chunks)
    Time | Spd. Up | Tot. Spd. Up
1.90592s |  1.391x |     4.41340x

-=-=-=-=-=-=-=-=-=-
 RESOLUTION: 16384
-=-=-=-=-=-=-=-=-=-
Numba Barebones:
Barebones Median: 28.70410s
 - - - - -

Dask Cluster Baseline (4 Workers | 4 Chunks):
Baseline Median: 10.60500s
Speed Up from Barebones: 2.707x
 - - - - -

-=- Experiment 1 -=-
Dask Cluster Chunk Sweeping (4 Workers):
Total Chunks | Compute Time |       x1 | Speed Up
           4 |    10.48102s |   2.739x |   1.012x
           8 |    10.67923s |   2.688x |   0.993x
          16 |     8.37795s |   3.426x |   1.266x
          32 |     6.99556s |   4.103x |   1.516x
          64 |     7.04489s |   4.074x |   1.505x
         128 |     7.07071s |   4.060x |   1.500x

Fastest Run: 32 Chunks -> 6.99556s
 - - - - -

-=- Experiment 2 -=-
Dask Cluster Worker Sweeping (8 Chunks per Worker):
Total Workers | Compute Time |       x1 | Speed Up
            4 |     7.56957s |   3.792x |   1.401x

Fastest Run: 4 Workers -> 7.56957s
 - - - - -

 -=- SUMMARY FOR RESOLUTION: 16384 -=-
Barebones: 28.70410s

Baseline: 10.60500s (2.706659x)

Best Chunk Amount per Worker: 8 (All Workers Active)
    Time | Spd. Up | Tot. Spd. Up
7.56957s |  1.500x |     4.05958x

Best Amount of Workers: 4 (8 Chunks)
    Time | Spd. Up | Tot. Spd. Up
7.56957s |  1.401x |     3.79204x

 -=- FINAL SUMMARY -=-
Test Type    |        1024 |        2048 |        4096 |        8192 |       16384
Barebones    |    0.07279s |    1.03823s |    3.69837s |    8.41157s |   28.70410s
Baseline     |    0.06725s |    0.19891s |    0.69447s |    2.65049s |   10.60500s
Chunk Sweep  | 0.06874( 2) | 0.16900( 4) | 0.51774(16) | 1.82216(32) | 6.99556( 8)
Worker Sweep | 0.07562( 4) | 0.16827( 4) | 0.50509( 4) | 1.90592( 4) | 7.56957( 4)
'''