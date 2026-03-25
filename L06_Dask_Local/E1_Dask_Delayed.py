import dask, random, time, statistics
from dask import delayed

def monte_carlo_chunk(num_samples):
    inside_circ = 0

    for sample in range(num_samples):
        x, y, = random.random(), random.random()

        if x*x + y*y <= 1:
            inside_circ += 1
        
    return inside_circ

total_samples = 1000000
num_chunks = 8
num_samples = total_samples // num_chunks

if __name__ == '__main__':
    # serial baseline
    serial_start = time.perf_counter()
    serial_results = [monte_carlo_chunk(num_samples) for chunk in range(num_chunks)]
    serial_time = time.perf_counter() - serial_start
    pi_serial = 4 * sum(serial_results) / total_samples
    print(f'Serial: {serial_time:.5f}s | pi est.: {pi_serial:.4f}')

    # dask delayed
    tasks = [delayed(monte_carlo_chunk)(num_samples) for chunk in range(num_chunks)]
    dask_start = time.perf_counter()
    dask_results = dask.compute(*tasks)
    dask_time = time.perf_counter() - dask_start
    pi_dask = 4 * sum(dask_results) / total_samples
    print(f'Dask: {serial_time:.5f}s | pi est.: {pi_dask:.4f}')

    # # visualization
    # # dask.config.set({'visualization.engine': 'python-graphviz'})
    # dask.visualize(*tasks, filename='task_graph.png') # visual tool has unfixable issues...
