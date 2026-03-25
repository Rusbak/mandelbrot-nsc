import time, dask
import numpy as np
from dask import delayed
from dask.distributed import Client, LocalCluster

@delayed
def generate(seed, n):
    time.sleep(0.3)
    chunk = np.random.default_rng(seed).standard_normal(n)
    return chunk

@delayed
def chunk_max(data):
    time.sleep(0.2)
    maxima = float(np.max(np.abs(data)))
    return maxima

@delayed
def global_max(maxima):
    time.sleep(0.2)
    glob_max = max(maxima)
    return glob_max

@delayed
def normalize(data, glob_max):
    time.sleep(0.3)
    norm = data / glob_max

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(client.dashboard_link)

    chunks = [generate(i, 50000) for i in range(8)]
    maxima = [chunk_max(chunk) for chunk in chunks]
    glob_max = global_max(maxima)
    normed = [normalize(chunk, glob_max) for chunk in chunks]

    # dask.visualize(*normed) # visualize tool does not work
    test_start = time.perf_counter()
    results = dask.compute(*normed)
    test_time = time.perf_counter() - test_start
    print(f'Wall Time: {test_time:.5f}s')

    client.close()
    cluster.close()
    