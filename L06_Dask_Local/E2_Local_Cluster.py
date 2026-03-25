import dask
from dask.distributed import Client, LocalCluster
from dask import delayed
from E1_Dask_Delayed import monte_carlo_chunk

if __name__ == '__main__':
    # local cluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")

    # rerun exercise 1 with local cluster
    total_samples = 1000000
    num_chunks = 8
    num_samples = total_samples // num_chunks

    tasks = [delayed(monte_carlo_chunk)(num_samples) for chunk in range(num_chunks)]
    results = dask.compute(*tasks)

    #Vary n_workers: scale() resizes without restarting the scheduler
    #(recreating Local Cluster while the browser is open breaks the dashboard)
    cluster.scale(4); 
    client.wait_for_workers(4)
    tasks = [delayed(monte_carlo_chunk)(num_samples) for chunk in range(num_chunks)]
    results = dask.compute(*tasks)

    print('shutting down')
    client.close(); 
    cluster.close()