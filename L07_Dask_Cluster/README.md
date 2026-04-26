# L07 - Dask (Cluster)
L07 scales Dask from a local machine to a multi-VM Strato cluster. The only code change is the Client() connection string, but performance characteristics shift due to Ethernet latency being far higher than IPC. Optimal chunk sizes become larger, and tasks must reduce network startup costs. The topic covers essential OpenStack concepts (instances, snapshots, security groups, floating IPs) and the workflow for creating a head node and spawning worker VMs. Mandelbrot is executed on the cluster, worker scaling is measured.

## Milestones
### Milestone 1 - Configure the Head Node
- [x] Open Extra Ports in Security Group
- [x] Resize VM (I'll probably use AAA.CPU.d.4-12)
- [x] SSH Into VM
- [x] Install MiniConda
- [x] Create Python Environment
- [x] Adapt Code from L06 for the Cluster
- [x] Test Locally

### Milestone 2 - Snapshot and Launch Worker VMs
- [x] Create a Snapshot
- [x] Launch Worker VMs from the Snapshot

### Milestone 3 - Start the Dask Cluster
- [x] Note the Head Node's IP
- [x] Start the Scheduler on the Head Node
- [x] Start a Worker on a VM

### Milestone 4 - Adapt Code and Run on the Head Node
- [x] SSH into Head Node
- [x] Run Script
- [x] Check the Dashboard
- [x] Smoke Test (Check that Each Worker is Identical)

### Milestone 5 - Benchmark
- [x] Sweep Resolution (1024 -> 2048 -> 4096 -> 8192 -> 16384)
- [x] Add Results to Performance Tracker
    - [x] Include Speed Up from Single-Core Numba at same Resolution

### Milestone 6 - Clean Up
- [x] Delete Worker VMs
- [x] Shelve Head Node
- [x] Keep Snapshots
---

## Results
This Implementation Compared to L06's Implementation, and L01's Implementation:

|Implementation|Compute Time|Speed-Up|
|:--|--:|--:|
|Naive (Baseline)        |3.56867 s| 1.000x|
|Dask Local (Last Week)  |0.06986 s|51.083x|
|Dask Cluster (This Week)|0.07562 s|47.192x|

Dask (Local) is faster than Dask (Cluster), because I was only able to connect 4 workers on the cluster, while I was able to have a worker on each of my 12 processors on my local CPU.