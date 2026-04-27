import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Chart 1: Implementations
# -----------------------
implementations = [
    "Naive Python",
    "Numpy Vectorization",
    "Numba (Serial)",
    "Numba (Chunking)",
    "Numba (Granularity)",
    "Dask (Local)",
    "Dask (Cluster)",
    "GPU (float32)"
]

times = [
    3.56867,
    1.44573,
    0.04954,
    0.02190,
    0.01294,
    0.06986,
    0.07562,
    0.0022
]

plt.figure()
plt.bar(implementations, times)
plt.yscale('log')
plt.xlabel("Implementation")
plt.ylabel("Time (seconds, log scale)")
plt.title("Mandelbrot Implementation Performance Comparison (1024)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("L10_GPU_Computation/Images/Implementation_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------
# Chart 2: Resolution Sweep
# -----------------------
resolutions = [1024, 2048, 4096, 8192, 16384]

dask_times = [0.0756, 0.1683, 0.5051, 1.9059, 7.5696]
gpu_times = [0.0022, 0.0052, 0.0167, 0.0618, 0.2379]

x = np.arange(len(resolutions))
width = 0.35

plt.figure()
plt.bar(x - width/2, dask_times, width, label="Dask")
plt.bar(x + width/2, gpu_times, width, label="GPU")

plt.yscale('log')
plt.xlabel("Resolution")
plt.ylabel("Time (seconds, log scale)")
plt.title("Resolution Scaling: Dask vs GPU")
plt.xticks(x, resolutions)
plt.legend()
plt.tight_layout()
plt.savefig("L10_GPU_Computation/Images/Resolution_Scaling.png", dpi=300, bbox_inches='tight')
plt.show()
