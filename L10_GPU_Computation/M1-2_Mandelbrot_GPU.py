import pyopencl as cl
import numpy as np
import time, statistics
from matplotlib import pyplot as plt

res = 1024
max_iterations = 100
x_min, x_max = -2.5, 1.0
y_min, y_max = -1.25, 1.25

# precision: float32
KERNEL_SRC_32 = """
__kernel void mandelbrot_32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog = cl.Program(ctx, KERNEL_SRC_32).build()

image_32 = np.zeros((res, res), dtype=np.int32)
image_dev_32 = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image_32.nbytes)

# warm-up
prog.mandelbrot_32(queue, (64, 64), None, image_dev_32,
                np.float32(x_min), np.float32(x_max),
                np.float32(y_min), np.float32(y_max),
                np.int32(64), np.int32(max_iterations))
queue.finish()

# real run
test_amount = 16
test_times = []

for test in range(test_amount):
    t0 = time.perf_counter()

    prog.mandelbrot_32(queue, (res, res), None, image_dev_32,
                    np.float32(x_min), np.float32(x_max),
                    np.float32(y_min), np.float32(y_max),
                    np.int32(res), np.int32(max_iterations))
    queue.finish()

    test_time = time.perf_counter() - t0
    test_times.append(test_time)

median = statistics.median(test_times)

cl.enqueue_copy(queue, image_32, image_dev_32)
queue.finish()

print(f"GPU {res}x{res}: {median*1e3:.1f} ms (float32)")
plt.imshow(image_32, cmap='twilight_shifted', origin='lower'); 
plt.axis('off')
plt.savefig("L10_GPU_Computation/Images/Mandelbrot_GPU_32.png", dpi=300, bbox_inches='tight')

# precision: float64
KERNEL_SRC_64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog  = cl.Program(ctx, KERNEL_SRC_64).build()

image_64 = np.zeros((res, res), dtype=np.int32)
image_dev_64 = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image_64.nbytes)

# warm-up
prog.mandelbrot_64(queue, (64, 64), None, image_dev_64,
                np.float32(x_min), np.float32(x_max),
                np.float32(y_min), np.float32(y_max),
                np.int32(64), np.int32(max_iterations))
queue.finish()

# real run
test_amount = 16
test_times = []

for test in range(test_amount):
    t0 = time.perf_counter()

    prog.mandelbrot_64(queue, (res, res), None, image_dev_64,
                    np.float32(x_min), np.float32(x_max),
                    np.float32(y_min), np.float32(y_max),
                    np.int32(res), np.int32(max_iterations))
    queue.finish()

    test_time = time.perf_counter() - t0
    test_times.append(test_time)

median = statistics.median(test_times)

cl.enqueue_copy(queue, image_64, image_dev_64)
queue.finish()

print(f"GPU {res}x{res}: {median*1e3:.1f} ms (float64)")
plt.imshow(image_64, cmap='twilight_shifted', origin='lower'); 
plt.axis('off')
plt.savefig("L10_GPU_Computation/Images/Mandelbrot_GPU_64.png", dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 2)

# Show images
axes[0].imshow(image_32)
axes[0].set_title("float32")
axes[0].axis('off')

axes[1].imshow(image_64)
axes[1].set_title("float64")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('L10_GPU_Computation/Images/Precision_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
