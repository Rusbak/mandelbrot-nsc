import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import statistics

# mandelbrot domain
x_min, x_max = -2.5, 1.0
y_min, y_max = -1.25, 1.25
max_iterations = 100

# float32 kernel
KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iterations)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float z_real = 0.0f, z_imag = 0.0f;
    int count = 0;
    while (count < max_iterations && z_real*z_real + z_imag*z_imag <= 4.0f) {
        float tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# float64 kernel
KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iterations)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double z_real = 0.0, z_imag = 0.0;
    int count = 0;
    while (count < max_iterations && z_real*z_real + z_imag*z_imag <= 4.0) {
        double tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""


def mandelbrot_gpu_f32(ctx, queue, N, x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max, max_iterations=max_iterations):
    prog = cl.Program(ctx, KERNEL_F32).build()
    result_host = np.zeros((N, N), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)

    prog.mandelbrot_f32(
        queue, (N, N), None, result_dev,
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(N), np.int32(max_iterations))
    cl.enqueue_copy(queue, result_host, result_dev)
    queue.finish()
    return result_host


def mandelbrot_gpu_f64(ctx, queue, N, x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max, max_iterations=max_iterations):
    dev = ctx.devices[0]
    if 'cl_khr_fp64' not in dev.extensions:
        print(f"  {dev.name} does not support float64 (cl_khr_fp64 missing).")
        return None

    prog = cl.Program(ctx, KERNEL_F64).build()
    result_host = np.zeros((N, N), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)

    prog.mandelbrot_f64(
        queue, (N, N), None, result_dev,
        np.float64(x_min), np.float64(x_max),
        np.float64(y_min), np.float64(y_max),
        np.int32(N), np.int32(max_iterations))
    cl.enqueue_copy(queue, result_host, result_dev)
    queue.finish()
    return result_host


def benchmark(func, *args, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    N = 1024
    runs = 3

    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev   = ctx.devices[0]
    print(f"Device: {dev.name}\n")

    # Warm-up pass (triggers kernel compilation)
    _ = mandelbrot_gpu_f32(ctx, queue, N=64)

    print(f"Benchmarking N={N}, max_iterations={max_iterations}, {runs} runs each:\n")

    # --- float32 ---
    t32 = benchmark(mandelbrot_gpu_f32, ctx, queue, N, runs=runs)
    img_f32 = mandelbrot_gpu_f32(ctx, queue, N)
    print(f"  float32: {t32*1e3:.1f} ms")

    # --- float64 ---
    t64 = None
    img_f64 = mandelbrot_gpu_f64(ctx, queue, N)
    if img_f64 is not None:
        t64 = benchmark(mandelbrot_gpu_f64, ctx, queue, N, runs=runs)
        print(f"  float64: {t64*1e3:.1f} ms")
        print(f"  Ratio float64/float32: {t64/t32:.2f}x")
        diff = np.abs(img_f32.astype(int) - img_f64.astype(int))
        print(f"  Max pixel difference (f32 vs f64): {diff.max()}")

    # --- Visualise ---
    fig, axes = plt.subplots(1, 2 if img_f64 is not None else 1,
                                figsize=(12 if img_f64 is not None else 6, 5))
    if img_f64 is None:
        axes = [axes]

    axes[0].imshow(img_f32, cmap='hot', origin='lower')
    axes[0].set_title(f"float32  ({t32*1e3:.1f} ms)")
    axes[0].axis('off')

    if img_f64 is not None:
        axes[1].imshow(img_f64, cmap='hot', origin='lower')
        axes[1].set_title(f"float64  ({t64*1e3:.1f} ms)")
        axes[1].axis('off')

    plt.suptitle(f"GPU Mandelbrot  N={N}  device: {dev.name}", fontsize=10)
    plt.tight_layout()
    out = "Images/Mandelbrot_opencl.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nSaved to {out}")
