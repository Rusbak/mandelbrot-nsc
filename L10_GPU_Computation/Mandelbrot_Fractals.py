import numpy as np
import time
import pyopencl as cl
import matplotlib.pyplot as plt

# some fractals are prettier at specific coordinates 
views = {
    "Mandelbrot":   (-2.5,  1.0, -1.25, 1.25),
    "Julia":        (-1.5,  1.5, -1.0,  1.0 ),
    "Burning Ship": (-2.2,  1.3, -2.0,  1.0 ),
    "Tricorn":      (-2.2,  1.5, -1.5,  1.5 ),
}
max_iterations = 200

# kernels for the different fractals
KERNELS = {}

# normal mandelbrot
KERNELS["Mandelbrot"] = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0); int row = get_global_id(1);
    if (col >= N || row >= N) return;

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

# julia set - c is constant and z_0 is determined by the pixel
KERNELS["Julia"] = """
__kernel void julia(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const float c_real, const float c_imag,
    const int N, const int max_iter)
{
    int col = get_global_id(0); int row = get_global_id(1);
    if (col >= N || row >= N) return;

    // z_0 = pixel coordinate; c is fixed.
    float zr = x_min + col * (x_max - x_min) / (float)N;
    float zi = y_min + row * (y_max - y_min) / (float)N;

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

# burning chip - squares the absolute value of z
KERNELS["Burning Ship"] = """
__kernel void burning_ship(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0); int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float ar = fabs(zr);
        float ai = fabs(zi);
        float tmp = ar*ar - ai*ai + c_real;
        zi = 2.0f * ar * ai + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# tricorn - the imaginary part is ignored
KERNELS["Tricorn"] = """
__kernel void tricorn(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0); int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = -2.0f * zr * zi + c_imag;      // <-- sign flip
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

def compute_fractal(ctx, queue, name, res, max_iter=max_iterations):
    x_min, x_max, y_min, y_max = views[name]

    prog = cl.Program(ctx, KERNELS[name]).build()
    kernel_name = {
        "Mandelbrot":   "mandelbrot",
        "Julia":        "julia",
        "Burning Ship": "burning_ship",
        "Tricorn":      "tricorn",
    }[name]
    kernel = getattr(prog, kernel_name)

    result_host = np.zeros((res, res), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)

    # julia takes two extra scalar args (c_real, c_imag).
    t0 = time.perf_counter()
    if name == "Julia":
        julia_c = (-0.7, 0.27015)
        kernel(
            queue, (res, res), None, result_dev,
            np.float32(x_min), np.float32(x_max),
            np.float32(y_min), np.float32(y_max),
            np.float32(julia_c[0]), np.float32(julia_c[1]),
            np.int32(res), np.int32(max_iter),
        )
    else:
        kernel(
            queue, (res, res), None, result_dev,
            np.float32(x_min), np.float32(x_max),
            np.float32(y_min), np.float32(y_max),
            np.int32(res), np.int32(max_iter),
        )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, result_host, result_dev)
    queue.finish()

    return result_host, elapsed

if __name__ == "__main__":
    res = 1024

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    print(f"Device: {ctx.devices[0].name}\n")

    # warm up each kernel
    for name in KERNELS:
        compute_fractal(ctx, queue, name, res=64)

    images = {}
    for name in KERNELS:
        img, t = compute_fractal(ctx, queue, name, res)
        images[name] = img
        print(f"  {name:<13}: {t*1e3:6.1f} ms  (view={views[name]})")

    output_dir = "L10_GPU_Computation/Images"

    for name, img in images.items():
        if name == "Burning Ship":
            img = img[::-1]  # looks better like this

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='twilight_shifted', origin='lower')
        plt.title(name, fontsize=12)
        plt.axis('off')

        filename = f"{name.replace(' ', '_')}.png"
        plt.savefig(f'{output_dir}/Fractal_{filename}', dpi=150, bbox_inches='tight')
        plt.close()
