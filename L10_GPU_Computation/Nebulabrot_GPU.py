import numpy as np
import time
import pyopencl as cl
import matplotlib.pyplot as plt

num_samples = 1 << 26
max_iterations = 2500
res = 1250

x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

KERNEL_SRC = """
inline uint xorshift32(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

inline float rand01(uint *state) {
    return (float)xorshift32(state) * (1.0f / 4294967296.0f);
}

__kernel void nebulabrot(
    __global uint *hist, // size = 3 * res * res
    const int res,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int max_iter,
    const uint base_seed)
{
    int gid = get_global_id(0);

    uint rng = base_seed ^ ((uint)gid * 2654435761u);
    if (rng == 0u) rng = 1u;

    // random complex c
    float c_real = x_min + rand01(&rng) * (x_max - x_min);
    float c_imag = y_min + rand01(&rng) * (y_max - y_min);

    float zr = 0.0f, zi = 0.0f;
    int iter = 0;

    while (iter < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        iter++;
    }

    // skip non-escaping points
    if (iter >= max_iter) return;

    // iteration bands
    int r_max = 15;
    int g_max = 75;
    int b_max = max_iter;

    zr = 0.0f;
    zi = 0.0f;

    for (int k = 0; k < iter; k++) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;

        if (zr < x_min || zr >= x_max || zi < y_min || zi >= y_max)
            continue;

        int col = (int)((zr - x_min) / (x_max - x_min) * (float)res);
        int row = (int)((zi - y_min) / (y_max - y_min) * (float)res);
        int idx = row * res + col;

        if (k < r_max) {
            atomic_inc(&hist[0 * res * res + idx]); // RED
        } else if (k < g_max) {
            atomic_inc(&hist[1 * res * res + idx]); // GREEN
        } else {
            atomic_inc(&hist[2 * res * res + idx]); // BLUE
        }
    }
}
"""

def run(ctx, queue):
    prog = cl.Program(ctx, KERNEL_SRC).build()

    hist = np.zeros((3, res, res), dtype=np.uint32)
    hist_dev = cl.Buffer(
        ctx,
        cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=hist
    )

    t0 = time.perf_counter()
    prog.nebulabrot(
        queue, (num_samples,), None,
        hist_dev,
        np.int32(res),
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(max_iterations),
        np.uint32(0xC0FFEE)
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, hist, hist_dev)
    queue.finish()

    return hist, elapsed

if __name__ == "__main__":
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    print(f"Device: {ctx.devices[0].name}")
    print(f"Samples: {num_samples:,} | Max Iterations: {max_iterations} | Resolution: {res}x{res}")

    hist, elapsed = run(ctx, queue)
    rate = num_samples / elapsed / 1e6
    print(f"Elapsed: {elapsed:.2f}s ({rate:.1f} M samples/s)")
    print(f"Hist sum: {hist.sum():,} | Max bin: {hist.max():,}")

    img = np.log1p(hist.astype(np.float32))

    # normalize each channel independently
    for i in range(3):
        img[i] /= (img[i].max() + 1e-8)

    # stack into RGB
    rgb = np.stack([img[0], img[1], img[2]], axis=-1)

    # transpose for classic orientation
    rgb = rgb.transpose(1, 0, 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin='lower')
    plt.title(f"Nebulabrot — {num_samples/1e6:.0f}M samples, {elapsed:.1f}s", fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("L10_GPU_Computation/Images/Nebulabrot.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()