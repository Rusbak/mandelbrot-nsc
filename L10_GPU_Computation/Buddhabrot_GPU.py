import numpy as np
import time
import pyopencl as cl
import matplotlib.pyplot as plt

num_samples = 1 << 24
max_iterations = 500
res = 800

x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

KERNEL_SRC = """
// each work-item carries its own 'state' so parallel streams don't collide
inline uint xorshift32(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// map a random uint32 to a float in [0, 1).
inline float rand01(uint *state) {
    return (float)xorshift32(state) * (1.0f / 4294967296.0f);
}

__kernel void buddhabrot(
    __global uint *hist, // res * res histogram
    const int res, // image side length
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int max_iter,
    const uint base_seed)
{
    int gid = get_global_id(0);

    // seed the PRNG so every work-item gets a different stream.
    // (multiplying by a large odd constant spreads nearby gids apart)
    uint rng = base_seed ^ ((uint)gid * 2654435761u);
    if (rng == 0u) rng = 1u; // xorshift can't start from zero

    // pick a random complex point c
    float c_real = x_min + rand01(&rng) * (x_max - x_min);
    float c_imag = y_min + rand01(&rng) * (y_max - y_min);

    // does c escape?
    float zr = 0.0f, zi = 0.0f;
    int   iter = 0;
    while (iter < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        iter++;
    }

    // c is in the set -> its trajectory is uninformative, skip.
    if (iter >= max_iter) return;

    // re-iterate and record the trajectory
    // We know c escapes in 'iter' steps. Replay that orbit and drop a hit
    // into the histogram bin each z_k lands in.
    zr = 0.0f; zi = 0.0f;
    for (int k = 0; k < iter; k++) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;

        // map (zr, zi) to pixel coordinates. skip if the orbit went
        // outside the view window.
        if (zr < x_min || zr >= x_max || zi < y_min || zi >= y_max) continue;
        int col = (int)((zr - x_min) / (x_max - x_min) * (float)res);
        int row = (int)((zi - y_min) / (y_max - y_min) * (float)res);

        // atomic scatter write
        // many work-items may try to increment the same bin at once.
        // atomic_inc serialises per-bin, so no increments are lost.
        atomic_inc(&hist[row * res + col]);
    }
}
"""

def run(ctx, queue):
    prog = cl.Program(ctx, KERNEL_SRC).build()

    hist = np.zeros((res, res), dtype=np.uint32)
    hist_dev = cl.Buffer(
        ctx,
        cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=hist,
    )

    t0 = time.perf_counter()
    prog.buddhabrot(
        queue, (num_samples,), None,
        hist_dev,
        np.int32(res),
        np.float32(x_min), np.float32(x_max),
        np.float32(y_min), np.float32(y_max),
        np.int32(max_iterations),
        np.uint32(0xC0FFEE),
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
    print(f"Samples: {num_samples:,} | Max Iterations: {max_iterations} | Image Resolution: {res}x{res}")

    hist, elapsed = run(ctx, queue)
    rate = num_samples / elapsed / 1e6
    print(f"Elapsed:{elapsed:.2f} s ({rate:.1f} M samples/s)")
    print(f"Hist: Sum={hist.sum():,} | Max Bin={hist.max():,}")

    # log(1+x) compresses the dynamic range so filaments stay visible
    img = np.log1p(hist.astype(np.float32))

    plt.figure(figsize=(6, 6))
    # .T for the classic 'seated Buddha' orientation (imaginary axis horizontal)
    plt.imshow(img.T, cmap='twilight', origin='lower')
    plt.title(
        f"Buddhabrot (simple) — {num_samples/1e6:.0f}M samples, {elapsed:.1f}s",
        fontsize=10,
    )
    plt.axis('off')
    plt.tight_layout()
    # plt.savefig("L10_GPU_Computation/Images/Buddhabrot.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
