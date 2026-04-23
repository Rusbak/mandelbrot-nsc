import numpy as np
import pyopencl as cl

# create 'some' context and a command queue
ctx   = cl.create_some_context(interactive=False)   # picks first available
queue = cl.CommandQueue(ctx)

dev = ctx.devices[0]
print(f"Device: {dev.name}")
print(f"  Vendor:  {dev.vendor}")
print(f"  OpenCL:  {dev.version}")
print(f"  Compute units: {dev.max_compute_units}")
print()

# square each element of a float32 array in the kernel
kernel_source = """
__kernel void square(__global float *a) {
    int i = get_global_id(0);
    a[i] = a[i] * a[i];
}
"""

prog = cl.Program(ctx, kernel_source).build()

# allocate the host and device buffers
a_host = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
result  = np.empty_like(a_host)

mf      = cl.mem_flags
a_dev   = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_host)

# launch the kernel
prog.square(queue, a_host.shape, None, a_dev)

# copy the result back to the host
cl.enqueue_copy(queue, result, a_dev)
queue.finish()

print("Kernel output:", result)   # expected: [1. 4. 9. 16.]

# verify result
expected = a_host ** 2
ok = np.allclose(result, expected)
print("All elements close?", ok)

if not ok:
    print("MISMATCH! Expected:", expected, "  Got:", result)
    raise SystemExit(1)

'''Copy + paste from terminal window
Device: Intel(R) Iris(R) Xe Graphics
Vendor:  Intel(R) Corporation
OpenCL:  OpenCL 3.0 NEO 
Compute units: 80

Kernel output: [ 1.  4.  9. 16.]
All elements close? True
'''