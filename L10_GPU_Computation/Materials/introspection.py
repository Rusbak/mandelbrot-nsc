# Find out about your computer's OpenCL situation
# From: https://github.com/benshope/PyOpenCL-Tutorial

# Import the OpenCL GPU computing API
import pyopencl as cl

print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')

# print each platform on this computer
for platform in cl.get_platforms():
    print('=' * 60)
    print('Platform - Name:  ' + platform.name)
    print('Platform - Vendor:  ' + platform.vendor)
    print('Platform - Version:  ' + platform.version)
    print('Platform - Profile:  ' + platform.profile)

    # print each device per-platform
    for device in platform.get_devices():
        print(f'\t ' + '-' * 56)
        print(f'\t Device - Name:  ' + device.name)
        print(f'\t Device - Type:  ' + cl.device_type.to_string(device.type))
        print(f'\t Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
        print(f'\t Device - Compute Units:  {0}'.format(device.max_compute_units))
        print(f'\t Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024.0))
        print(f'\t Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024.0))
        print(f'\t Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
        print(f'\t Device - Max Buffer/Image Size: {0:.0f} MB'.format(device.max_mem_alloc_size/1048576.0))
        print(f'\t Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
