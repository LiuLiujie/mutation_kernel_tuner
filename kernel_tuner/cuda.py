"""This module contains all CUDA specific kernel_tuner functions"""
import numpy

#embedded in try block to be able to generate documentation
try:
    import pycuda.driver as drv
    from pycuda.autoinit import context
    from pycuda.compiler import SourceModule
except Exception:
    pass



class CudaFunctions(object):
    """Class that groups the CUDA functions on maintains some state about the device"""


    def __init__(self, device=0):
        #inspect device properties
        devprops = { str(k): v for (k, v) in drv.Device(device).get_attributes().items() }
        self.max_threads = devprops['MAX_THREADS_PER_BLOCK']
        self.cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])


    def create_gpu_args(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem"""
        gpu_args = []
        for arg in arguments:
            # if arg i is a numpy array copy to device
            if isinstance(arg, numpy.ndarray):
                gpu_args.append(drv.mem_alloc(arg.nbytes))
                drv.memcpy_htod(gpu_args[-1], arg)
            else: # if not an array, just pass argument along
                gpu_args.append(arg)
        return gpu_args


    def compile(self, kernel_name, kernel_string):
        """call the CUDA compiler to compile the kernel, return the device function"""
        try:
            func = SourceModule(kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + self.cc, code='sm_' + self.cc,
                    cache_dir=False).get_function(kernel_name)
            return func
        except drv.CompileError, e:
            if "uses too much shared data" in e.stderr:
                raise Exception("uses too much shared data")
            else:
                raise e

    def benchmark(self, func, gpu_args, threads, grid):
        """runs the kernel and measures time repeatedly, returns average time"""
        ITERATIONS = 7
        start = drv.Event()
        end = drv.Event()
        times = []
        for _ in range(ITERATIONS):
            context.synchronize()
            start.record()
            func(*gpu_args, block=threads, grid=grid)
            end.record()
            context.synchronize()
            times.append(end.time_since(start))
        times = sorted(times)
        return numpy.mean(times[1:-1])
