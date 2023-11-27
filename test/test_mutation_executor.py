import numpy as np
import pytest
from kernel_tuner import core
from kernel_tuner.testing.mutation.mutation_analyzer import MutationAnalyzer
from kernel_tuner.testing.mutation.mutation_exectutor import MutationExecutor
from kernel_tuner.testing.mutation.mutation_operator import *

from test.context import skip_backend
backends = ["cuda", "cupy"]

@pytest.fixture()
def test_kernel():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<<1024, 256>>>();
        cudaDeviceSynchronize();
    }
    """
    n = np.int32(100)
    a, b, c = np.random.random((3, n)).astype(np.float32)
    params = {"block_size_x": 384}
    return "vector_add", kernel_string, n, [c, a, b, n], [a+b, None, None, None], params

@pytest.mark.parametrize("backend", backends)
def test_allocation_swap(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, allocation_swap)
    mutants = analyzer.analyze()
    assert len(mutants) == 1
    mut1 = MutationExecutor.mutate(kernel_string, mutants[0].start, mutants[0].end, mutants[0].operator.replacement)
    assert mut1 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<< 256,1024>>>();
        cudaDeviceSynchronize();
    }
    """

@pytest.mark.parametrize("backend", backends)
def test_allocation_increase(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, allocation_increase)
    mutants = analyzer.analyze()
    assert len(mutants) == 2
    mut1 = MutationExecutor.mutate(kernel_string, mutants[0].start, mutants[0].end, mutants[0].operator.replacement)
    assert mut1 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<<1024+1, 256>>>();
        cudaDeviceSynchronize();
    }
    """
    mut2 = MutationExecutor.mutate(kernel_string, mutants[1].start, mutants[1].end, mutants[1].operator.replacement)
    assert mut2 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<<1024, 256+1>>>();
        cudaDeviceSynchronize();
    }
    """

@pytest.mark.parametrize("backend", backends)
def test_allocation_decrease(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, allocation_decrease)
    mutants = analyzer.analyze()
    assert len(mutants) == 2
    mut1 = MutationExecutor.mutate(kernel_string, mutants[0].start, mutants[0].end, mutants[0].operator.replacement)
    assert mut1 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<<1024-1, 256>>>();
        cudaDeviceSynchronize();
    }
    """
    mut2 = MutationExecutor.mutate(kernel_string, mutants[1].start, mutants[1].end, mutants[1].operator.replacement)
    assert mut2 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<<1024, 256-1>>>();
        cudaDeviceSynchronize();
    }
    """
        
@pytest.mark.parametrize("backend", backends)
def test_atomic_add_sub_removal(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, atom_removal)
    mutants = analyzer.analyze()
    assert len(mutants) == 1
    mut1 = MutationExecutor.mutate(kernel_string, mutants[0].start, mutants[0].end, mutants[0].operator.replacement)
    assert mut1 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        a[i]+= 1;
        vector_add<<<1024, 256>>>();
        cudaDeviceSynchronize();
    }
    """

@pytest.mark.parametrize("backend", backends)
def test_sync_child_removal(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, sync_child_removal)
    mutants = analyzer.analyze()
    assert len(mutants) == 1
    mut1 = MutationExecutor.mutate(kernel_string, mutants[0].start, mutants[0].end, mutants[0].operator.replacement)
    print(mut1)
    assert mut1 == """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n && i!=n) {
            c[i] = a[i] + b[i];
            c[i]++;
            c[i]--;
            c[i] += a[i];
            c[i] -= a[i];
            c[i] *= a[i];
            c[i] /= a[i];
        }
        __syncthreads();
        atomicAdd(&a[i], 1);
        vector_add<<<1024, 256>>>();
        //cudaDeviceSynchronize();
    }
    """