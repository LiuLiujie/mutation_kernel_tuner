import numpy as np
import pytest
import json

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
    }
    """
    n = np.int32(100)
    a, b, c = np.random.random((3, n)).astype(np.float32)
    params = {"block_size_x": 384}
    return "vector_add", kernel_string, n, [c, a, b, n], [a+b, None, None, None], params

@pytest.mark.parametrize("backend", backends)
def test_conditional_boundary_replacement(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel
    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, conditional_boundary_replacement)
    mutants = analyzer.analyze()
    assert len(mutants) == 1 
    assert mutants[0].operator.replacement == "<="

@pytest.mark.parametrize("backend", backends)
def test_arithmetic_operator_replacement_shortcut(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, arithmetic_operator_replacement_shortcut)
    mutants = analyzer.analyze()
    assert len(mutants) == 2

@pytest.mark.parametrize("backend", backends)
def test_conditional_operator_replacement(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel
    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, conditional_operator_replacement)
    mutants = analyzer.analyze()
    assert len(mutants) == 1 
    assert mutants[0].operator.replacement == "||"

@pytest.mark.parametrize("backend", backends)
def test_math_replacement(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, math_replacement_operators)
    mutants = analyzer.analyze()
    # Include 3 compilation errors for '*' in parameter list, and 1 error in getting address
    assert len(mutants) == 7

@pytest.mark.parametrize("backend", backends)
def test_math_assignment_replacement_shortcut(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, math_assignment_replacement_shortcut)
    mutants = analyzer.analyze()
    assert len(mutants) == 4

@pytest.mark.parametrize("backend", backends)
def test_negate_conditional_replacement(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, negate_conditional_replacement)
    mutants = analyzer.analyze()
    assert len(mutants) == 2

@pytest.mark.parametrize("backend", backends)
def test_allocation_swap(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, allocation_swap)
    mutants = analyzer.analyze()
    assert len(mutants) == 1

@pytest.mark.parametrize("backend", backends)
def test_allocation_increase(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, allocation_increase)
    mutants = analyzer.analyze()
    assert len(mutants) == 2

@pytest.mark.parametrize("backend", backends)
def test_allocation_decrease(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, allocation_decrease)
    mutants = analyzer.analyze()
    assert len(mutants) == 2

@pytest.mark.parametrize("backend", backends)
def test_atomic_add_sub_removal(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, atom_removal)
    mutants = analyzer.analyze()
    assert len(mutants) == 1

@pytest.mark.parametrize("backend", backends)
def test_gpu_index_replacement(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, gpu_index_replacement)
    mutants = analyzer.analyze()
    assert len(mutants) == 2
    mut_string_1 = MutationExecutor.mutate(kernel_string, mutants[0].start,
                             mutants[0].end, mutants[0].operator.replacement)
    ref_mut_kernel_1 = kernel_string.replace("blockIdx.x", "threadIdx.x")
    assert mut_string_1 == ref_mut_kernel_1
    mut_string_2 = MutationExecutor.mutate(kernel_string, mutants[1].start,
                             mutants[1].end, mutants[1].operator.replacement)
    ref_mut_kernel_2 = kernel_string.replace("threadIdx.x", "blockIdx.x")
    assert mut_string_2 == ref_mut_kernel_2

@pytest.mark.parametrize("backend", backends)
def test_gpu_index_increment(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, gpu_index_increment)
    mutants = analyzer.analyze()
    assert len(mutants) == 2

@pytest.mark.parametrize("backend", backends)
def test_gpu_index_decrement(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, gpu_index_decrement)
    mutants = analyzer.analyze()
    assert len(mutants) == 2

@pytest.mark.parametrize("backend", backends)
def test_sync_removal(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, sync_removal)
    mutants = analyzer.analyze()
    assert len(mutants) == 1
