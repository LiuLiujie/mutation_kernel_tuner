import numpy as np
import pytest


from kernel_tuner import core
from kernel_tuner.testing.mutation.mutation_analyzer import MutationAnalyzer
from kernel_tuner.testing.mutation.mutation_exectutor import MutationExecutor
from kernel_tuner.testing.testing_kernel import TestingKernel, TestingKernelBuilder
from kernel_tuner.testing.mutation.mutation_operator import MutationOperator
from test.context import skip_backend

backends = ["cuda", "cupy"]
operators = MutationOperator('math_replacement', "+", "-", ignores=["++", "+="]),

@pytest.fixture()
def test_kernel():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    n = np.int32(100)
    a, b, c = np.random.random((3, n)).astype(np.float32)
    params = {"block_size_x": 384}
    return "vector_add", kernel_string, n, [c, a, b, n], [a+b, None, None, None], params

@pytest.mark.parametrize("backend", backends)
def test_mutation_kernel_builder(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    builder = TestingKernelBuilder(kernel_name, kernel_string, n, args, expected_output, params)
    kernel = builder.build()
    assert isinstance(kernel, TestingKernel)

@pytest.mark.parametrize("backend", backends)
def test_mutation_kernel(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source,operators)
    mutants = analyzer.analyze()

    builder = TestingKernelBuilder(kernel_name, kernel_string, n, args, expected_output, params)
    kernel = builder.build()

    result = kernel.execute()
    assert np.allclose(result[0], expected_output[0])

    # execute the mutant
    for mutant in mutants:
        mutated_kernel_string = MutationExecutor.mutate(kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
        kernel.update_kernel(mutated_kernel_string)
        result = kernel.execute()
        assert not np.allclose(result[0], expected_output[0])
    
    