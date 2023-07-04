import numpy
from kernel_tuner import run_kernel, tune_kernel, mut_kernel, test_kernel
import pytest
from kernel_tuner.testing.mutation.mutant import MutantStatus

from kernel_tuner.testing.testing_test_case import TestCase

try:
    # CUDA required
    import pycuda.driver as drv
    drv.init()
except Exception:
    pytest.skip("PyCuda not installed or no CUDA device detected")

def test_tune_kernel():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 1<<20
    problem_size = (size, 1)

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    answer = [a+b, None, None, None]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    tune_kernel("vector_add", kernel_string, problem_size, args, tune_params, answer = answer,
                 objective="time", objective_higher_is_better=False)


def test_run_kernel():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 1<<20
    problem_size = (size, 1)

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    params = {"block_size_x": 512}

    answer = run_kernel("vector_add", kernel_string, problem_size, args, params)

    assert numpy.allclose(answer[0], a+b, atol=1e-8)

def test_test_kernel():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    n1 = numpy.int32(100)
    a1, b1, c1 = numpy.random.random((3, n1)).astype(numpy.float32)

    n2 = numpy.int32(1000)
    a2, b2, c2 = numpy.random.random((3, n2)).astype(numpy.float32)

    n3 = numpy.int32(1000)
    a3, b3, c3 = numpy.random.random((3, n3)).astype(numpy.float32)

    test_params = dict()
    test_params["block_size_x"] = [128+64*i for i in range(4)]
    test_case_1 = TestCase(0, [c1, a1, b1, n1], [a1+b1, None, None, None], 100)
    test_case_2 = TestCase(1, [c2, a2, b2, n2], [a2+b2, None, None, None], 1000)
    test_case_3 = TestCase(2, [c3, a3, b3, n3], [a3+b3, None, None, None], 1000)
    testing_result = test_kernel("vector_add", kernel_string, [test_case_1, test_case_2, test_case_3], test_params)
    for test_case in testing_result.test_cases:
        assert test_case.passed

def test_mut_kernel():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    n1 = numpy.int32(100)
    a1, b1, c1 = numpy.random.random((3, n1)).astype(numpy.float32)

    n2 = numpy.int32(1000)
    a2, b2, c2 = numpy.random.random((3, n2)).astype(numpy.float32)

    test_params = dict()
    test_params["block_size_x"] = [128+64*i for i in range(4)]
    test_case_1 = TestCase(0, [c1, a1, b1, n1], [a1+b1, None, None, None], 100)
    test_case_2 = TestCase(0, [c2, a2, b2, n2], [a2+b2, None, None, None], 1000)
    mutation_result = mut_kernel("vector_add", kernel_string, [test_case_1, test_case_2], test_params)
    assert test_case_1.passed and test_case_2.passed
    for mutant in mutation_result.mutants:
        assert mutant.status != MutantStatus.PENDING
    