import numpy as np
import pytest
from kernel_tuner.testing.mutation.mutant import Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.mutation.mutation_exectutor import MutationExecutor
from kernel_tuner.testing.mutation.mutation_operator import MutationOperator

from kernel_tuner.testing.test_case import TestCase
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder

try:
    # CUDA required
    import pycuda.driver as drv
    drv.init()
except Exception:
    pytest.skip("PyCuda not installed or no CUDA device detected")

def test_deadloop():
    kernel_string = r"""
    #include <cstdint>
    __global__ void histogram(uint32_t *pixels, uint32_t *histogram,
                              uint32_t num_colors, uint32_t num_pixels,
                              volatile uint8_t *interrupt) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      int gsize = gridDim.x * blockDim.x;
      int i=0;
      while (true) {
          i = i+1;
      }
    }
    """

    kernel_name = "histogram"

    data=[
      [
          8,
          [0, 1, 2, 3, 0, 1, 2, 3],
          4,
          [2, 2, 2, 2]
      ]
    ]
    test_cases = []
    test_params=dict()
    test_params["block_size_x"] = 32
    for p_num, pixels, c_num, exp_res in data:
        size = p_num
        p_num = np.uint32(p_num)
        pixels = np.array(pixels).astype(np.uint32)
        histogram = np.zeros(c_num).astype(np.uint32)
        c_num = np.uint32(c_num)
        exp_res = np.array(exp_res).astype(np.uint32)
    
        input = [pixels, histogram, c_num, p_num]
        output = [None, exp_res, None, None]

        test_cases.append(TestCase(id=0, input=input, output = output, problem_size = size))

        #mut_kernel(kernel_name, kernel_string, test_cases, test_params)
        builder = TestingKernelBuilder(kernel_name, kernel_string, size, test_cases[0].input, test_cases[0].output, test_params)
        mutant = Mutant(id=16,
                        operator=MutationOperator("math_replacement", "","/"),
                        start= MutantPosition(line=7, column=29),
                        end=MutantPosition(line = 7, column=30))
        executor = MutationExecutor(builder, [mutant], test_cases, timeout_second=10)
        executor.execute()
        assert mutant.status == MutantStatus.TIMEOUT


    