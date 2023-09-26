import numpy as np
import pytest
from kernel_tuner.testing.mutation.mutant import Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.mutation.mutation_exectutor import MutationExecutor
from kernel_tuner.testing.mutation.mutation_operator import MutationOperator

from kernel_tuner.testing.testing_test_case import TestCase
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder

try:
    # CUDA required
    import pycuda.driver as drv
    drv.init()
except Exception:
    pytest.skip("PyCuda not installed or no CUDA device detected")

def test_deadloop():
    kernel_string = """
    #include <stdio.h>

    __global__ void itrpt(volatile bool *interrupt)
    {
        printf("Device: %d\\n", *interrupt);
        while (true) {
            if (*interrupt){
                printf("Kernel interrupted, device: %d\\n", *interrupt);
                return;
            }
        }
    }
    """

    kernel_name = "itrpt"

    test_params=dict()
    test_params["block_size_x"] = 1

    test_cases = []
    flag = np.array([False]).astype(bool)
    input = [flag]
    output = [None]
    test_cases.append(TestCase(id=0, input=input, output = output, problem_size = 1))
    builder = TestingKernelBuilder(kernel_name, kernel_string, 1, test_cases[0].input, test_cases[0].output, test_params) \
                    .enable_testing_timeout(5)
    mutant = Mutant(id=0,
                    operator=MutationOperator("fake_operator", "",""),
                    start= MutantPosition(line=1, column=1),
                    end=MutantPosition(line = 1, column=2))
    executor = MutationExecutor(builder, [mutant], test_cases)
    executor.execute()
    assert mutant.status == MutantStatus.TIMEOUT
