import itertools
import numpy as np
import pytest
import json

from kernel_tuner import core
from kernel_tuner.testing.mutation.mutant import HigherOrderMutant, MutantStatus
from kernel_tuner.testing.mutation.mutation_analyzer import MutationAnalyzer
from kernel_tuner.testing.mutation.mutation_exectutor import MutationExecutor
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder
from kernel_tuner.testing.mutation.mutation_operator import MutationOperator
from kernel_tuner.testing.testing_test_case import TestCase
from test.context import skip_backend
backends = ["cuda", "cupy"]

operators = [MutationOperator('math_replacement', "+", "-", ignores=["++", "+="])]

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
def test_analyzer_analyze(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel
    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)

    analyzer = MutationAnalyzer(kernel_source, operators)
    mutants = analyzer.analyze()
    assert len(mutants) == 2
    list = [mutant.toJSON() for mutant in mutants]
    js= json.dumps(list)
    with open("examples/mutation/mutants.json", "w") as fo:
        fo.write(js)

@pytest.mark.parametrize("backend", backends)
def test_exectutor_mutate(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel
    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, operators)
    mutants = analyzer.analyze()

    string_list = []
    kernel_string = kernel_source.get_kernel_string(0)
    for mutant in mutants:
        mutated_kernel_string = MutationExecutor.mutate(kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
        string_list.append(mutated_kernel_string)
    with open("examples/mutation/mutated_demo_kernel.txt", "w") as fo:
        fo.write("\n".join(string_list))

@pytest.mark.parametrize("backend", backends)
def test_executor_execute(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, operators)
    mutants = analyzer.analyze()

    builder = TestingKernelBuilder(kernel_name, kernel_string, n, args, expected_output, params)

    test_case = TestCase(0, args, expected_output, n)

    executor = MutationExecutor(builder, mutants, [test_case])
    mutation_result = executor.execute()

    for mutant in mutation_result.mutants:
        assert mutant.status == MutantStatus.KILLED

    js = mutation_result.exportJSONStr()
    with open("examples/mutation/mutation_testing_result.json", "w") as fo:
        fo.write(js)

@pytest.mark.parametrize("backend", backends)
def test_ho_executor_execute(test_kernel, backend):
    skip_backend(backend)
    kernel_name, kernel_string, n, args, expected_output, params = test_kernel

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang=backend)
    analyzer = MutationAnalyzer(kernel_source, operators)
    mutants = analyzer.analyze()

    builder = TestingKernelBuilder(kernel_name, kernel_string, n, args, expected_output, params)

    test_case = TestCase(0, args, expected_output, n)

    mutation_order = 2
    comb_mutants_list = itertools.combinations(mutants, mutation_order)
    ho_mutants = [HigherOrderMutant(id = "Combi-"+("-".join([str(mutant.id) for mutant in comb_mutants])),
                                    mutants = comb_mutants,
                                    mutation_order = mutation_order) for comb_mutants in comb_mutants_list]

    executor = MutationExecutor(builder, mutants, [test_case], ho_mutants)
    mutation_result = executor.execute()

    for mutant in mutation_result.mutants:
        assert mutant.status == MutantStatus.KILLED

    for ho_mutant in mutation_result.higher_order_mutants:
        assert ho_mutant.status == MutantStatus.KILLED

    js = mutation_result.exportJSONStr()
    with open("examples/mutation/high_order_mutation_testing_result.json", "w") as fo:
        fo.write(js)