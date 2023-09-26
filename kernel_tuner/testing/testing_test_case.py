import numpy as np

from kernel_tuner import util

try:
    import cupy as cp
except ImportError:
    cp = np

try:
    import torch
except ImportError:
    torch = util.TorchPlaceHolder()

class TestCase():
    def __init__(self, id: int, input: list, output: list, problem_size: int, verify = None, atol = 1e-6) -> None:
        self.id = id
        self.input = input
        self.output = output
        self.problem_size = problem_size
        self.verify = verify
        self.atol = atol
        self.passed = None
        self.description = ""

    def toJSON(self, without_input_output=False) -> str:
        if without_input_output:
            return {
            "id": self.id,
            "problem_size": self.problem_size,
            "test_passed": self.passed
            }
        return {
            "id": self.id,
            "problem_size": self.problem_size,
            "input": self.input,
            "expected_output": self.output,
            "test_passed": self.passed
        }
    
    def verify_result(self, result) -> (bool, str):
        if self.verify:
            correct = self.verify(self.output, result, atol=self.atol)
            msg = "The kernel output fails the customized verification function"
        else:
            correct, msg = _default_testing_verification_function(result, self.output, self.atol)
        if correct:
            return True, "Passed"
        else:
            return False, msg
    
    def test_fail(self, description = None):
        self.passed = False
        self.description = description
        return self

    def test_pass(self):
        self.passed = True
        return self

def verification(result, test_case: TestCase) -> TestCase:
    correct, msg = test_case.verify_result(result)
    if correct:
        return test_case.test_pass()
    else:
        return test_case.test_fail(msg)

def _default_testing_verification_function(result, answer, atol) -> (bool, str):

    #first check if the length is the same
    if len(result) != len(answer):
        return False, "length of expected result is not the same length as the kernel output"
    
    #for each element in the argument list, check if the types match
    for i, arg in enumerate(result):
        if answer[i] is not None:    #skip None elements in the answer list
            if isinstance(answer[i], (np.ndarray, cp.ndarray)) and isinstance(arg, (np.ndarray, cp.ndarray)):
                if answer[i].dtype != arg.dtype:
                    return False, (f"Element {i} of the expected results list is not of the same dtype as the kernel output: " + str(answer[i].dtype) +
                                    " != " + str(arg.dtype) + ".")
                if answer[i].size != arg.size:
                    return False, (f"Element {i} of the expected results list has a size different from " + "the kernel argument: " + str(answer[i].size) +
                                    " != " + str(arg.size) + ".")
            elif isinstance(answer[i], torch.Tensor) and isinstance(arg, torch.Tensor):
                if answer[i].dtype != arg.dtype:
                    return False, (f"Element {i} of the expected results list is not of the same dtype as the kernel output: " + str(answer[i].dtype) +
                                    " != " + str(arg.dtype) + ".")
                if answer[i].size() != arg.size():
                    return False, (f"Element {i} of the expected results list has a size different from " + "the kernel argument: " + str(answer[i].size) +
                                    " != " + str(arg.size) + ".")

            elif isinstance(answer[i], np.number) and isinstance(arg, np.number):
                if answer[i].dtype != arg.dtype:
                    return False, (f"Element {i} of the expected results list is not the same as the kernel output: " + str(answer[i].dtype) + " != " +
                                    str(arg.dtype) + ".")
            else:
                #either answer[i] and argument have different types or answer[i] is not a numpy type
                if not isinstance(answer[i], (np.ndarray, cp.ndarray, torch.Tensor)) or not isinstance(answer[i], np.number):
                    return False, (f"Element {i} of expected results list is not a numpy/cupy ndarray, torch Tensor or numpy scalar.")
                else:
                    return False, (f"Element {i} of expected results list and kernel arguments have different types.")

    def _ravel(a):
        if hasattr(a, 'ravel') and len(a.shape) > 1:
            return a.ravel()
        return a

    def _flatten(a):
        if hasattr(a, 'flatten'):
            return a.flatten()
        return a

    for i, expected in enumerate(answer):
        if expected is not None:
            result = _ravel(result[i])
            expected = _flatten(expected)
            if any([isinstance(array, cp.ndarray) for array in [expected, result]]):
                output_test = cp.allclose(expected, result, atol=atol)
            elif isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor):
                output_test = torch.allclose(expected, result, atol=atol)
            else:
                output_test = np.allclose(expected, result, atol=atol)
            if not output_test:
                return False, (f"Element {i} of expected results is not the same as the kernel output")
        
    return True, ""

def filter_by_problem_size(test_cases: list[TestCase]):
    problem_sizes = []
    filted_test_cases_list=[]
    for case in test_cases:
        if case.problem_size not in problem_sizes:
            problem_sizes.append(case.problem_size)
            filted_test_cases_list.append([case])
        else:
            idx = problem_sizes.index(case.problem_size)
            filted_test_cases_list[idx].append(case)

    return problem_sizes, filted_test_cases_list