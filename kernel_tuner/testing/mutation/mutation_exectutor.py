from kernel_tuner.core import _default_verify_function
from kernel_tuner.testing.mutation.mutant import Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder
from kernel_tuner.testing.mutation.mutation_result import MutationResult
from kernel_tuner.testing.test_case import TestCase

class MutationExecutor():
    def __init__(self, mutation_kernel_builder: TestingKernelBuilder, mutants: list[Mutant], test_cases: list[TestCase]):
        self.mutants = mutants
        self.kernel_builder = mutation_kernel_builder
        self.test_cases = test_cases
    
    def execute(self) -> MutationResult:
        original_kernel_string = self.kernel_builder.kernel_string
        kernel = self.kernel_builder.build()
        for mutant in self.mutants:
            mutant.status = MutantStatus.PENDING

            #Mutate the kenel code
            mutated_kernel_string = MutationExecutor.mutate(original_kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            kernel.update_kernel(mutated_kernel_string)

            for test_case in self.test_cases:
                kernel.update_gpu_args(test_case.input)
                result = kernel.execute()
                if not kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                    #Killed
                    mutant.updateResult(MutantStatus.KILLED, test_case.id)
            
            # Not killed by any test cases: mutant survived
            if mutant.status == MutantStatus.PENDING:
                mutant.updateResult(MutantStatus.SURVIVED)
        return MutationResult(self.mutants, self.test_cases)
            
    def mutate(kernel_string: str, start: MutantPosition, end: MutantPosition, replacement: str) -> str:
        string_lines = kernel_string.splitlines()

        start_line_idx = start.line - 1
        start_col_idx = start.column -1
        end_line_idx = end.line - 1
        end_col_idx = end.column - 1

        if start_line_idx == end_line_idx:
            #Only influence single line of code
            line_idx = start_line_idx
            code = string_lines[line_idx]
            code = code.replace(code[start_col_idx: end_col_idx], replacement)
            string_lines[line_idx] = code
        else:
            #TODO: Influence multiple lines of code
            raise RuntimeError("Unsupport method")
            #for line_idx in range(start_line_idx, end_line_idx):

        mutated_kernel_string = "\n".join(string_lines)

        return mutated_kernel_string