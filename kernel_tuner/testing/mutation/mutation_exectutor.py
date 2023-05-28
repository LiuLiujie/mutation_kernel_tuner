import itertools
from kernel_tuner.testing.mutation.mutant import HigherOrderMutant, Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder
from kernel_tuner.testing.mutation.mutation_result import MutationResult
from kernel_tuner.testing.test_case import TestCase

class MutationExecutor():
    def __init__(self, mutation_kernel_builder: TestingKernelBuilder, mutants: list[Mutant], test_cases: list[TestCase],
                  high_order_mutants: list[HigherOrderMutant] = []):
        self.mutants = mutants
        self.kernel_builder = mutation_kernel_builder
        self.test_cases = test_cases
        self.high_order_mutants = high_order_mutants
    
    def execute(self) -> MutationResult:
        self.__execute_mutants()
        
        if len(self.high_order_mutants):
            self.__execute_ho_mutants()
            return MutationResult(self.mutants, self.test_cases, self.high_order_mutants)
        
        return MutationResult(self.mutants, self.test_cases)
    
    def __execute_mutants(self) -> None:
        original_kernel_string = self.kernel_builder.kernel_string
        kernel = self.kernel_builder.build()
        for mutant in self.mutants:
            #Ingore the killed and error mutants
            if (mutant.status in [MutantStatus.KILLED, MutantStatus.PERF_KILLED, MutantStatus.COMPILE_ERROR]):
                continue
            
            #Start a new mutant
            if (mutant.status == MutantStatus.CREATED):
                mutant.status = MutantStatus.PENDING

            #Mutate the kenel code
            mutated_kernel_string = MutationExecutor.mutate(original_kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            try:
                kernel.update_kernel(mutated_kernel_string)
            except Exception:
                mutant.updateResult(MutantStatus.COMPILE_ERROR)
                continue

            for test_case in self.test_cases:
                kernel.update_gpu_args(test_case.input)
                result = kernel.execute()
                if not kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                    #Killed
                    mutant.updateResult(MutantStatus.KILLED, test_case.id)
            
            # Not killed by any test cases: mutant survived
            if mutant.status == MutantStatus.PENDING:
                mutant.updateResult(MutantStatus.SURVIVED)

    def __execute_ho_mutants(self, ) -> None:
        kernel_string = self.kernel_builder.kernel_string
        kernel = self.kernel_builder.build()
        for ho_mutant in self.high_order_mutants:

            #Ingore the killed and error mutants
            if (ho_mutant.status in [MutantStatus.KILLED, MutantStatus.PERF_KILLED, MutantStatus.COMPILE_ERROR]):
                continue

            #Start a new mutant
            if (ho_mutant.status == MutantStatus.CREATED):
                #If the mutant contains compile error, then the ho_mutant will also have
                for mutant in ho_mutant.mutants:
                    if (mutant.status == MutantStatus.COMPILE_ERROR):
                        ho_mutant.status == MutantStatus.COMPILE_ERROR
                
                ho_mutant.status = MutantStatus.PENDING

            #Mutate the kenel code
            for mutant in ho_mutant.mutants:
                kernel_string = MutationExecutor.mutate(kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            
            try:
                kernel.update_kernel(kernel_string)
            except Exception:
                ho_mutant.updateResult(MutantStatus.COMPILE_ERROR)
                continue

            for test_case in self.test_cases:
                kernel.update_gpu_args(test_case.input)
                result = kernel.execute()
                if not kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                    #Killed
                    ho_mutant.updateResult(MutantStatus.KILLED, test_case.id)
            
            # Not killed by any test cases: mutant survived
            if ho_mutant.status == MutantStatus.PENDING:
                ho_mutant.updateResult(MutantStatus.SURVIVED)
            
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