from kernel_tuner.testing.mutation.mutant import HigherOrderMutant, Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder, TimeoutException
from kernel_tuner.testing.mutation.mutation_result import MutationResult
from kernel_tuner.testing.testing_test_case import TestCase

class MutationExecutor():
    def __init__(self, mutation_kernel_builder: TestingKernelBuilder, mutants: list[Mutant], test_cases: list[TestCase],
                  high_order_mutants: list[HigherOrderMutant] = []):
        self.mutants = mutants
        self.kernel_builder = mutation_kernel_builder
        self.test_cases = test_cases
        self.high_order_mutants = high_order_mutants
    
    def execute(self) -> MutationResult:
        self._execute_mutants()
        
        if len(self.high_order_mutants):
            self._execute_ho_mutants()
            return MutationResult(self.mutants, self.test_cases, self.high_order_mutants)
        
        return MutationResult(self.mutants, self.test_cases)
    
    def _execute_mutants(self) -> None:
        kernel = self.kernel_builder.build()
        for mutant in self.mutants:
            #Ingore the killed and error mutants
            if (mutant.status in [MutantStatus.KILLED, MutantStatus.PERF_KILLED, MutantStatus.COMPILE_ERROR]):
                continue
            
            #Start a new mutant
            if (mutant.status == MutantStatus.CREATED):
                mutant.status = MutantStatus.PENDING

            print("Start running mutant", mutant.id)
            #Mutate the kernel code
            original_kernel_string = self.kernel_builder.kernel_string
            mutated_kernel_string = MutationExecutor.mutate(original_kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            try:
                kernel.update_kernel(kernel_string = mutated_kernel_string)
            except:
                print("Mutant", mutant.id, "contains compilation error(s)")
                mutant.updateResult(MutantStatus.COMPILE_ERROR)
                continue

            for test_case in self.test_cases:
                kernel.update_gpu_args(test_case.input)
                try:
                    result = kernel.execute()
                    if not kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                        print("Mutant", mutant.id, "killed by test case: ", test_case.id)
                        mutant.updateResult(MutantStatus.KILLED, test_case.id)
                except TimeoutException:
                    print("Mutant", mutant.id, "timeout with test case: ", test_case.id)
                    mutant.updateResult(MutantStatus.TIMEOUT)
                    continue
                except Exception as e: 
                    print("Mutant", mutant.id, "has runtime error with test case: ", test_case.id)
                    print(e)
                    mutant.updateResult(MutantStatus.RUNTIME_ERROR)
                    continue
    
            # Not killed by any test cases: mutant survived
            if mutant.status == MutantStatus.PENDING:
                print("Mutant", mutant.id, "survived")
                mutant.updateResult(MutantStatus.SURVIVED)

    def _execute_ho_mutants(self) -> None:
        kernel = self.kernel_builder.build()
        for ho_mutant in self.high_order_mutants:
            #Ingore the killed and error mutants
            if (ho_mutant.status in [MutantStatus.KILLED, MutantStatus.PERF_KILLED, MutantStatus.COMPILE_ERROR]):
                continue

            #Start a new mutant
            if (ho_mutant.status == MutantStatus.CREATED):
                #If the mutant contains compile error, then the ho_mutant will also have
                skip_ho_mutant = False
                for mutant in ho_mutant.mutants:
                    if (mutant.status == MutantStatus.COMPILE_ERROR):
                        print("Skip high order mutant", ho_mutant.id, "because it contains error or timeout mutant(s)")
                        skip_ho_mutant = True
                        break
                
                if skip_ho_mutant:
                    ho_mutant.updateResult(MutantStatus.COMPILE_ERROR)
                    continue
                else:
                    ho_mutant.status = MutantStatus.PENDING

            print("Start running higher order mutant", ho_mutant.id)
            #Mutate the kenel code
            mutated_kernel_string = self.kernel_builder.kernel_string
            for mutant in ho_mutant.mutants:
                mutated_kernel_string = MutationExecutor.mutate(mutated_kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            
            try:
                kernel.update_kernel(kernel_string=mutated_kernel_string)
            except:
                print("Mutant", mutant.id, "contains compilation error(s)")
                ho_mutant.updateResult(MutantStatus.COMPILE_ERROR)
                continue
            for test_case in self.test_cases:
                kernel.update_gpu_args(test_case.input)
                try:
                    result = kernel.execute()
                    if not kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                        #Killed
                        print("Higher Order Mutant", ho_mutant.id, "killed by test case: ", test_case.id)
                        ho_mutant.updateResult(MutantStatus.KILLED, test_case.id)
                except TimeoutException:
                    print("Higher order mutant", ho_mutant.id, "timeout with test case: ", test_case.id)
                    mutant.updateResult(MutantStatus.TIMEOUT)
                    continue
                except Exception as e:
                    print("Higher order mutant", ho_mutant.id, "has runtime error with test case: ", test_case.id)
                    print(e)
                    mutant.updateResult(MutantStatus.RUNTIME_ERROR)
                    continue

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