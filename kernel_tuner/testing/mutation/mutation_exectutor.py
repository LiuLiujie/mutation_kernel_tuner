from copy import deepcopy
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
   pass
import os
import signal
import threading

from kernel_tuner.testing.mutation.mutant import HigherOrderMutant, Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder, TimeoutException
from kernel_tuner.testing.mutation.mutation_result import MutationResult
from kernel_tuner.testing.testing_test_case import TestCase

def timeout_handler(signum, frame):
        print("Runner timeout")
        raise TimeoutException

class MTRunnerKiller(threading.Thread):
    """A killer to terminate the mutation testing runner"""

    def __init__(self, target_process, repeat_sec=2.0):
        super(MTRunnerKiller,self).__init__()
        self.target_process = target_process
        self.repeat_sec = repeat_sec
        self.daemon = True

    def run(self):
        while self.target_process.is_alive():
            os.kill(self.target_process.pid, signal.SIGKILL)
            self.target_process.join(self.repeat_sec)

class MTRunner(multiprocessing.Process):
    """A mutation testing runner
       When a GPU kernel throws an runtime exception, the runner will be killed to 
       force the GPU driver to release the corresponding GPU context. 
    """

    def __init__(self, kernel_builder: TestingKernelBuilder, mutants: list[Mutant or HigherOrderMutant],
                 test_cases: list[TestCase], result_queue, current_mutant_idx):
        
        super().__init__()
        self.kernel_builder = kernel_builder
        self.mutants = mutants
        self.test_cases = test_cases
        self.result_queue = result_queue
        self.current_mutant_idx = current_mutant_idx
    
    def run(self):
        if not len(self.mutants):
            self.result_queue.put(self.mutants)
            return
        self.mutants_runner()

    def mutants_runner(self):
        kernel = self.kernel_builder.build()
        for idx, mutant in enumerate(self.mutants):
            if (idx < self.current_mutant_idx):
                continue
            
            #Start a mutant
            self.current_mutant_idx = idx
            mutant.status = MutantStatus.PENDING
            print("Mutant", mutant.id, "starts")
            
            if self.kernel_builder.verbose:
                print("Mutant", mutant.id, "operator", mutant.operator.name,
                      "replacement", mutant.operator.replacement,
                      "from line", mutant.start.line, "column", mutant.start.column,
                      "to line", mutant.end.line, "column", mutant.end.column)
                
            #Mutate the kernel code
            mutated_kernel_string = deepcopy(self.kernel_builder.kernel_string)
            if isinstance(mutant, Mutant):
                mutated_kernel_string = MutationExecutor.mutate(mutated_kernel_string, mutant.start,
                                                                mutant.end, mutant.operator.replacement)
            elif isinstance(mutant, HigherOrderMutant):
                for sub_mutant in mutant.mutants:
                    mutated_kernel_string = MutationExecutor.mutate(mutated_kernel_string, sub_mutant.start,
                                                                    sub_mutant.end, sub_mutant.operator.replacement)
            
            try:
                kernel.update_kernel(kernel_string = mutated_kernel_string)
            except:
                print("Mutant", mutant.id, "contains compilation error(s)")
                mutant.updateResult(MutantStatus.COMPILE_ERROR)
                self.result_queue.put(mutant)
                break

            for test_case in self.test_cases:
                print("Mutant", mutant.id, "executes with test case", test_case.id)
                kernel.update_gpu_args(test_case.input)
                kernel.update_expected_output(test_case.output)
                try:
                    result = kernel.execute()
                    passed, _ = test_case.verify_result(result)
                    if not passed:
                        print("Mutant", mutant.id, "killed by test case:", test_case.id)
                        mutant.updateResult(MutantStatus.KILLED, test_case.id)
                except TimeoutException:
                    print("Mutant", mutant.id, "timeout with test case:", test_case.id)
                    mutant.updateResult(MutantStatus.TIMEOUT)
                    break
                except Exception as e: 
                    #The subprocess need to return so that the corresponding GPU error context can also be released
                    print("Mutant", mutant.id, "has runtime error with test case: ", test_case.id)
                    if kernel.verbose:
                        print(e)
                    mutant.updateResult(MutantStatus.RUNTIME_ERROR)
                    self.result_queue.put(mutant)
                    return

            # Not killed by any test cases: mutant survived
            if mutant.status in [MutantStatus.PENDING, MutantStatus.SURVIVED]:
                print("Mutant", mutant.id, "survived")
                mutant.updateResult(MutantStatus.SURVIVED)
            self.result_queue.put(mutant)

    def terminate(self, repeat_sec=2.0):
        if self.is_alive() is False:
            return True
        killer = MTRunnerKiller(self, repeat_sec=repeat_sec)
        killer.start()

class MutationExecutor():
    def __init__(self, mutation_kernel_builder: TestingKernelBuilder, mutants: list[Mutant], test_cases: list[TestCase],
                  high_order_mutants: list[HigherOrderMutant] = []):
        self.mutants = mutants
        self.kernel_builder = mutation_kernel_builder
        self.test_cases = test_cases
        self.high_order_mutants = high_order_mutants
        self.golbal_timeout = mutation_kernel_builder.global_timeout_second
    
    def execute(self):
        
        kernel_builder = deepcopy(self.kernel_builder)
        current_mutant_idx = 0
        total_mutant = len(self.mutants)
        print("Total mutants:", total_mutant)
        results = []
        result_queue = multiprocessing.Manager().Queue()

        while(current_mutant_idx < total_mutant):
            process = MTRunner(kernel_builder, self.mutants, self.test_cases, result_queue,
                                current_mutant_idx)
            process.start()
            try:
                while(current_mutant_idx < total_mutant):
                    result = result_queue.get(timeout=self.golbal_timeout)
                    results.append(result)
                    print("Finish getting results from:", result.id, "with mutant index", current_mutant_idx)
                    current_mutant_idx+=1
                    if result.status in [MutantStatus.RUNTIME_ERROR, MutantStatus.COMPILE_ERROR]:
                        print("Restarting the mutant runner")
                        break
            except:
                print("Global timeout for mutant:", current_mutant_idx)
                #TODO: mark mutant timeout
                process.terminate()
                current_mutant_idx += 1
            process.join(timeout=self.golbal_timeout)
            if process.is_alive():
                print("Golbal timeout for mutant runner")
                process.terminate()
                print("Process terminated")
        
        self.mutants = results
                
        if len(self.high_order_mutants):
            print("Total HO mutants:", len(self.high_order_mutants))
            current_mutant_idx = 0
            while(True):
                #Must use Manager (shared memory) here because the size of high_order_mutants
                # is too big for a normal multiprocessing queue and will block the runner process.
                result_queue = multiprocessing.Manager().Queue()
                process = MTRunner(kernel_builder, self.high_order_mutants, self.test_cases, result_queue,
                                    current_mutant_idx)
                process.start()
                try:
                    self.high_order_mutants = result_queue.get(timeout=3600)
                except:
                    print("Golbal timeout for fetching results")
                    process.terminate()
                    break
                process.join(timeout=5)
                if process.is_alive():
                    print("Golbal timeout")
                    process.terminate()
                    print("Process terminated")
                
                if current_mutant_idx != -1:
                    current_mutant_idx += 1
                else:
                    break
                    
        return self.mutants, self.high_order_mutants

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