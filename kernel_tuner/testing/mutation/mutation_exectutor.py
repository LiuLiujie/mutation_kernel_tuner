from copy import deepcopy
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
   pass

import signal
import threading
import os

from kernel_tuner.testing.mutation.mutant import HigherOrderMutant, Mutant, MutantPosition, MutantStatus
from kernel_tuner.testing.testing_kernel import TestingKernelBuilder
from kernel_tuner.testing.mutation.mutation_result import MutationResult
from kernel_tuner.testing.test_case import TestCase

class TimeoutException(Exception):
    pass

class CompilationException(Exception):
    pass

class RuntimeException(Exception):
    pass

class UnknownException(Exception):
    pass

class MutantRunnerKiller(threading.Thread):
    """Separate thread to kill TerminableMutantRunner"""

    def __init__(self, target_process, exception_cls, repeat_sec=2.0):
        super(MutantRunnerKiller,self).__init__()
        self.target_process = target_process
        self.exception_cls = exception_cls
        self.repeat_sec = repeat_sec
        self.daemon = True

    def run(self):
        """loop raising exception incase it's caught hopefully this breaks us far out"""
        while self.target_process.is_alive():
            os.kill(self.target_process.pid, signal.SIGKILL)
            self.target_process.join(self.repeat_sec)

class TerminableMutantRunner(multiprocessing.Process):
    """A mutant runner that can be stopped by forcing an exception in the execution context"""

    def __init__(self, kernel_builder: TestingKernelBuilder, mutant: Mutant, test_cases: list[TestCase],
                  result_queue, exception_queue):
        super().__init__()
        self.kernel_builder = kernel_builder
        self.mutant = mutant
        self.test_cases = test_cases
        self.result_queue = result_queue
        self.exception_queue = exception_queue

    def mutant_runner(self):
        results = []
        exceptions = []
        try:
            try:
                kernel = self.kernel_builder.build()
            except Exception as e:
                print(e)
                raise CompilationException
            for test_case in self.test_cases:
                kernel.update_gpu_args(test_case.input)
                try:
                    res = kernel.execute()
                    results.append({'id': test_case.id , 'result': res})
                except Exception as e:
                    exceptions.append("RuntimeException")
        except CompilationException as e:
            exceptions.append("CompilationException")
        except Exception as e:
            print("Mutant", self.mutant.id, "unknown error")
            print(e)
            exceptions.append("UnknownException")
        finally:
            self.result_queue.put(results)
            self.exception_queue.put(exceptions)

    def terminate(self, exception_cls, repeat_sec=2.0):
        if self.is_alive() is False:
            return True
        killer = MutantRunnerKiller(self, exception_cls, repeat_sec=repeat_sec)
        killer.start()
    
    def run(self):
        self.mutant_runner()

class MutationExecutor():
    def __init__(self, mutation_kernel_builder: TestingKernelBuilder, mutants: list[Mutant], test_cases: list[TestCase],
                  high_order_mutants: list[HigherOrderMutant] = [], timeout_second: int = 30):
        self.mutants = mutants
        self.kernel_builder = mutation_kernel_builder
        self.test_cases = test_cases
        self.high_order_mutants = high_order_mutants
        self.timeout_second = timeout_second
    
    def execute(self) -> MutationResult:
        self._execute_mutants()
        
        if len(self.high_order_mutants):
            self._execute_ho_mutants()
            return MutationResult(self.mutants, self.test_cases, self.high_order_mutants)
        
        return MutationResult(self.mutants, self.test_cases)
    
    def _execute_mutants(self) -> None:
        original_kernel_string = self.kernel_builder.kernel_string
        for mutant in self.mutants:
            #Ingore the killed and error mutants
            if (mutant.status in [MutantStatus.KILLED, MutantStatus.PERF_KILLED, MutantStatus.COMPILE_ERROR]):
                continue
            
            #Start a new mutant
            if (mutant.status == MutantStatus.CREATED):
                mutant.status = MutantStatus.PENDING

            #Mutate the kernel code
            mutated_kernel_string = MutationExecutor.mutate(original_kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            
            result_queue = multiprocessing.Queue()
            exception_queue = multiprocessing.Queue()
            current_kernel_builder = deepcopy(self.kernel_builder)
            current_kernel_builder.kernel_string = mutated_kernel_string

            try:
                print("Start to execute mutant: ", mutant.id)
                process = TerminableMutantRunner(current_kernel_builder, mutant, self.test_cases, result_queue, exception_queue)
                process.start()
                process.join(timeout = self.timeout_second)

                if process.is_alive():
                    # The mutant timeout, force kill the GPU kernel and raise TimeoutException
                    process.terminate(exception_cls=TimeoutException, repeat_sec=2.0)
                    raise TimeoutException

                test_kernel = self.kernel_builder.build()
                try: 
                    #Fetch and check kernel exceptions
                    exceptions = exception_queue.get(False)
                    results = result_queue.get(False)
                except Exception:
                    print("Fail to fetch result from kernel, ingore this mutant")
                    raise UnknownException
                
                #Check the exceptions
                if len(exceptions) > 0:
                    if "CompilationException" in exceptions:
                        raise CompilationException
                    elif "UnknownException" in exceptions:
                        raise UnknownException
                    elif "RuntimeException" in exceptions and exceptions.count("RuntimeException") == len(self.test_cases):
                        # All test cases will cause runtime exception
                        raise RuntimeException

                #Verify the result
                for result in results:
                    #Get the corresponding test case
                    filter_test_cases = [tc for tc in self.test_cases if tc.id == result['id']]
                    if len(filter_test_cases) == 1:
                       test_case = filter_test_cases[0]
                       if not test_kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                           #Killed
                            print("Mutant", mutant.id, "killed by test case: ", test_case.id)
                            mutant.updateResult(MutantStatus.KILLED, test_case.id)

            except CompilationException:
                print("Mutant", mutant.id, "has compilation error")
                mutant.updateResult(MutantStatus.COMPILE_ERROR)
            except UnknownException:
                print("Mutant", mutant.id, "has unknown error, ignore the mutant")
                mutant.updateResult(MutantStatus.IGNORE)
            except RuntimeException:
                print("Mutant", mutant.id, "has runtime error")
                mutant.updateResult(MutantStatus.RUNTIME_ERROR)
            except TimeoutException:
                print("Mutant", mutant.id, "timeout")
                mutant.updateResult(MutantStatus.TIMEOUT)
            else:
                # Not killed by any test cases: mutant survived
                if mutant.status == MutantStatus.PENDING:
                    print("Mutant", mutant.id, "survived")
                    mutant.updateResult(MutantStatus.SURVIVED)

    def _execute_ho_mutants(self) -> None:
        kernel_string = self.kernel_builder.kernel_string
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

            mutated_kernel_string = deepcopy(kernel_string)
            #Mutate the kenel code
            for mutant in ho_mutant.mutants:
                mutated_kernel_string = MutationExecutor.mutate(mutated_kernel_string, mutant.start, mutant.end, mutant.operator.replacement)
            
            result_queue = multiprocessing.Queue()
            exception_queue = multiprocessing.Queue()
            current_kernel_builder = deepcopy(self.kernel_builder)
            current_kernel_builder.kernel_string = mutated_kernel_string

            try:
                print("Start to execute high order mutant: ", ho_mutant.id)
                process = TerminableMutantRunner(current_kernel_builder, ho_mutant, self.test_cases, result_queue, exception_queue)
                process.start()
                process.join(timeout = self.timeout_second)

                if process.is_alive():
                    # The mutant timeout, force kill the GPU kernel and raise TimeoutException
                    process.terminate(exception_cls=TimeoutException, repeat_sec=2.0)
                    raise TimeoutException

                #Fetch and verify the result
                test_kernel = self.kernel_builder.build()
                try: 
                    #Fetch and check kernel exceptions
                    exceptions = exception_queue.get(False)
                    results = result_queue.get(False)
                except Exception:
                    print("Fail to fetch result from kernel, ingore this mutant")
                    raise UnknownException
                
                #Check the exceptions
                if len(exceptions) > 0:
                    if "CompilationException" in exceptions:
                        raise CompilationException
                    elif "UnknownException" in exceptions:
                        raise UnknownException
                    elif "RuntimeException" in exceptions and exceptions.count("RuntimeException") == len(self.test_cases):
                        # All test cases will cause runtime exception
                        raise RuntimeException

                for result in results:
                    #Get the corresponding test case
                    filter_test_cases = [tc for tc in self.test_cases if tc.id == result['id']]
                    if len(filter_test_cases) == 1:
                       test_case = filter_test_cases[0]
                       if not test_kernel.verify(result, test_case.output, test_case.verify, test_case.atol):
                           #Killed
                            print("Higher Order Mutant", ho_mutant.id, "killed by test case: ", test_case.id)
                            ho_mutant.updateResult(MutantStatus.KILLED, test_case.id)
                
            except CompilationException:
                print("Mutant", mutant.id, "has compilation error")
                mutant.updateResult(MutantStatus.COMPILE_ERROR)
            except UnknownException:
                print("Mutant", mutant.id, "has unknown error, ignore the mutant")
                mutant.updateResult(MutantStatus.IGNORE)
            except RuntimeException:
                print("Mutant", mutant.id, "has runtime error")
                mutant.updateResult(MutantStatus.RUNTIME_ERROR)
            except TimeoutException:
                print("Higher Order Mutant", mutant.id, "timeout")
                ho_mutant.status = MutantStatus.TIMEOUT
            else: 
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