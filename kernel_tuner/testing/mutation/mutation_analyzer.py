from kernel_tuner.core import KernelSource
from kernel_tuner.testing.mutation.mutant import Mutant, MutantPosition
from kernel_tuner.testing.mutation.mutation_operator import MutationOperator

import re

class MutationAnalyzer:

    def __init__(self, kernel_source: KernelSource, operators: list[MutationOperator]) -> None:
        self.kernel_source = kernel_source
        self.operators = operators

    def analyze(self) -> list[Mutant]:
        mutants = self.__analyze_mutants(self.operators)
        return mutants

    def __analyze_mutants(self, operators: list[MutationOperator]) -> list[Mutant]:
        mutants = []
        current_id = 0

        #TODO: currently only support single file kernel
        kernel_string = self.kernel_source.get_kernel_string(0)
        string_lines = kernel_string.splitlines()

        for operator in operators:
            if callable(operator.find):
                new_mutants, current_id = operator.find(string_lines, operator, current_id)
            else:
                new_mutants, current_id = self.__re_analyzer(string_lines, operator, current_id)
            mutants += new_mutants
        return mutants
    
    def __re_analyzer(self, string_lines, operator, current_id):
        mutants = []
        for ridx, line in enumerate(string_lines):
            # Detect ingored patterns
            ignore_columns = []
            if operator.ignores:
                for ignore in operator.ignores:
                    for substr in re.finditer(ignore, line):
                        ignore_columns += list(range(substr.start(), substr.end()))
            
            columns = [substr.start() for substr in re.finditer(operator.find, line) if substr.start() not in ignore_columns]
            for cidx in columns:
                row = ridx + 1
                column = cidx + 1
                start = MutantPosition(row, column)
                end = MutantPosition(row, column + len(operator.find))
                mutants.append(Mutant(current_id, operator, start, end))
                current_id+=1
        return mutants, current_id



        





    
