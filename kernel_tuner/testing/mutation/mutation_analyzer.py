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
            for ridx, line in enumerate(string_lines):
                columns = [substr.start() for substr in re.finditer(re.escape(operator.old), line)]
                for cidx in columns:
                    row = ridx + 1
                    column = cidx + 1
                    start = MutantPosition(row, column)
                    end = MutantPosition(row, column + len(operator.old))
                    mutants.append(Mutant(current_id, operator, start, end))
                    current_id+=1
        return mutants



        





    
