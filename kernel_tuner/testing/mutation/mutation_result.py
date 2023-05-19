from kernel_tuner.testing.mutation.mutant import Mutant
from kernel_tuner.testing.test_case import TestCase
from kernel_tuner.testing.testing_result import TestingResult


class MutationResult(TestingResult):
    
    def __init__(self, mutants: list[Mutant], test_cases: list[TestCase]) -> None:
        super().__init__(test_cases)
        self.mutants = mutants

    def conbine_result(self, result):
        self.mutants.append(result.mutants)
        self.test_cases.append(result.test_cases)

    def toJSON(self) -> dict:
        return {**super().toJSON(), **{"mutants": [mutant.toJSON() for mutant in self.mutants]}}
     
    