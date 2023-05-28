from kernel_tuner.testing.mutation.mutant import HigherOrderMutant, Mutant
from kernel_tuner.testing.test_case import TestCase
from kernel_tuner.testing.testing_result import TestingResult


class MutationResult(TestingResult):
    
    def __init__(self, mutants: list[Mutant], test_cases: list[TestCase],
                  higher_order_mutants: list[HigherOrderMutant] = []) -> None:
        super().__init__(test_cases)
        self.mutants = mutants
        self.higher_order_mutants = higher_order_mutants

    def toJSON(self) -> dict:
        if len(self.higher_order_mutants):
            return {
                **super().toJSON(),
                **{"mutants": [mutant.toJSON() for mutant in self.mutants]},
                **{"higher_order_mutants": [ho_mutant.toJSON() for ho_mutant in self.higher_order_mutants]}
            }

        return {
            **super().toJSON(),
            **{"mutants": [mutant.toJSON() for mutant in self.mutants]}
        }            