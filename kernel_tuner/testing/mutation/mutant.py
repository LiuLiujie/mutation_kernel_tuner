#from kernel_tuner.testing.mutation.mutation_operator import MutationOperator
from enum import Enum, unique

@unique
class MutantStatus(Enum):
    #When created, init state
    CREATED = 'Created'

    #When first run starts
    PENDING = 'Pending'
    
    #After at least one test, no longer run for following tests
    KILLED = 'killed'
    PERF_KILLED = 'KilledByPerf'
    COMPILE_ERROR = 'CompileError'

    #Intermedia state for a single test run, will change after all tests
    TIMEOUT = 'Timeout'             # to KILLED
    RUNTIME_ERROR = 'RuntimeError'  # to KILLED
    NO_COVERAGE = 'NoCoverage'      # to SURVIVED

    #All tests survive
    SURVIVED = 'Survived'

    #Unknown error
    IGNORE = 'Ignore'

class MutantPosition:
    # Start from 1
    def __init__(self, line: int, column: int):
        self.line = line
        self.column = column

    def toJSON(self) -> str:
        return {"line":self.line, "column": self.column}

class Mutant:
    def __init__(self, id: int, operator, start: MutantPosition, end: MutantPosition,
                  status = MutantStatus.CREATED):
        self.id = id
        self.operator = operator
        self.start = start
        self.end = end
        self.status = status
        self.killedBy = []
        self.coveredBy = []

    def updateMutantPosition(self, start: MutantPosition, end: MutantPosition):
        self.start = start
        self.end = end
    
    def updateResult(self, status: MutantStatus, killed_by_id: int = None):
        self.status = status
        if killed_by_id is not None:
            self.killedBy.append(killed_by_id)
            self.coveredBy.append(killed_by_id)

    def toJSON(self) -> dict:
        return {
            "id": self.id,
            "operatorName": self.operator.name,
            "replacement": self.operator.replacement,
            "status": self.status.value,
            "killedById": self.killedBy,
            "coveredById": self.coveredBy,
            "location":{
                "start": self.start.toJSON(),
                "end": self.end.toJSON()
            }
        }

class HigherOrderMutant:
    def __init__(self, id: str, mutants: list[Mutant], mutation_order: int, status = MutantStatus.CREATED) -> None:
        self.id = id
        self.mutants = mutants
        self.status = status
        self.mutation_order = mutation_order
        self.killedBy = []
        self.coveredBy = []
    
    def updateResult(self, status: MutantStatus, killed_by_id: int = None):
        self.status = status
        if killed_by_id is not None:
            self.killedBy.append(killed_by_id)
            self.coveredBy.append(killed_by_id)
    
    def toJSON(self) -> dict:
        return {
            "combined_mutants": [mutant.id for mutant in self.mutants],
            "mutation_order": self.mutation_order,
            "status": self.status.value,
            "killedById": self.killedBy,
            "coveredById": self.coveredBy,
        }
    