from kernel_tuner.testing.mutation.mutation_operator import MutationOperator
from enum import Enum, unique

@unique
class MutantStatus(Enum):
    CREATED = 'Created'
    PENDING = 'Pending'
    KILLED='killed'
    PERF_KILLED = 'KilledByPerf'
    SURVIVED = 'Survived'
    NO_COVERAGE = 'NoCoverage'
    COMPILE_ERROR = 'CompileError'
    RUNTIME_ERROR = 'RuntimeError'
    TIMEOUT = 'Timeout'
    IGNORE = 'Ignore'

class MutantPosition:
    # Start from 1
    def __init__(self, line: int, column: int):
        self.line = line
        self.column = column

    def toJSON(self) -> str:
        return {"line":self.line, "column": self.column}

class Mutant:
    def __init__(self, id: int, operator: MutationOperator,
                  start: MutantPosition, end: MutantPosition, status = MutantStatus.CREATED):
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
            "opertorName": self.operator.name,
            "replacement": self.operator.replacement,
            "status": self.status.value,
            "killedById": self.killedBy,
            "coveredById": self.coveredBy,
            "location":{
                "start": self.start.toJSON(),
                "end": self.end.toJSON()
            }
        }
