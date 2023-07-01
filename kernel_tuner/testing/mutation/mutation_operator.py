import re

from kernel_tuner.testing.mutation.mutant import Mutant, MutantPosition

class MutationOperator():

    def __init__(self, name: str, find: str or function, replacement: str, ignores: list[str] = None, backends: list[str] = None,tags: list[str] = []):
        self.name = name
        self.find = find
        self.replacement = replacement
        self.ignores = ignores
        self.backends = backends
        self.tags = tags

    def getNewDynamicOperator(self, replacement: str):
        return MutationOperator(self.name, self.find, replacement, self.ignores, self.backends, self.tags)

def loadAllOperators(name: str = None, tags: list[str] = []) -> list[MutationOperator]:
    #TODO: load all operators
    return loadAllTraditionalOperators(name, tags)

def loadAllTraditionalOperators(name: str = None, tags: list[str] = []) -> list[MutationOperator]:
    return conditional_boundary_replacement \
            + arithmetic_operator_replacement_shortcut \
            + conditional_operator_replacement \
            + math_replacement_operators \
            + math_assignment_replacement_shortcut \
            + negate_conditional_replacement \
            + arithmetic_operator_insertion \
            + conditional_operator_deletion \
            + arithmetic_operator_deletion \
            + conditional_statement_deletion 

def loadAllGPUNativeOperators(name: str = None, tags: list[str] = []) -> list[MutationOperator]:
    return allocation_swap \
            + allocation_increase \
            + allocation_decrease \
            + share_removal \
            + atom_removal \
            + gpu_index_replacement \
            + gpu_index_increment \
            + gpu_index_decrement \
            + sync_removal \
            + sync_child_removal

conditional_boundary_replacement = [
    MutationOperator('conditional_boundary_replacement', "<", "<=", ignores=["<=", "<<<"]),
    MutationOperator('conditional_boundary_replacement', "<=", "<"),
    MutationOperator('conditional_boundary_replacement', ">", ">=", ignores=[">=", ">>>"]),
    MutationOperator('conditional_boundary_replacement', ">=", ">")
]

arithmetic_operator_replacement_shortcut = [
    MutationOperator('arithmetic_operator_replacement_shortcut', "++", "--"),
    MutationOperator('arithmetic_operator_replacement_shortcut', "--", "++")
]

conditional_operator_replacement = [
    MutationOperator('conditional_operator_replacement', "&&", "||"),
    MutationOperator('conditional_operator_replacement', "||", "&&")
]

math_replacement_operators = [
    MutationOperator('math_replacement', "+", "-", ignores=["++", "+="]),
    MutationOperator('math_replacement', "-", "+", ignores=["--", "-="]),
    MutationOperator('math_replacement', "*", "/", ignores=["*="]),
    MutationOperator('math_replacement', "/", "*", ignores=["/="]),
    MutationOperator('math_replacement', "%", "*", ignores=["&="]),
    MutationOperator('math_replacement', "&", "|", ignores=["&&"]),
    MutationOperator('math_replacement', "|", "&", ignores=["||", "|="]),
    MutationOperator('math_replacement', "^", "&", ignores=["^="]),
    MutationOperator('math_replacement', "<<", ">>", ignores=["<<<"]),
    MutationOperator('math_replacement', ">>", "<<", ignores=[">>>"])
]

math_assignment_replacement_shortcut = [
    MutationOperator('math_assignment_replacement_shortcut', "+=", "-="),
    MutationOperator('math_assignment_replacement_shortcut', "-=", "+="),
    MutationOperator('math_assignment_replacement_shortcut', "*=", "/="),
    MutationOperator('math_assignment_replacement_shortcut', "/=", "*="),
    MutationOperator('math_assignment_replacement_shortcut', "%=", "*=") 
]

negate_conditional_replacement = [
    MutationOperator('negate_conditional_replacement', "<", ">=", ignores=["<<","<<<"]),
    MutationOperator('negate_conditional_replacement', ">", "<=", ignores=[">>",">>>"]),
    MutationOperator('negate_conditional_replacement', "<=", ">"),
    MutationOperator('negate_conditional_replacement', ">=", ">"),
    MutationOperator('negate_conditional_replacement', "==", "!="),
    MutationOperator('negate_conditional_replacement', "!=", "==")
]

def __arithmetic_operator_insertion(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

arithmetic_operator_insertion = [
    MutationOperator('arithmetic_operator_insertion', __arithmetic_operator_insertion, None),   
]

conditional_operator_deletion = [
    MutationOperator('conditional_operator_deletion', "!", "", ignores=["!="])
]

arithmetic_operator_deletion = [
    #MutationOperator('arithmetic_operator_deletion', "^", "")
]

def __conditional_statement_deletion(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

conditional_statement_deletion = [
    MutationOperator('conditional_operator_deletion', __conditional_statement_deletion, None)
]

def __allocation_swap(string_lines: list[str], operator, current_id):
    left_pattern = "<<<"
    right_pattern = ">>>"
    mutants = []
    for ridx, line in enumerate(string_lines):
        if (left_pattern in line) and (right_pattern in line):
            start_indexs = [substr.end() for substr in re.finditer(re.escape(left_pattern), line)]
            end_indexs = [substr.start() for substr in re.finditer(re.escape(right_pattern), line)]
            for (li, ri) in zip(start_indexs, end_indexs):
                subStr = line[li:ri] # "1024, 256"
                spStrs = subStr.split(",") # ["1024", " 256"]
                if len(spStrs) == 2:
                    row = ridx + 1
                    rep= ",".join(spStrs[::-1])
                    start_column = li + 1
                    end_column = ri + 1
                    op = operator.getNewDynamicOperator(rep)
                    start =  MutantPosition(row, start_column)
                    end = MutantPosition(row, end_column)
                    mutants.append(Mutant(current_id, op, start, end))
                    current_id += 1
    return mutants, current_id

allocation_swap = [
    MutationOperator('allocation_swap', __allocation_swap, None)
]

def __allocation_increase(string_lines, operator, current_id):
    left_pattern = "<<<"
    right_pattern = ">>>"
    mutants = []
    for ridx, line in enumerate(string_lines):
        if (left_pattern in line) and (right_pattern in line):
            start_indexs = [substr.end() for substr in re.finditer(re.escape(left_pattern), line)]
            end_indexs = [substr.start() for substr in re.finditer(re.escape(right_pattern), line)]
            for li, ri in zip(start_indexs, end_indexs):
                subStr = line[li: ri] # "1024, 256"
                spStrs = subStr.split(",") # ["1024", " 256"]
                if len(spStrs) == 2:
                    row = ridx + 1
                    start_column = li + 1
                    end_column = ri + 1
                    start =  MutantPosition(row, start_column)
                    end = MutantPosition(row, end_column)
                    gridRep = spStrs[0] + "+1" + "," + spStrs[1]  # "1024+1, 256"
                    blockRep = spStrs[0] + "," + spStrs[1]+ "+1" # "1024, 256+1"
                    opGrid = operator.getNewDynamicOperator(gridRep)
                    opBlock = operator.getNewDynamicOperator(blockRep)
                    mutants.append(Mutant(current_id, opGrid, start, end))
                    mutants.append(Mutant(current_id, opBlock, start, end))
                    current_id += 1
    return mutants, current_id

allocation_increase = [
    MutationOperator('allocation_increase', __allocation_increase, None)
]
    
def __allocation_decrease(string_lines, operator, current_id):
    left_pattern = "<<<"
    right_pattern = ">>>"
    mutants = []
    for ridx, line in enumerate(string_lines):
        if (left_pattern in line) and (right_pattern in line):
            start_indexs = [substr.end() for substr in re.finditer(re.escape(left_pattern), line)]
            end_indexs = [substr.start() for substr in re.finditer(re.escape(right_pattern), line)]
            for li, ri in zip(start_indexs, end_indexs):
                subStr = line[li:ri] # "1024, 256"
                spStrs = subStr.split(",") # ["1024", " 256"]
                if len(spStrs) == 2:
                    row = ridx + 1
                    start_column = li + 1
                    end_column = ri + 1
                    start =  MutantPosition(row, start_column)
                    end = MutantPosition(row, end_column)
                    gridRep = spStrs[0] + "-1" + "," + spStrs[1]  # "1024+1, 256"
                    blockRep = spStrs[0] + "," + spStrs[1]+ "-1" # "1024, 256+1"
                    opGrid = operator.getNewDynamicOperator(gridRep)
                    opBlock = operator.getNewDynamicOperator(blockRep)
                    mutants.append(Mutant(current_id, opGrid, start, end))
                    mutants.append(Mutant(current_id, opBlock, start, end))
                    current_id += 1
    return mutants, current_id

allocation_decrease = [
    MutationOperator('allocation_decrease', __allocation_decrease, None)
]

share_removal = [
    MutationOperator('share_removal', "__shared__", "")
]

def __atomic_add_sub_removal(string_lines, operator, current_id):
    left_patterns = ["atomicAdd(", "atomicSub("]
    rep_patterns = ["+=", "-="]
    right_pattern = ");"
    mutants = []
    for left_pattern, rep_pattern in zip(left_patterns, rep_patterns):
        for ridx, line in enumerate(string_lines):
            if (left_pattern in line) and (right_pattern in line): #Only support atomic op in one line now
                left_pattern_indexs = [[substr.start(), substr.end()] for substr in re.finditer(re.escape(left_pattern), line)]
                right_pattern_indexs = [[substr.start(), substr.end()] for substr in re.finditer(re.escape(right_pattern), line)]
                for li, ri in zip(left_pattern_indexs, right_pattern_indexs):
                    subStr = line[li[1] : ri[0]] 
                    spStrs = subStr.split(",") # Support only one "," in line
                    if len(spStrs) == 2:
                        #Calc positions
                        row = ridx + 1
                        start_column = li[0] + 1
                        end_column = ri[1] + 1
                        start_pos =  MutantPosition(row, start_column)
                        end_pos = MutantPosition(row, end_column)

                        #Calc replacement
                        addr = spStrs[0].removeprefix("&") #Remove the possible get addr op
                        num = spStrs[1]
                        rep = addr + rep_pattern + num + ";"
                        op = operator.getNewDynamicOperator(rep)
                        mutants.append(Mutant(current_id, op, start_pos, end_pos))
                        current_id += 1

    return mutants, current_id

atom_removal = [
    MutationOperator('atom_removal', __atomic_add_sub_removal, None)
]

gpu_index_replacement = [
    MutationOperator('gpu_index_replacement', "blockIdx.x", "threadIdx.x"),
    MutationOperator('gpu_index_replacement', "threadIdx.x;", "blockIdx.x;")
]

gpu_index_increment = [
    MutationOperator('gpu_index_increment', "blockIdx.x", "(blockIdx.x + 1)"),
    MutationOperator('gpu_index_increment', "threadIdx.x", "(threadIdx.x + 1)")
]

gpu_index_decrement = [
    MutationOperator('gpu_index_decrement', "blockIdx.x", "(blockIdx.x - 1)"),
    MutationOperator('gpu_index_decrement', "threadIdx.x", "(threadIdx.x - 1)")
]

sync_removal = [
    MutationOperator('sync_removal', "__syncthreads", "//__syncthreads")
]

sync_child_removal = [
    MutationOperator('sync_child_removal', "cudaDeviceSynchronize", "//cudaDeviceSynchronize")
]
