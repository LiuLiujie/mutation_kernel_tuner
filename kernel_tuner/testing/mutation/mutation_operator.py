import pyparsing as pp
ppc = pp.pyparsing_common

class MutationOperator():

    def __init__(self, name: str, find: str or function,replacement: str, ignores: list[str] = None, backends: list[str] = None,tags: list[str] = []):
        self.name = name
        self.find = find
        self.ignores = ignores
        self.replacement = replacement
        self.backends = backends
        self.tags = tags

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
    MutationOperator('conditional_boundary_replacement', "\<(?![=])", "<="),
    MutationOperator('conditional_boundary_replacement', "<=", "<"),
    MutationOperator('conditional_boundary_replacement', "\>(?![=])", ">="),
    MutationOperator('conditional_boundary_replacement', ">=", ">")
]

arithmetic_operator_replacement_shortcut = [
    MutationOperator('arithmetic_operator_replacement_shortcut', "\+\+", "--"),
    MutationOperator('arithmetic_operator_replacement_shortcut', "--", "++")
]

conditional_operator_replacement = [
    MutationOperator('conditional_operator_replacement', "&&", "||"),
    MutationOperator('conditional_operator_replacement', "\|\|", "&&")
]

math_replacement_operators = [
    MutationOperator('math_replacement', "\+(?![\+\=\r\n\;])", "-"),
    MutationOperator('math_replacement', "\-(?![\-\=\r\n\;])", "+"),
    MutationOperator('math_replacement', "\*(?![\=])", "/"),
    MutationOperator('math_replacement', "\/(?![\=])", "*"),
    MutationOperator('math_replacement', "\%(?![\=])", "*"),
    MutationOperator('math_replacement', "\&(?![\=])", "|", ignores=["&&"]),
    MutationOperator('math_replacement', "\|(?![\=])", "&", ignores=["||"]),
    MutationOperator('math_replacement', "\^(?![\=])", "&"),
    MutationOperator('math_replacement', "\<\<", ">>", ignores=["<<<"]),
    MutationOperator('math_replacement', "\>\>", "<<", ignores=[">>>"])
]

math_assignment_replacement_shortcut = [
    MutationOperator('math_assignment_replacement_shortcut', "\+\=", "-="), # += -> -=
    MutationOperator('math_assignment_replacement_shortcut', "\-\=", "+="), # -= -> +=
    MutationOperator('math_assignment_replacement_shortcut', "\*\=", "/="), # *= -> /=
    MutationOperator('math_assignment_replacement_shortcut', "\/\=", "*="), # /= -> *=
    MutationOperator('math_assignment_replacement_shortcut', "\%\=", "*=")  # %= -> *=
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
    MutationOperator('conditional_operator_deletion', "\!(?![\=])", "")
]

arithmetic_operator_deletion = [
    #MutationOperator('arithmetic_operator_deletion', "^", "")
]

def __conditional_statement_deletion(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

conditional_statement_deletion = [
    MutationOperator('arithmetic_operator_deletion', __conditional_statement_deletion, None)
]

def __allocation_swap(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

allocation_swap = [
    MutationOperator('allocation_swap', __allocation_swap, None)
]

def __allocation_increase(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

allocation_increase = [
    MutationOperator('allocation_increase', __allocation_increase, None)
]
    
def __allocation_decrease(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

allocation_decrease = [
    MutationOperator('allocation_decrease', __allocation_decrease, None)
]

share_removal = [
    MutationOperator('share_removal', "__shared__", "")
]

def __atom_removal(string_lines, operator, current_id):
    mutants = []
    return mutants, current_id

atom_removal = [
    MutationOperator('atom_removal', __atom_removal, None)
]

gpu_index_replacement = [
    MutationOperator('gpu_index_replacement', "blockIdx\.x", "threadIdx.x"),
    MutationOperator('gpu_index_replacement', "threadIdx\.x;", "blockIdx.x;")
]

gpu_index_increment = [
    MutationOperator('gpu_index_increment', "blockIdx\.x", "(blockIdx.x + 1)"),
    MutationOperator('gpu_index_increment', "threadIdx\.x", "(threadIdx.x + 1)")
]

gpu_index_decrement = [
    MutationOperator('gpu_index_decrement', "blockIdx\.x", "(blockIdx.x - 1)"),
    MutationOperator('gpu_index_decrement', "threadIdx\.x", "(threadIdx.x - 1)")
]

sync_removal = [
    MutationOperator('sync_removal', "__syncthreads", "//__syncthreads")
]

sync_child_removal = [
    MutationOperator('sync_child_removal', "cudaDeviceSynchronize", "//cudaDeviceSynchronize")
]