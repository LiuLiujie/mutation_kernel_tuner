class MutationOperator():

    def __init__(self, name: str, old: str, replacement: str, tags: list[str] = []):
        self.name = name
        self.old = old
        self.replacement = replacement
        self.tags = tags

def loadAllOperators(name: str = None, tags: list[str] = []) -> list[MutationOperator]:
    #TODO: demo for loading operator
    return [MutationOperator('math_replacement', "+", "-")]