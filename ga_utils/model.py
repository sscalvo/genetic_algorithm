from typing import List, Tuple

Genome = List[float] # A random sample of values, one for each parameter
GenomeX = Tuple[Genome, float] # A Genome and the it's rating provided by the Model through the fitness method

class Model(object):
    def __init__(self):
        pass

    def eval(self, genome: Genome):
        raise Exception("NotImplementedException")


# ------------------------------------------

class TestModel(Model):
    def __init__(self):
       Model.__init__(self)

    def eval(self, genome: Genome) -> GenomeX:
       # print('TestModel evaluating genome..')
        score = genome[0] - genome[1] + genome[2] - genome[3] + genome[4]
        return genome, score


# ------------------------------------------

class MathlabModel(Model):
    def __init__(self):
       Model.__init__(self)

    def eval(self, genome: Genome) -> GenomeX:
        # print('MathlabModel evaluating genome..')
        score = self.call_to_mathlab(*genome)
        return genome, score    
    
    def call_to_mathlab(*args):
        # Your code to invoke mathlab functions
        pass