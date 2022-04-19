from collections import namedtuple
from functools import partial
from typing import Callable, List, Tuple
from random import randint, random, uniform

from ga_utils.model import Model, TestModel
from ga_utils.extras import choices_with_replacement

# Type aliases (https://docs.python.org/3/library/typing.html)
Linspace = List[int] # Evenly spaced numbers over a specified interval
Parameter = namedtuple('Parameter', ['name', 'min_val', 'max_val'])
Genome = List[float] # Random set of values, one for each parameter, within their defined boundaries
GenomeX = Tuple[Genome, float] # A Genome and the it's score (provided by the Model through the eval method)
Population = List[GenomeX]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], GenomeX]
SelectionFunc = Callable[[Population], Tuple[GenomeX, GenomeX]]
CrossoverFunc = Callable[[Tuple, Tuple], Tuple[GenomeX, GenomeX]]
MutationFunc = Callable[[GenomeX], GenomeX]

def generate_genomeX(params: List[Parameter], model: Model) -> GenomeX:
    """ A Genome is a sequence of values (within it's respective boundaries),
    each one corresponding to an input parameter in your model. 
    a Genome becomes GenomeX when gets associated to the Model rating (evaluation)"""
    genome = [uniform(p.min_val, p.max_val) for p in parameters]
    return model.eval(genome)

def generate_population(size: int, params: List[Parameter], model: Model) -> Population: 
    return [generate_genomeX(params, model) for _ in range(size)]

def selection_pair(population: Population) -> Population:
    """ Selects k=2 genomex to act as parents for genetic evolution
    Caveat: The higher the rating is for a genomeX, more likely is to be selected"""
    return choices_with_replacement(
        population=population,
        weights=[genomeX[1] for genomeX in population],
        k=2
    )

def single_point_crossover(aX: GenomeX, bX: GenomeX, model: Model) -> Tuple[GenomeX, GenomeX]:
    """ Takes 2 genomes, splits them at the same length, recombines them y evaluates its fitness """
    a = aX[0] # Let's work only with the Genome part of the GenomeX
    b = bX[0]
    length = len(a)
    if length < 2:
        return a, b
        
    p = randint(1, length -1)
    # Build the new generated GenomeX
    aX = model.eval(a[0:p] + b[p:])
    bX = model.eval(b[0:p] + a[p:])
    return aX, bX

def mutation(genomex: GenomeX, params: List[Parameter], num: int = 1, prob: float = 0.5) -> GenomeX:
    genome = genomex[0]
    for _ in range(num):
        if random() < prob:
            p = randint(0, len(genome)-1) # p is the randomly chosen parameter (gene) to be mutated
            genome[p] = uniform(params[p].min_val, params[p].max_val)
    return model.eval(genome)

def run_evolution(
    populate_func: PopulateFunc,
    fitness_limit: int, # TODO Esto serian los segundos que tarda un GenoMoto en llegar a la meta
    crossover_func: CrossoverFunc,
    selection_func: SelectionFunc = selection_pair,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()
    for i in range(generation_limit):
        population = sorted(
            population,
            key = lambda genomeX: genomeX[1],
            reverse=True
        )
        if population[0][1] >= fitness_limit:
            break

        next_generation = population[0:2] # Elitism
        for j in range(int(len(population) / 2) - 1):
            parent_a, parent_b = selection_func(population)
            a, b = crossover_func(parent_a, parent_b)
            a = mutation_func(a)
            b = mutation_func(b)
            next_generation += [a, b]

        population = next_generation
    
    population = sorted(
        population,
        key = lambda genomeX: genomeX[1],
        reverse=True
    )
    return population, i    

# -------- run ----------
model = TestModel()

parameters = [
        Parameter('Weight', 80, 150),
        Parameter('Fork', 20, 50),
        Parameter('Angle', 10, 15),
        Parameter('CoG height', 0.4, 0.6),
        Parameter('Power', 30, 50),
    ]

population, generations = run_evolution(
    populate_func=partial( generate_population, size=10, params=parameters, model=model),
    crossover_func=partial(single_point_crossover, model=model),
    mutation_func=partial(mutation, params=parameters),
    fitness_limit=500, #194.6,
    generation_limit=10000
)

def genomeX_to_parameters(genomeX: GenomeX, params: List[Parameter]) -> List[Tuple]:
    return [(param.name, genomeX[0][i]) for i, param in enumerate(params)]
    
print(f'Number of generations: {generations}')
print(f'Best solution: {genomeX_to_parameters(population[0], parameters)}')
print(f'Best solution score: {population[0][1]}')
