from random import choices

def choices_with_replacement(population, weights, k):
    # By using indexes, not only we obtain the randomly chosen indexes, but also, 
    # if some (or all) elements in population where repeated, then repeated elements
    # would be allowed in the final returned value, otherwise not.
    indexes = [x for x in range(len(population))] 
    if len(population) < k:
        return []
    c = []
    while len(c) < k:
        c += choices(population=indexes, weights=weights, k=k - len(c))
        c = list(set(c))
    return [population[i] for i in c]
