# genetic algorithm search for continuous function optimization

import numpy


def fitness_function(x):
        return x[0] ** 2.0 + x[1] ** 2.0  # smaller is better


# tournament selection
def tournament_selection(pop, scores, k=3):
        # first random selection
        selection_ix = numpy.random.randint(len(pop))
        for ix in numpy.random.randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]


    
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if numpy.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = numpy.random.randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if numpy.random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
    return bitstring

def genetic_algorithm_search(fitness_function, selection_method, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    # decode bitstring to numbers
    def decode(bitstring):
        decoded = list()
        largest = 2**n_bits
        for i in range(len(bounds)):
            # extract the substring
            start, end = i * n_bits, (i * n_bits) + n_bits
            substring = bitstring[start:end]
            # convert bitstring to a string of chars
            chars = "".join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
            # store
            decoded.append(value)
        return decoded

    # initial population of random bitstring
    pop = [numpy.random.randint(2, size=n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, numpy.inf
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(p) for p in pop]
        # evaluate all candidates in the population
        scores = [fitness_function(d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, decoded[i], scores[i]))
        # select parents
        selected = [selection_method(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                c = mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval, decoded]

if __name__ == "__main__":
    # problem configuration
    bounds = [[-5,5],[-5,5]]
    # algorithm configuration
    n_bits = 16
    n_iter = 100
    n_pop = 100
    r_cross = 0.9
    r_mut = 1.0 / (float(n_bits) * len(bounds))
    # execute the algorithm
    best, score, decoded = genetic_algorithm_search(fitness_function, tournament_selection, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    print("Best Solution:")
    print(best) # [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # sort decoded population by fitness
    decoded.sort(key=lambda d: abs(fitness_function(d)))
    # print the best result found
    print("f(%s) = %.10g" % (decoded[0], fitness_function(decoded[0])))



