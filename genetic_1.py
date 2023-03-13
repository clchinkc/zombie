# genetic algorithm search for discrete function optimization

import random

import matplotlib.pyplot as plt
import numpy as np


def elitism_selection(population_sorted, fitness_scores, n_parents):
    elite = population_sorted[:n_parents]
    return elite


def roulette_selection(population, fitness_scores, n_parents):
    e_x = np.exp(fitness_scores - np.max(fitness_scores))
    softmax = e_x / e_x.sum(axis=0)
    roulette = random.choices(population, weights=softmax, k=n_parents)
    return roulette


def rank_selection(population, fitness_scores, n_parents):
    rank = np.arange(len(population))
    rank = rank / np.sum(rank)
    rank_selection = random.choices(population, weights=rank.tolist(), k=n_parents)
    return rank_selection


def tournament_selection(population, fitness_scores, n_parents):
    k = min(5, len(population))
    selected_parents = []
    tournament_indices = np.random.choice(len(population), size=(n_parents, k), replace=True)
    for indices in tournament_indices:
        tournament_scores = [fitness_scores[j] for j in indices]
        best_index = indices[np.argmax(tournament_scores)]
        selected_parents.append(population[best_index])
    return selected_parents


def calculate_fitness_score(population, fitness_fn):
    scores = np.array([fitness_fn(chromosome) for chromosome in population])
    inds = np.argsort(scores)[::-1]
    return list(scores[inds]), list(np.array(population)[inds, :])


def point_crossover(parent1, parent2):
    crossover_point = random.randint(0, chrom_len - 1) # int(chrom_len / 2)
    child = parent1.copy()
    child[crossover_point:] = parent2[crossover_point:].copy()
    return child


def subset_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [parent1[i] if i >= start and i < end else None for i in range(len(parent1))]
    child = [x if x is not None else y for x, y in zip(child, parent2)]
    return child

def random_crossover(parent1, parent2):
    child = [random.choice([x, y]) for x, y in zip(parent1, parent2)]
    return child

def blend_crossover(parent1, parent2):
    alpha = 0.5
    a = np.minimum(parent1, parent2)
    b = np.maximum(parent1, parent2)
    range_ = b - a
    child = np.random.uniform(a - range_ * alpha, b + range_ * alpha)
    return child

def average_crossover(parent1, parent2):
    child = (parent1 + parent2) / 2
    return child

def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    child = [x if x is not None else y for x, y in zip(child, parent2)]
    return child


def crossover(population_nextgen, crossover_md, offspring_size, crossover_rate):
    offspring = []
    for i in range(offspring_size):
        parent1 = population_nextgen[i % len(population_nextgen)]
        parent2 = population_nextgen[(i + 1) % len(population_nextgen)]
        # parent1 = max(random.sample(population_nextgen, 1), key=lambda x: x[0]) # tournament selection
        # parent2 = random.choice(population_nextgen) # random selection
        if random.random() < crossover_rate:
            child = crossover_md(parent1, parent2)
            offspring.append(child)
        else:
            offspring.append(random.choice([parent1, parent2]))
    return offspring

def xor_mutation(offspring_population, mutation_mask):
    return np.logical_xor(offspring_population, mutation_mask)
    
def gaussian_mutation(chromosome, mutation_mask):
    mu = 0.5
    sigma = 0.25
    bounds = (0, 1)
    mutation_array = np.random.normal(mu, sigma, np.array(chromosome).shape)
    mutation_array = mutation_array * mutation_mask
    return np.clip(chromosome + mutation_array, bounds[0], bounds[1])

def mutate(offspring_population, mutate_md, mutation_rate):
    mutation_array = np.random.random(np.array(offspring_population).shape)
    mutation_mask = mutation_array < mutation_rate
    offspring_population = mutate_md(offspring_population, mutation_mask)
    return offspring_population

def genetic_algorithm(problem, keep_md, selection_md, crossover_md, mutate_md, pop_size, chrom_len, n_generations, mutation_rate, crossover_rate):
    population_nextgen = problem.init_population(pop_size, chrom_len)
    best_chromosomes = []
    best_fitness = []
    elite_size = pop_size // 2
    parent_size = pop_size // 2
    offspring_size = pop_size - elite_size

    for i in range(n_generations):
        scores, population_sorted = calculate_fitness_score(population_nextgen, problem.fitness)
        population_nextgen = []
        elite_population = keep_md(population_sorted, scores, elite_size)
        population_nextgen.extend(elite_population)
        
        parents = selection_md(population_sorted, scores, parent_size)
        offspring_population = crossover(parents, crossover_md, offspring_size, crossover_rate)
        offspring_population = mutate(offspring_population, mutate_md, mutation_rate)
        population_nextgen.extend(offspring_population)
        
        best_chromosomes.append(population_sorted[0])
        best_fitness.append(scores[0])

    return best_chromosomes, best_fitness

# Define fitness function for One Max problem, a binary optimization problem
class OneMaxProblem:

    @staticmethod
    def init_population(pop_size, n_feat):
        population = np.random.randint(2, size=(pop_size, n_feat))
        return population
    
    @staticmethod
    def fitness(chromosome):
        return sum(chromosome)

# Define fitness function for One Max problem, a continuous optimization problem
class ContinuousOneMaxProblem:
    
    @staticmethod
    def init_population(pop_size, n_feat):
        # float number population
        population = np.random.random((pop_size, n_feat))
        return population
    
    @staticmethod
    def fitness(chromosome):
        return sum(chromosome)

# Define fitness function for Traveling Salesman Problem, a permutation optimization problem
class TravelingSalesmanProblem:
    @staticmethod
    def init_population(pop_size, chrom_len):
        # Generate a random population of chromosomes
        population = np.random.permutation(chrom_len)
        population = np.tile(population, (pop_size, 1))
        return population

    @staticmethod
    def fitness(chromosome):
        # Calculate the total distance of the route
        total_distance = 0
        for i in range(len(chromosome)):
            j = (i + 1) % len(chromosome)
            distance = abs(chromosome[j] - chromosome[i])
            total_distance += distance
        # Calculate the fitness as the inverse of the total length of the route
        fitness = 1 / total_distance
        return fitness
    
# Define problem
# problem = OneMaxProblem()
problem = ContinuousOneMaxProblem()
# problem = TravelingSalesmanProblem() # need other genetic operators

# Define parameters
pop_size = 100
chrom_len = 100
n_generations = 100
mutation_rate = 0.05
crossover_rate = 1.0


def main():
    # Run genetic algorithm
    best_chromosome, best_fitness = genetic_algorithm(problem, elitism_selection, roulette_selection, random_crossover, gaussian_mutation,
                                                    pop_size, chrom_len, n_generations, mutation_rate, crossover_rate)
    
    best_chromosome_1, best_fitness_1 = genetic_algorithm(problem, elitism_selection, roulette_selection, blend_crossover, gaussian_mutation,
                                                    pop_size, chrom_len, n_generations, mutation_rate, crossover_rate)

    # Print results
    # print("Best chromosome:", '\n', '\n'.join(map(str, best_chromosome)))
    # print("Best fitness:", '\n', best_fitness)
    print("Final chromosome:", best_chromosome[-1])
    print("Final fitness:", best_fitness[-1])
    print("Final fitness roulette:", best_fitness_1[-1])

    # Plot results

    plt.plot(best_fitness)
    plt.plot(best_fitness_1)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Best fitness", "Best fitness 1"])
    plt.show()
    

if __name__ == "__main__":
    main()





