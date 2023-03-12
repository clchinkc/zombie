
import random

# Define parameters
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.1
ELITISM = True

# Define the fitness function (from the first implementation)
def fitness(chromosome):
    return sum(chromosome)

# Define the selection method (from the first implementation)
def selection(population):
    total_fitness = sum(fitness(chromosome) for chromosome in population)
    r = random.uniform(0, total_fitness)
    partial_sum = 0
    for chromosome in population:
        partial_sum += fitness(chromosome)
        if partial_sum > r:
            return chromosome

# Define the crossover method (from the second implementation)
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1)-1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Define the mutation method (from the first implementation)
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i] # flip the bit
    return chromosome

# Define the population initialization method (from the first implementation)
def init_population():
    population = []
    for i in range(POPULATION_SIZE):
        chromosome = [random.randint(0, 1) for _ in range(10)]
        population.append(chromosome)
    return population

# Define the parent selection method (from the second implementation)
def parent_selection(population):
    parent1 = selection(population)
    parent2 = selection(population)
    while parent2 == parent1:
        parent2 = selection(population)
    return parent1, parent2

# Define the main function that runs the genetic algorithm
def run_genetic_algorithm():
    population = init_population()
    for i in range(GENERATIONS):
        # Evaluate the fitness of the population
        fitness_scores = [fitness(chromosome) for chromosome in population]
        
        # Select the parents for the next generation
        parents = [parent_selection(population) for _ in range(POPULATION_SIZE // 2)]
        
        # Perform crossover on the parents
        children = []
        for parent1, parent2 in parents:
            child1, child2 = crossover(parent1, parent2)
            children.append(child1)
            children.append(child2)
        
        # Perform mutation on the children
        mutated_children = [mutation(child) for child in children]
        
        # Combine parents and children into the next generation
        if ELITISM:
            elites = sorted(population, key=fitness, reverse=True)[:2]
            next_generation = elites + mutated_children
        else:
            next_generation = mutated_children
        
        # Set the current population to be the next generation
        population = next_generation
    
    # Return the best chromosome from the final population
    best_chromosome = max(population, key=fitness)
    return best_chromosome

# Run the genetic algorithm
best_chromosome = run_genetic_algorithm()

# Print the result
print("Best chromosome found:", best_chromosome)
