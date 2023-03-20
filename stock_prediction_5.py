
# genetic algorithm search for continuous or discrete function optimization

"""
This code implements a genetic algorithm to optimize a trading strategy for Apple stock data. The trading strategy is defined by a chromosome of length 10, where each element corresponds to a weight for a different feature of the stock data (e.g. close price, open price, volume).

The fitness function for each chromosome is calculated as the correlation between the predicted stock prices and the actual stock prices. The genetic algorithm involves selection, crossover, and mutation steps to evolve the population of chromosomes towards the optimal solution.

The selection function selects the population for the next generation based on their fitness scores, with higher fitness scores having a higher probability of being selected.

The crossover function performs crossover between pairs of selected chromosomes, where a random crossover point is chosen and the offspring is created by combining the first part of one parent and the second part of the other parent.

The mutation function applies random mutations to the offspring population, where each element of each chromosome has a probability of being randomly replaced with a new value between -1 and 1. The mutation rate is initially set to 0.5 and decreases over time as the algorithm converges.

The genetic algorithm is run for 100 generations. After each generation, the population is selected, crossed over, mutated, and the mutation rate is adjusted based on convergence. The best chromosome found after the final generation is printed along with its fitness score.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    bounds = (-1, 1)
    mutation_array = np.random.normal(mu, sigma, np.array(chromosome).shape)
    mutation_array = mutation_array * mutation_mask
    return np.clip(chromosome + mutation_array, bounds[0], bounds[1])

def random_mutation(chromosome, mutation_mask):
    bounds = (-1, 1)
    mutation_array = np.random.random(np.array(chromosome).shape)
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

# Define fitness function for the Stock Prediction problem
class StockPredictionProblem:
    def __init__(self, df):
        self.df = df
    
    @staticmethod
    def init_population(pop_size, n_feat):
        # float number population
        population = np.random.uniform(-1, 1, (pop_size, n_feat))
        return population
    
    def fitness(self, chromosome):
        # Perform calculations using the chromosome
        predictions = []
        for i in range(len(self.df)):
            # Make predictions using the chromosome
            if i == 0:
                prediction = 0
            else:
                prediction = chromosome[0]*self.df['Close'][i-1] + chromosome[1]*self.df['Open'][i-1] + chromosome[2]*self.df['High'][i-1] + chromosome[3]*self.df['Low'][i-1] + chromosome[4]*self.df['Volume'][i-1]
            predictions.append(prediction)
        # Calculate the fitness score, which is the inverse of the MSE plus the L1 norm of the chromosome
        mse = np.mean((self.df['Close'] - predictions)**2)
        print("MSE: ", mse)
        l1_norm = np.sum(np.abs(chromosome))
        print("L1 norm: ", l1_norm)
        return 1/(mse + l1_norm)

class StockPredictionProblem2:
    def __init__(self, df):
        self.df = df
    
    @staticmethod
    def init_population(pop_size, n_feat):
        # float number population
        population = np.random.uniform(-1, 1, (pop_size, n_feat))
        return population
    
    def fitness(self, chromosome):
        # Perform calculations using the chromosome
        predictions = []
        for i in range(len(self.df)):
            # Make predictions using the chromosome
            if i == 0:
                prediction = self.df['Close'][i]
            elif i == 1:
                prediction = chromosome[0]*self.df['Close'][i-1] + chromosome[1]*self.df['Open'][i-1] + chromosome[2]*self.df['High'][i-1] + chromosome[3]*self.df['Low'][i-1] + chromosome[4]*self.df['Volume'][i-1] + chromosome[5]*self.df['Close'][i-1] + chromosome[6]*self.df['Open'][i-1] + chromosome[7]*self.df['High'][i-1] + chromosome[8]*self.df['Low'][i-1] + chromosome[9]*self.df['Volume'][i-1]
            else:
                prediction = chromosome[0]*self.df['Close'][i-1] + chromosome[1]*self.df['Open'][i-1] + chromosome[2]*self.df['High'][i-1] + chromosome[3]*self.df['Low'][i-1] + chromosome[4]*self.df['Volume'][i-1] + chromosome[5]*self.df['Close'][i-2] + chromosome[6]*self.df['Open'][i-2] + chromosome[7]*self.df['High'][i-2] + chromosome[8]*self.df['Low'][i-2] + chromosome[9]*self.df['Volume'][i-2]
            predictions.append(prediction)
        # Calculate the fitness score, which is the inverse of the MSE plus the L1 norm of the chromosome
        mse = np.mean((self.df['Close'] - predictions)**2)
        l1_norm = np.sum(np.abs(chromosome))
        return 1/(mse + l1_norm)

# Define the problem
problem = StockPredictionProblem(pd.read_csv('apple_stock_data.csv'))

# Define parameters
pop_size = 100
chrom_len = 5
n_generations = 100
mutation_rate = 0.1
crossover_rate = 0.99


best_chromosome, best_fitness = genetic_algorithm(problem, elitism_selection, roulette_selection, blend_crossover, gaussian_mutation,
                                                pop_size, chrom_len, n_generations, mutation_rate, crossover_rate)

# Print results
print("Final chromosome:", list(best_chromosome[-1]))
print("Final fitness:", best_fitness[-1])

# Plot results
plt.plot(best_fitness)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(["Best fitness"])
plt.show()

# Use the best chromosome to make predictions on the training data
predictions = []
for i in range(len(problem.df)):
    # Make predictions using the chromosome
    if i == 0:
        prediction = 0
    else:
        prediction = best_chromosome[-1][0]*problem.df['Close'][i-1] + best_chromosome[-1][1]*problem.df['Open'][i-1] + best_chromosome[-1][2]*problem.df['High'][i-1] + best_chromosome[-1][3]*problem.df['Low'][i-1] + best_chromosome[-1][4]*problem.df['Volume'][i-1]
    predictions.append(prediction)
    
# Plot results
plt.plot(problem.df['Close'], label="Actual")
plt.plot(predictions, label="Predicted")
plt.xlabel("Day")
plt.ylabel("Stock price")
plt.legend()
plt.show()


"""

# Define mutation parameters
mutation_rate = 0.5
mutation_rate_decay = 0.9
mutation_rate_min = 0.01
convergence_counter = 0
convergence_threshold = 10

"""
