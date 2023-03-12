
import random

import numpy as np


class TravelingSalesmanProblem:
    def __init__(self, locations):
        self.locations = locations
        
    def init_population(self, pop_size):
        pop = []
        for i in range(pop_size):
            route = list(range(len(self.locations)))
            random.shuffle(route)
            pop.append(route)
        return pop
        
    def fitness(self, route):
        # Calculate the total distance of the route
        dist = 0
        for i in range(len(route)):
            j = (i + 1) % len(route)
            city1, city2 = route[i], route[j]
            x1, y1 = self.locations[city1]
            x2, y2 = self.locations[city2]
            dist += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist
    
class GeneticAlgorithm:
    def __init__(self, problem, n_generations=500, pop_size=100, elite_size=20, mutation_rate=0.1):
        self.problem = problem
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        
    def run(self):
        pop = self._initialize_population()
        for i in range(self.n_generations):
            graded = self._grade_population(pop)
            elite = [route for _, route in graded[:self.elite_size]]
            selection_probs = self._calculate_selection_probs(graded)
            children = self._breed_population(pop, selection_probs)
            pop = elite + children
        return graded[0]
    
    def _initialize_population(self):
        pop = []
        for i in range(self.pop_size):
            route = list(range(len(self.problem.locations)))
            random.shuffle(route)
            pop.append(route)
        return pop
    
    def _grade_population(self, pop):
        graded = [(self.problem.fitness(route), route) for route in pop]
        return sorted(graded, key=lambda x: x[0])
    
    def _calculate_selection_probs(self, graded):
        total_fitness = sum(fitness for fitness, _ in graded)
        return [fitness / total_fitness for fitness, _ in graded]
    
    def _breed_population(self, pop, selection_probs):
        children = []
        for i in range(len(pop) - self.elite_size):
            parent1, parent2 = self._select_parents(pop, selection_probs)
            child = self._breed(parent1, parent2)
            children.append(child)
        return children
    
    def _select_parents(self, pop, selection_probs):
        parents = random.choices(pop, selection_probs, k=2)
        return parents
    
    def _breed(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        for i in range(start, end):
            child[i] = parent1[i]
        remaining = [city for city in parent2 if city not in child]
        remaining_index = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = remaining[remaining_index]
                remaining_index += 1
        return child

# Problem configuration
locations = [(0, 0), (1, 5), (2, 3), (5, 4), (4, 2), (6, 7)]
problem = TravelingSalesmanProblem(locations)

# Algorithm configuration
n_generations = 500
pop_size = 200
elite_size = 100
mutation_rate = 0.1

# Execute the algorithm
ga = GeneticAlgorithm(problem, n_generations, pop_size, elite_size, mutation_rate)
best_route, best_fitness = ga.run()

# Print final results
print('Best solution: %s' % best_route)
print('Best fitness: %s' % best_fitness)


