

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



"""
A genetic algorithm is an optimization technique inspired by the process of natural selection, used to solve complex problems by mimicking the process of evolution. It can be employed to control NPC agents in a game or simulation. Here are the steps to implement a genetic algorithm:

Define NPC behavior parameters: Identify the characteristics that influence NPC actions, such as movement patterns, decision-making rules, and reactions to players.
Define the fitness function: Create a function that evaluates the quality of each NPC's behavior based on performance criteria relevant to the game or simulation, like survival time or enemies defeated.
Implement the genetic algorithm: Develop code that applies natural selection principles to evolve NPC behavior parameters over generations. This includes creating functions for:
a. Initializing the population of genes.
b. Selecting the fittest NPCs for reproduction.
c. Generating offspring through crossover and mutation.
d. Evaluating each NPC's fitness using the fitness function.
e. Termination once the criteria is met.
f. Extract the solution from the most fit individual in the population.
Integrate the genetic algorithm into the game or simulation: Incorporate the genetic algorithm functions into the game loop to update NPC behavior parameters and apply the algorithm as needed.
Please design flexible NPC behavior parameters and fitness functions: Ensure these components can be easily modified for experimenting with different configurations and observing their effects on NPC behavior.
"""

"""
Important parameters of a genetic algorithm include:
Mutation Probability. If the mutation probability is too small, evolution will occur slowly, but if it is too large, useful adaptations passed down from parents to children will often be damaged.
Incentives. Fitness function depending on the average/minimum/mazimum of several games affect the robustness and variance of the resulted strategy a lot. When we selected for players that performed better on average, they turned out to be less consistent and made more stupid mistakes. When we selected for players with the best worst performance, they were much more consistent, although they may not have been quite as good on average.
Population Size. Larger populations evolve faster, since there are more opportunities for useful mutations to occur. On the other hand, larger populations exhibit more variability, meaning that the worst players in large populations are worse than the worst players in small populations.
Number of Generations. It takes time for good strategies to evolve.
Sheer Dumb Luck. Even if a perfectly optimal strategy were to evolve by chance, it’s possible that its environment could (by chance) be so severe that it performs poorly and dies out. Conversely, poor strategies can do well if they receive gentle environments.
"""

"""
The provided code is a Python implementation of a genetic algorithm designed to solve a movement control problem in a simulated zombie survival game environment. The goal is to find the best movement pattern for a survivor, represented as a chromosome (x, y position), that maximizes survival time while minimizing the number of hits received from zombies. The genetic algorithm is implemented using two classes: GeneticAlgorithm and GeneticFunctions.

The problem-specific implementation of the GeneticFunctions class is called MovementControl. It takes input parameters like the game map, survivor's position, zombies' positions, and genetic algorithm parameters such as population size, probabilities of crossover and mutation, and maximum number of iterations. The MovementControl class implements the fitness function, which computes a chromosome's fitness as the minimum distance to the nearest zombie. The crossover operator returns the parents unchanged, and the mutation operator randomly changes the survivor's position on the map.

Several methods are implemented in the MovementControl class, including probability_crossover, probability_mutation, initial, fitness, check_stop, parents, crossover, mutation, random_chromo, and distance. These methods define the problem's environment and how the genetic algorithm will behave, as well as manipulate and evaluate chromosomes representing potential solutions.

An instance of the MovementControl class is created with the necessary parameters, and then an instance of the GeneticAlgorithm class is created using the MovementControl instance. The genetic algorithm is executed using the ga.run() method, which returns the best solution found. The best solution is then printed.

In summary, the code demonstrates a genetic algorithm implementation for a movement control problem in a zombie survival game, with the objective of finding the best position for the survivor. The algorithm iteratively evolves a population of potential solutions by applying genetic operators such as crossover and mutation and evaluates each solution using a fitness function based on the distance to the nearest zombie.
"""

"""
To implement evolutionary algorithms in a simulation of a student's activity in a zombie apocalypse, you would need to define a set of input features that represent the student's actions and characteristics, such as their physical fitness, weapons proficiency, decision-making skills, and resourcefulness. You would then need to define a set of rules or objectives for the simulation, such as surviving for a certain number of days or reaching a safe zone.

Next, you would need to create a population of "student" objects that represent different combinations of the input features. These students would need to be able to interact with their environment and make decisions based on their input features and the rules of the simulation.

You would then need to define a fitness function that measures how well each student performs in the simulation based on their actions and the simulation's rules. This fitness function could be used to evaluate the performance of each student and to select the "best" students for reproduction.

Finally, you would need to implement an evolution loop that repeats the following steps:

Evaluate the fitness of each student in the population.
Select the "best" students for reproduction based on their fitness.
Create new students by combining the input features of the selected students in different ways (e.g., through crossover and mutation).
Add the new students to the population.
Repeat the process until the simulation ends or the population reaches a predetermined size.
This process would allow the students in the population to adapt and evolve over time based on their actions and the rules of the simulation, potentially resulting in a more efficient and effective strategy for surviving a zombie apocalypse.
"""

"""
https://github.com/topics/natural-selection?l=python
https://stackoverflow.com/questions/46337156/python-genetic-algorithm-natural-selection
https://blog.csdn.net/weixin_37790882/article/details/84034956
https://blog.csdn.net/oxuzhenyi/article/details/70037833
https://medium.datadriveninvestor.com/genetic-algorithm-made-intuitive-with-natural-selection-and-python-project-from-scratch-3462f7793a3f
https://pygad.readthedocs.io/en/latest/README_pygad_nn_ReadTheDocs.html
https://github.com/franklindyer/easy-ga
https://github.com/franklindyer/AP-Research-Genetic-Algorithm-Project
"""