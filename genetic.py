
"""
To incorporate a genetic algorithm like the one shown in the previous example into a larger project to control the activity of NPC agents in a game or simulation, you would need to do the following:

Define the NPC behavior parameters: These are the characteristics that determine the NPC's behavior. They could include things like movement patterns, decision-making rules, or responses to player actions.

Define the fitness function: This is a function that measures the quality of each NPC's behavior based on how well it performs in the game or simulation. It could be based on metrics such as survival time, number of enemies defeated, or other criteria that are relevant to the specific game or simulation.

Implement the genetic algorithm: This involves writing code to apply the principles of natural selection to evolve the NPC behavior parameters over multiple generations. This might involve functions to select the fittest NPCs for reproduction, generate offspring through crossover and mutation, and evaluate the fitness of each NPC using the fitness function.

Integrate the genetic algorithm into the game or simulation: Once you have implemented the genetic algorithm, you can integrate it into your game or simulation by calling the genetic algorithm functions at appropriate points in the game loop. For example, you might call the evolve function at the end of each game tick to update the NPC behavior parameters and apply the genetic algorithm.

It's also a good idea to design your NPC behavior parameters and fitness function in a way that is flexible and easy to modify, so you can experiment with different configurations and see how they affect the NPC behavior.
"""

import random

"""
class Survivor:
    def __init__(self, behavior_params):
        self.behavior_params = behavior_params

    def update(self, game_state):
        # Update the Survivor's behavior based on the game state and the behavior parameters
        pass
        
# Define the behavior parameters for the initial population of students
initial_pop = [
    {
        'movement_pattern': 'random_walk',
        'decision_making_rule': 'flee_from_zombies',
        'response_to_player': 'follow'
    },
    {
        'movement_pattern': 'follow_player',
        'decision_making_rule': 'attack_zombies',
        'response_to_player': 'help'
    }
]

# Create the initial population of students
pop = [Student(behavior_params) for behavior_params in initial_pop]



def fitness(survivor, game_state):
    # Evaluate the Survivor's behavior based on the game state and return a score
    score = 0

    # Calculate the Survivor's survival time
    survival_time = game_state['time'] - survivor.time_of_death
    if survivor.is_alive:
        survival_time += 1
    score += survival_time

    # Add points for each enemy defeated
    score += survivor.enemies_defeated

    # Penalize the Survivor for each time they were hit by an enemy
    score -= survivor.hits_received * 5

    return score
    
# The game state includes the current state of the game, such as the positions of players, zombies, and students, the time elapsed since the start of the game, and any other relevant data.

"""

class GeneticAlgorithm(object):
    def __init__(self, genetics):
        self.genetics = genetics

    def run(self):
        population = self.genetics.initial()
        while True:
            fitness_value_dict = [(self.genetics.fitness(chromosome),  chromosome) for chromosome in population]
            if self.genetics.check_stop(fitness_value_dict): break
            population = self.update_population(fitness_value_dict)
        return population

    def update_population(self, fitness_population):
        parents_generator = self.genetics.parents(fitness_population)
        size = len(fitness_population)
        nexts = []
        while len(nexts) < size:
            parents = next(parents_generator)
            cross = random.random() < self.genetics.probability_crossover()
            children = self.genetics.crossover(parents) if cross else parents
            for chromosome in children:
                mutate = random.random() < self.genetics.probability_mutation()
                nexts.append(self.genetics.mutation(chromosome) if mutate else chromosome)
        return nexts[0:size]

class GeneticFunctions(object):
    def probability_crossover(self):
        r"""returns rate of occur crossover(0.0-1.0)"""
        return 1.0

    def probability_mutation(self):
        r"""returns rate of occur mutation(0.0-1.0)"""
        return 0.0

    def initial(self):
        r"""returns list of initial population
        """
        return []

    def fitness(self, chromosome):
        r"""returns domain fitness value of chromosome
        """
        return 0

    def check_stop(self, fits_populations):
        r"""stop run if returns True
        - fits_populations: list of (fitness_value, chromosome)
        """
        return False

    def parents(self, fits_populations):
        r"""generator of selected parents
        """
        gen = iter(sorted(fits_populations, reverse=True, key=lambda x: x[0]))
        while True:
            f1, ch1 = next(gen)
            f2, ch2 = next(gen)
            yield (ch1, ch2)
        return

    def crossover(self, parents):
        r"""breed children
        """
        return parents

    def mutation(self, chromosome):
        r"""mutate chromosome
        """
        return chromosome
    pass



class MovementControl(GeneticFunctions):
    def __init__(self, map, person_pos, zombie_positions,
                 limit=450, size=400,
                 prob_crossover=0.9, prob_mutation=0.2):
        self.map = map
        self.person_pos = person_pos
        self.zombie_positions = zombie_positions
        self.counter = 0
        self.limit = limit
        self.size = size
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        
    # self.game_state = game_state

    # GeneticFunctions interface impls
    def probability_crossover(self):
        return self.prob_crossover

    def probability_mutation(self):
        return self.prob_mutation

    def initial(self):
        return [self.random_chromo() for j in range(self.size)]

    def fitness(self, chromo):
        # larger is better, matched == 0
        distance_to_zombies = min(self.distance(chromo, zombie_pos) for zombie_pos in self.zombie_positions)
        return distance_to_zombies
    
    # return fitness(survivor, self.game_state)
    # put survivor in the game_state and return reward

    def check_stop(self, fits_populations):
        self.counter += 1
        if self.counter % 10 == 0:
            best_match = list(sorted(fits_populations))[-1][1]
            fits = [f for f, ch in fits_populations]
            best = max(fits)
            worst = min(fits)
            ave = sum(fits) / len(fits)
            print(f"[G {self.counter:3d}] score=({best:4d}, {worst:4d}, {ave:4d})")
        return self.counter >= self.limit

    def parents(self, fits_populations):
        gen = iter(sorted(fits_populations))
        while True:
            f1, ch1 = next(gen)
            f2, ch2 = next(gen)
            yield (ch1, ch2)
            pass
        return
    
    """
    def parents(self, fits_populations):
        # Select the fittest individuals using the tournament selection operator.
        return tournament(fits_populations, size=2, tournsize=3)
    
    def tournament(fits_populations, size, tournsize):
    # Select the best individual among tournsize randomly chosen individuals,
    # and repeat until size individuals are chosen.
        chosen = []
        for i in range(size):
            aspirants = [random.choice(fits_populations) for i in range(tournsize)]
            chosen.append(max(aspirants))
        return chosen
    
    """

    def crossover(self, parents):
        child1 = []
        child2 = []
        for i in parents[0]:
            if random.random() < 0.5:
                child1.append(parents[0][i])
                child2.append(parents[1][i])
            else:
                child1.append(parents[1][i])
                child2.append(parents[0][i])
        return child1, child2
        # return parents
    
    """
    def crossover(self, parents):
        father, mother = parents
        index1 = random.randint(1, self.map.width - 2)
        index2 = random.randint(1, self.map.height - 2)
        if index1 > index2: index1, index2 = index2, index1
        child1 = (father[0][:index1] + mother[0][index1:index2] + father[0][index2:],
                  father[1][:index1] + mother[1][index1:index2] + father[1][index2:])
        child2 = (mother[0][:index1] + father[0][index1:index2] + mother[0][index2:],
                  mother[1][:index1] + father[1][index1:index2] + mother[1][index2:])
        return (child1, child2)
    """
    """
    def crossover(self, parents):
        params1, params2 = parents
        child_params = {}
        for key in params1:
            if random.random() < 0.5:
                child_params[key] = params1[key]
            else:
                child_params[key] = params2[key]
        return child_params
    """

    def mutation(self, chromosome):
        # Apply a random mutation that changes a single gene in the chromosome.
        # The mutation should be applied with probability 1/length of the chromosome.
        # Return the mutated chromosome.
        if random.random() < self.probability_mutation():
            index = random.randint(0, len(chromosome) - 1)
            chromosome[index] = self.random_chromo()
        return chromosome
    
    """
    def mutation(self, chromosome):
        for i in chromosome:
            if random.random() < self.probability_mutation():
                chromosome[i] = self.random_chromo()
        return chromosome
    """
    """
    def mutation(self, survivor):
        params = survivor.behavior_params
        for key in params:
            if random.random() < self.probability_mutation():
            params[key] = self.random_chromo()
        return params
    """

    def random_chromo(self):
        return (random.randint(0, self.map.width), random.randint(0, self.map.height))

    # or all legal moves
    # or return 0 for all illegal moves

    def distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

"""
# Define the initial population of Survivors
initial_pop = [Survivor({'movement_pattern': 'random', 'decision_making_rule': 'attack_weakest'}),
              Survivor({'movement_pattern': 'follow_player', 'decision_making_rule': 'run_away'}),
              Survivor({'movement_pattern': 'stationary', 'decision_making_rule': 'attack_nearest'})]

# Define the game state
game_state = {'time': 0, 'players': [], 'zombies': []}

for i in range(1000):
    # Update the game state
    game_state = update_game_state()

    # Update the student behaviors
    for student in pop:
        student.update(game_state)

    # Evolve the student population
    pop = evolve(pop, game_state)
"""

# create an instance of the MovementControl class
control = MovementControl(map, person_pos, zombie_positions, limit=1000, size=200, prob_crossover=0.9, prob_mutation=0.2)

# create an instance of the GeneticAlgorithm class
ga = GeneticAlgorithm(control)

# run the genetic algorithm
best = ga.run()[0]

# print the best solution found by the genetic algorithm
print(best)

"""
The MovementControl class takes the following arguments in its __init__ method:
map: a 2D array representing the map of the simulation
person_pos: a tuple with the position (x, y) of the person on the map
zombie_positions: a list of tuples with the positions (x, y) of the zombies on the map
limit: an integer representing the maximum number of iterations the genetic algorithm will run
size: an integer representing the size of the population (number of chromosomes in each generation)
prob_crossover: a float representing the probability of crossover (a genetic operator that combines the genetic information of two parents to produce offspring)
prob_mutation: a float representing the probability of mutation (a genetic operator that randomly modifies the genetic information of an individual)

The MovementControl class also defines the following methods:
probability_crossover: returns the probability of crossover (a float).
probability_mutation: returns the probability of mutation (a float).
initial: returns a list of initial population (a set of potential solutions to the problem). Each chromosome in the population is a tuple representing the position (x, y) of the person on the map.
fitness: returns the fitness value (a measure of how good a solution is) of a chromosome (a potential solution). The fitness value is the minimum distance to the nearest zombie.
check_stop: returns True if the genetic algorithm should stop, False otherwise. The genetic algorithm stops when the number of iterations exceeds the limit.
parents: a generator that yields pairs of parents (chromosomes) that will be used to produce offspring. The parents are chosen from the fittest members of the current population.
crossover: takes a pair of parents and produces offspring by applying crossover. In this case, the crossover operator simply returns the parents unchanged.
mutation: takes a chromosome and produces a mutated version by applying mutation. In this case, the mutation operator randomly changes the position of the person on the map.
random_chromo: returns a randomly generated chromosome (a tuple representing the position (x, y) of the person on the map).
distance: returns the Euclidean distance between two positions (tuples with x and y coordinates).
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
"""