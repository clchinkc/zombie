
import random

# define the states and transitions between states for the population model and the state machine model
from enum import Enum

class State(Enum):
    HEALTHY = 1
    INFECTED = 2
    ZOMBIE = 3
    DEAD = 4

class Transition(Enum):
    INFECTION = 1
    TURNING = 2
    DEATH = 3

class CellState(Enum):
    HEALTHY = 1
    INFECTED = 2
    ZOMBIE = 3
    
LOCATIONS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    
class Individual:
    def __init__(self):
        self.location = ()
        self.state = State.HEALTHY
        self.severity = 0

THRESHOLD = 0.5
    
    
class ZombieOutbreakSimulation:
    def __init__(self, num_rows, num_cols, num_individuals, num_time_steps):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_individuals = num_individuals
        self.num_time_steps = num_time_steps
        self.population = self.create_population()
        self.cells = self.create_grid()
        self.zombies = []
        self.infected = []
        
    # create a population of individuals and assign them initial states and attributes
    # creating a list of individual objects and assigning them attributes such as location, and state
    def create_population(self):
        population = []
        for i in range(self.num_individuals):
            individual = Individual()
            individual.location = random.choice(LOCATIONS)
            individual.state = State.HEALTHY
            population.append(individual)
        return population
    
    # define the rules or events that trigger transitions between states in the population model and the state machine model
    # take in an individual's attributes and return the appropriate transition based on those attributes
    def get_transition(self, individual, zombies, infected):
        if individual.state == State.HEALTHY:
            if individual.location in zombies or individual.location in infected:
                return Transition.INFECTION
            else:
                return None
        elif individual.state == State.INFECTED:
            if individual.severity > THRESHOLD:
                return Transition.TURNING
            else:
                return None
        elif individual.state == State.ZOMBIE:
            return None
        elif individual.state == State.DEAD:
            return None
    
    # implement the transitions between states and the rules or events that trigger them 
    # in the population model and the state machine model. 
    # iterates through the population and updates the states of the individuals based on the transitions that are triggered.
    def update_states(self, population, zombies, infected):
        for individual in population:
            transition = self.get_transition(individual, zombies, infected)
            if transition == Transition.INFECTION:
                individual.state = State.INFECTED
                infected.append(individual.location)
            elif transition == Transition.TURNING:
                individual.state = State.ZOMBIE
                infected.remove(individual.location)
                zombies.append(individual.location)
    
    # create a grid of cells and define the rules or logic that govern the behavior of each cell. 
    # creating a 2D list of cell objects and defining functions that update the states of the cells 
    # based on the states of the people within each cell.
    class Cell:
        def __init__(self):
            self.state = CellState.HEALTHY
            self.people = []

        def update_cell_states(cells, population):
            for row in cells:
                for cell in row:
                    cell.people = [individual for individual in population if individual.location == cell.location]
                    if any(individual.state == State.INFECTED for individual in cell.people):
                        cell.state = CellState.INFECTED
                    elif any(individual.state == State.ZOMBIE for individual in cell.people):
                        cell.state = CellState.ZOMBIE
                    else:
                        cell.state = CellState.HEALTHY

    def create_grid(self):
        grid = [[self.Cell() for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                grid[i][j].location = (i, j)
        return grid
    
    # run the simulation for a given time period and observe the changes 
    # in the population, the state machine model, and the cellular automaton over time. 
    # iterates through each time step and updates the states of the individuals, the state machine model, and the cellular automaton 
    # based on the rules and events that are triggered.
    def run_simulation(self):
        for t in range(self.num_time_steps):
            self.update_states(self.population, self.zombies, self.infected)
            self.CellState.update_cell_states(self.cells, self.population)
            # Observe and analyze the results

zombieoutbreak = ZombieOutbreakSimulation(10, 10, 100, 100)
zombieoutbreak.run_simulation()