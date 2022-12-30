
import random

# define the states and transitions between states for the population model and the state machine model
from enum import Enum, auto


class State(Enum):
    HEALTHY = auto()
    INFECTED = auto()
    ZOMBIE = auto()
    DEAD = auto()


class Transition(Enum):
    INFECTION = auto()
    TURNING = auto()
    DEATH = auto()


class CellState(Enum):
    HEALTHY = auto()
    INFECTED = auto()
    ZOMBIE = auto()


LOCATIONS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
             (1, 2), (2, 0), (2, 1), (2, 2)]


class Individual:

    __slots__ = ['location', 'state', 'severity']

    def __init__(self):
        self.location = ()
        self.state = State.HEALTHY
        self.severity = 0


THRESHOLD = 0.5

# create a grid of cells and define the rules or logic that govern the behavior of each cell.
# creating a 2D list of cell objects and defining functions that update the states of the cells
# based on the states of the people within each cell.


class Cell:

    __slots__ = ['state', 'people']

    def __init__(self):
        self.state = CellState.HEALTHY
        self.people = []

    def update_cell_states(self, population):
        self.people = [
            individual for individual in population if individual.location == self.location]
        if any(individual.state == State.INFECTED for individual in self.people):
            self.state = CellState.INFECTED
        elif any(individual.state == State.ZOMBIE for individual in self.people):
            self.state = CellState.ZOMBIE
        else:
            self.state = CellState.HEALTHY


class ZombieOutbreakSimulation:

    __slots__ = ['num_rows', 'num_cols', 'num_individuals',
                 'num_time_steps', 'population', 'cells', 'zombies', 'infected', 'Cell']

    def __init__(self, num_rows, num_cols, num_individuals, num_time_steps):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_individuals = num_individuals
        self.num_time_steps = num_time_steps
        self.population = self.create_population()
        self.cells = self.create_grid()
        self.zombies = []
        self.infected = []
        self.Cell = Cell()

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

    def create_grid(self):
        grid = [[self.Cell for _ in range(self.num_cols)]
                for _ in range(self.num_rows)]
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
            self.Cell.update_cell_states(self.population)
            # Observe and analyze the results


zombieoutbreak = ZombieOutbreakSimulation(10, 10, 100, 100)
zombieoutbreak.run_simulation()
