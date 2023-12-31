"""
To implement a simulation of a person's activity during a zombie apocalypse at school, we would need to define several classes and functions to represent the different elements of the simulation.

First, we would need a Person class to represent each person in the simulation. This class would have attributes to track the person's location, state (alive, undead, or escaped), health, and any weapons or supplies they may have. It would also have methods to move the person on the grid and interact with other people and zombies.

Next, we would need a Zombie class to represent each zombie in the simulation. This class would have similar attributes and methods as the Person class, but would also include additional attributes and methods to simulate the behavior of a zombie (such as attacking living people and spreading the infection).

We would also need a School class to represent the layout of the school and track the locations of people and zombies on the grid. This class would have a two-dimensional array to represent the grid, with each cell containing a Person or Zombie object, or None if the cell is empty. The School class would also have methods to move people and zombies on the grid and update their states based on the rules of the simulation.

Finally, we would need a main simulate function that would set up the initial conditions of the simulation (such as the layout of the school, the number and distribution of people and zombies, and any weapons or supplies), and then run the simulation for a specified number of steps. This function would use the School class to move people and zombies on the grid and update their states, and could also include additional code to track and display the progress of the simulation.
"""

from __future__ import annotations

import math
import random
import time
import tkinter as tk
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from itertools import product
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import seaborn as sns
from matplotlib import animation, colors, patches
from matplotlib.transforms import Bbox
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras import layers, models


class HealthState(Enum):

    HEALTHY = auto()
    INFECTED = auto()
    ZOMBIE = auto()
    DEAD = auto()

    @classmethod
    def name_list(cls) -> list[str]:
        return [enm.name for enm in HealthState]

    @classmethod
    def value_list(cls) -> list[int]:
        return [enm.value for enm in HealthState]


# state pattern

class StateMachine(ABC):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @abstractmethod
    def update_state(self, individual: Individual, severity: float) -> None:
        pass

    def is_infected(self, individual: Individual, severity: float, randomness=random.random()) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        if any(other.health_state == HealthState.ZOMBIE for other in individual.connections):
            if randomness < infection_probability:
                return True
        return False

    def is_turned(self, individual: Individual, severity: float, randomness=random.random()) -> bool:
        turning_probability = individual.infection_severity
        if randomness < turning_probability:
            return True
        return False

    def is_died(self, individual: Individual, severity: float, randomness=random.random()) -> bool:
        death_probability = severity
        if any(other.health_state == HealthState.HEALTHY or other.health_state == HealthState.INFECTED for other in individual.connections):
            if randomness < death_probability:
                return True
        return False

# can add methods that change behaviour based on the state

class HealthyStateMachine(StateMachine):
    def update_state(self, individual: Individual, severity: float) -> None:
        if self.is_infected(individual, severity):
            individual.health_state = HealthState.INFECTED

class InfectedStateMachine(StateMachine):
    def update_state(self, individual: Individual, severity: float) -> None:
        individual.infection_severity = round(min(1, individual.infection_severity + 0.1), 1)
        if self.is_turned(individual, severity):
            individual.health_state = HealthState.ZOMBIE
        elif self.is_died(individual, severity):
            individual.health_state = HealthState.DEAD

class ZombieStateMachine(StateMachine):
    def update_state(self, individual: Individual, severity: float) -> None:
        if self.is_died(individual, severity):
            individual.health_state = HealthState.DEAD

class DeadStateMachine(StateMachine):
    def update_state(self, individual: Individual, severity: float) -> None:
        pass

class StateMachineFactory:
    @staticmethod
    def get_instance():
        return {
            HealthState.HEALTHY: HealthyStateMachine(),
            HealthState.INFECTED: InfectedStateMachine(),
            HealthState.ZOMBIE: ZombieStateMachine(),
            HealthState.DEAD: DeadStateMachine()
        }

    @staticmethod
    def update_state(individual: Individual, severity: float) -> None:
        state_machines = StateMachineFactory.get_instance()
        state_machine = state_machines[individual.health_state]
        state_machine.update_state(individual, severity)


# Strategy pattern


class MovementStrategy(ABC):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    @abstractmethod
    def choose_direction(self, individual, school):
        raise NotImplementedError


@dataclass
# if no neighbors, choose random direction
class RandomMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        return random.choice(self.legal_directions)
    
@dataclass
# simulate Brownian motion
class BrownianMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]
    
    def __post_init__(self):
        self.damping_coefficient = 0.5
        self.std_dev = 1
        self.scale_factor = math.sqrt(2 * self.damping_coefficient)*self.std_dev

    def choose_direction(self):
        while True:
            direction = np.rint(np.random.normal(loc=0, scale=self.scale_factor, size=2)).astype(int)
            if tuple(direction) in self.legal_directions:
                return direction

@dataclass
# if healthy or other having no alive neighbors
class FleeZombiesStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        zombies_locations = [zombies.location for zombies in self.neighbors if zombies.health_state == HealthState.ZOMBIE]
        return self.direction_against_closest(self.individual, self.legal_directions, zombies_locations)

    # find the closest zombie and move away from it
    def direction_against_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]]) -> tuple[int, int]:
        distances = [np.linalg.norm(np.subtract(np.array(individual.location), np.array(target))) for target in target_locations]
        closest_index = np.argmin(distances)
        closest_target = target_locations[closest_index]
        direction_distances = [np.linalg.norm(np.add(d, np.array(individual.location)) - closest_target) for d in np.array(legal_directions)]
        max_distance = np.max(direction_distances)
        farthest_directions = np.where(direction_distances == max_distance)[0]
        return legal_directions[random.choice(farthest_directions)]

@dataclass
# if zombie or other having no zombie neighbors
class ChaseHumansStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        alive_locations = [alive.location for alive in self.neighbors if alive.health_state == HealthState.HEALTHY]
        return self.direction_towards_closest(self.individual, self.legal_directions, alive_locations)

    # find the closest human and move towards it
    def direction_towards_closest(self, individual: Individual,legal_directions: list[tuple[int, int]],target_locations: list[tuple[int, int]],) -> tuple[int, int]:
        current_location = np.array(individual.location)
        new_locations = current_location + np.array(legal_directions)
        distance_matrix = np.linalg.norm(new_locations[:, np.newaxis, :] - np.array(target_locations)[np.newaxis, :, :], axis=2)
        min_distance = np.min(distance_matrix[distance_matrix != 0])
        min_directions = np.where(distance_matrix == min_distance)[0]
        return legal_directions[random.choice(min_directions)]

    # If the closest person is closer than the closest zombie, move towards the person, otherwise move away from the zombie
    # or move away from zombie is the priority and move towards person is the secondary priority


@dataclass
class NoMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        return (0, 0)


# may use init to store individual and school
# may use function, use closure to store individual and school


class MovementStrategyFactory:
    def create_strategy(self, individual, school):
        # get legal directions
        legal_directions = school.get_legal_directions(individual)
        # early exit
        if not legal_directions:
            return NoMovementStrategy(individual, legal_directions, [])
        # get neighbors
        neighbors = school.get_neighbors(individual.location, individual.sight_range)
        # early exit
        # if no neighbors, random movement
        if not neighbors:
            return RandomMovementStrategy(individual, legal_directions, neighbors)
        # get number of human and zombies neighbors
        alive_number = len([alive for alive in neighbors if alive.health_state == HealthState.HEALTHY])
        zombies_number = len([zombies for zombies in neighbors if zombies.health_state == HealthState.ZOMBIE])
        # if no human neighbors, move away from the closest zombies
        if alive_number == 0 and zombies_number > 0:
            return FleeZombiesStrategy(individual, legal_directions, neighbors)
        # if no zombies neighbors, move towards the closest human
        elif zombies_number == 0 and alive_number > 0:
            return ChaseHumansStrategy(individual, legal_directions, neighbors)
        # if both human and zombies neighbors, zombies move towards the closest human and human move away from the closest zombies
        else:
            if individual.health_state == HealthState.ZOMBIE and alive_number > 0:
                return ChaseHumansStrategy(individual, legal_directions, neighbors)
            elif (individual.health_state == HealthState.HEALTHY or individual.health_state == HealthState.INFECTED) and zombies_number > 0:
                return FleeZombiesStrategy(individual, legal_directions, neighbors)
            elif individual.health_state == HealthState.DEAD:
                return NoMovementStrategy(individual, legal_directions, neighbors)
            else:
                return BrownianMovementStrategy(individual, legal_directions, neighbors)

    # may consider update the grid according to the individual's location
    # after assigning all new locations to the individuals
    # may add a extra attribute to store new location


class Individual:

    __slots__ = ("id","state","location","connections","infection_severity","interact_range","__dict__",)

    def __init__(self,id: int, health_state: HealthState, location: tuple[int, int],) -> None:
        self.id: int = id
        self.health_state: HealthState = health_state
        self.location: tuple[int, int] = location
        self.connections: list[Individual] = []
        self.infection_severity: float = 0.0
        self.interact_range: int = 2

        # different range for different states
        # may use random distribution

    @cached_property
    def sight_range(self) -> int:
        return self.interact_range + 3

    # fluent interface
    def add_connection(self, other: Individual) -> None:
        self.connections.append(other)

    def move(self, direction: tuple[int, int]) -> None:
        dx, dy = direction
        x, y = self.location
        self.location = (x + dx, y + dy)

    def choose_direction(self, movement_strategy) -> tuple[int, int]:
        return movement_strategy.choose_direction()

    def update_state(self, severity: float) -> None:
        StateMachineFactory.update_state(self, severity)

    def get_info(self) -> str:
        return f"Individual {self.id} is {self.health_state.name} and is located at {self.location}, having connections with {self.connections}, infection severity {self.infection_severity}, interact range {self.interact_range}, and sight range {self.sight_range}."

    def __str__(self) -> str:
        return f"Individual {self.id}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id}, {self.health_state.value}, {self.location})"


# separate inheritance for human and zombie class


class School:

    __slots__ = ("size", "grid", "strategy_factory", "__dict__",)

    def __init__(self, size: int) -> None:
        self.size = size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid: np.ndarray = np.full((size, size), None, dtype=object)
        self.strategy_factory = MovementStrategyFactory()

        # may turn to width and height
        # may put Cell class in the grid where Cell class has individual attributes and rates

    def add_individual(self, individual: Individual) -> None:
        self.grid[int(individual.location[0])][int(
            individual.location[1])] = individual

    def get_individual(self, location: tuple[int, int]) -> Optional[Individual]:
        return self.grid[location[0]][location[1]]

    def remove_individual(self, location: tuple[int, int]) -> None:
        self.grid[location[0]][location[1]] = None

    # may change the add\remove function to accept individual class as well as location

    def update_connections(self) -> None:
        for row in self.grid:
            for cell in row:
                if cell is None:
                    continue
                neighbors = self.get_neighbors((cell.location), cell.interact_range)
                for neighbor in neighbors:
                    cell.add_connection(neighbor)

    def update_individual_location(self, individual: Individual) -> None:
        movement_strategy = self.strategy_factory.create_strategy(individual, self)
        direction = individual.choose_direction(movement_strategy)
        self.move_individual(individual, direction)

    # update the grid in the population based on their interactions with other people
    def update_grid(self, population: list[Individual], migration_probability: float, randomness=random.random()) -> None:
        for individuals in population:
            cell = self.get_individual(individuals.location)

            if cell is None or cell.health_state == HealthState.DEAD:
                continue

            if randomness < migration_probability:
                self.update_individual_location(cell)

                # if right next to then don't move
                # get neighbors with larger and larger range until there is a human
                # sight range is different from interact range

    def get_neighbors(self, location: tuple[int, int], interact_range: int = 2):
        x, y = location
        x_range = range(max(0, x - interact_range), min(self.size, x + interact_range + 1))
        y_range = range(max(0, y - interact_range), min(self.size, y + interact_range + 1))
        
        neighbors = filter(
            lambda pos: (pos[0] != x or pos[1] != y) and self.within_distance(self.grid[x][y], self.grid[pos[0]][pos[1]], interact_range),
            product(x_range, y_range)
        )

        return [self.grid[i][j] for i, j in neighbors]

    def within_distance(self, individual1: Optional[Individual], individual2: Optional[Individual], interact_range: int):
        if individual1 is None or individual2 is None:
            return False
        # check if the two individuals are within a certain distance of each other
        distance = self.distance(individual1.location, individual2.location)
        return distance < interact_range

    def distance(self, location1: tuple[int, int], location2: tuple[int, int]) -> float:
        # get the distance between two individuals
        distance = float(np.linalg.norm(np.subtract(location1, location2)))
        return distance

    def get_legal_directions(self, individual: Individual) -> list[tuple[int, int]]:
        # get all possible legal moves for the individual
        legal_directions = [
            (i, j) for i in range(-1, 2) for j in range(-1, 2)
            if (i == 0 and j == 0)
            or self.legal_location((individual.location[0] + i, individual.location[1] + j))
        ]
        return legal_directions

    def legal_location(self, location: tuple[int, int]) -> bool:
        return self.in_bounds(location) and self.not_occupied(location)

    def in_bounds(self, location: tuple[int, int]) -> bool:
        # check if the location is in the grid
        return (0 <= location[0] < self.size and 0 <= location[1] < self.size)

    def not_occupied(self, location: tuple[int, int]) -> bool:
        # check if the location is empty
        return self.grid[location[0]][location[1]] is None

    # return any(agent.position == (x, y) for agent in self.agents)

    def move_individual(self, individual: Individual, direction: tuple[int, int]) -> None:
        old_location = individual.location
        individual.move(direction)
        self.remove_individual(old_location)
        self.add_individual(individual)

    def get_info(self) -> str:
        return "\n".join(
            " ".join(str(individual.health_state.value) if individual else " " for individual in column)
            for column in self.grid
            )

    # return the count inside the grid
    
    def __str__(self) -> str:
        return f"School({self.size})"

    def __repr__(self) -> str:
        return "%s(%d,%d)" % (self.__class__.__name__, self.size)


class Population:
    def __init__(self, school_size: int, population_size: int) -> None:
        self.school: School = School(school_size)
        self.agent_list: list[Individual] = []
        self.severity: float = 0.0
        self.init_population(school_size, population_size)
        self.update_population_metrics()
        self.observers = []

    def add_individual(self, individual: Individual) -> None:
        self.agent_list.append(individual)
        self.school.add_individual(individual)

    def remove_individual(self, individual: Individual) -> None:
        self.agent_list.remove(individual)
        self.school.remove_individual(individual.location)

    def create_individual(self, id: int, school_size: int) -> Individual:
        state = random.choices(list(HealthState), weights=[0.7, 0.1, 0.2, 0.0])[0]
        available_locations = [(i, j) for i in range(school_size) for j in range(school_size) if self.school.legal_location((i, j))]
        location = random.choice(available_locations)
        return Individual(id, state, location)

    def init_population(self, school_size: int, population_size: int) -> None:
        for i in range(population_size):
            individual = self.create_individual(i, school_size)
            self.add_individual(individual)

    def clear_population(self) -> None:
        self.agent_list.clear()
        self.school.grid = np.full((self.school.size, self.school.size), None, dtype=object)

    # a method to init using a grid of "A", "I", "Z", "D"

    def run_population(self, num_time_steps: int) -> None:
        for time in range(num_time_steps):
            print("Time step: ", time + 1)
            self.severity = time / num_time_steps
            print("Severity: ", self.severity)
            self.school.update_grid(self.agent_list, self.migration_probability)
            print("Updated Grid")
            self.school.update_connections()
            print("Updated Connections")
            self.update_individual_states()
            print("Updated State")
            self.update_population_metrics()
            print("Updated Population Metrics")
            individual_info = self.get_all_individual_info()
            print(individual_info)
            print("Got Individual Info")
            school_info = self.school.get_info()
            print(school_info)
            print("Got School Info")
            self.notify_observers()
            print("Notified Observers")
        self.clear_population()

    def update_individual_states(self) -> None:
        for individual in self.agent_list:
            individual.update_state(self.severity)
            if individual.health_state == HealthState.DEAD:
                self.school.remove_individual(individual.location)

    def update_population_metrics(self) -> None:
        self.calculate_state_counts()
        self.calculate_probabilities()

    def calculate_state_counts(self) -> None:
        state_counts = Counter([individual.health_state for individual in self.agent_list])
        self.num_healthy = state_counts[HealthState.HEALTHY]
        self.num_infected = state_counts[HealthState.INFECTED]
        self.num_zombie = state_counts[HealthState.ZOMBIE]
        self.num_dead = state_counts[HealthState.DEAD]
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie
        
    def calculate_probabilities(self) -> None:
        self.infection_probability = 1 - (1 / (1 + math.exp(-self.severity))) # logistic function
        self.turning_probability = self.severity / (1 + self.severity) # softplus function
        self.death_probability = self.severity  # linear function
        self.migration_probability = self.population_size / (self.population_size + 1)

        # may use other metrics or functions to calculate the probability of infection, turning, death, migration

    def get_all_individual_info(self) -> str:
        return f"Population of size {self.population_size}\n" + \
            "\n".join(individual.get_info() for individual in self.agent_list)

    def attach_observer(self, observer: Observer) -> None:
        self.observers.append(observer)

    def notify_observers(self) -> None:
        for observer in self.observers:
            observer.update()

    def __str__(self) -> str:
        return f"Population with {self.num_healthy} healthy, {self.num_infected} infected, and {self.num_zombie} zombie individuals"

    def __repr__(self) -> str:
        return f"Population({self.school.size}, {self.population_size})"


# Observer Pattern

class Observer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def display_observation(self) -> None:
        pass


class SimulationObserver(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.statistics = []
        self.grid = self.subject.school.grid
        self.agent_list = self.subject.agent_list

    def update(self) -> None:
        statistics =  {
            "num_healthy": self.subject.num_healthy,
            "num_infected": self.subject.num_infected,
            "num_zombie": self.subject.num_zombie,
            "num_dead": self.subject.num_dead,
            "population_size": self.subject.population_size,
            "infection_probability": self.subject.infection_probability,
            "turning_probability": self.subject.turning_probability,
            "death_probability": self.subject.death_probability,
            "migration_probability": self.subject.migration_probability,
        }
        self.statistics.append(deepcopy(statistics))
        self.grid = deepcopy(self.subject.school.grid)
        self.agent_list = deepcopy(self.subject.agent_list)
        

    def display_observation(self, format="statistics"):
        if format == "statistics":
            self.print_statistics_text()
        elif format == "grid":
            self.print_grid_text()
        elif format == "bar":
            self.print_bar_graph()
        elif format == "scatter":
            self.print_scatter_graph()

    def print_statistics_text(self):
        population_size = self.statistics[-1]["population_size"]
        num_healthy = self.statistics[-1]["num_healthy"]
        num_infected = self.statistics[-1]["num_infected"]
        num_zombie = self.statistics[-1]["num_zombie"]
        num_dead = self.statistics[-1]["num_dead"]
        healthy_percentage = num_healthy / (population_size + 1e-10)
        infected_percentage = num_infected / (population_size + 1e-10)
        zombie_percentage = num_zombie / (population_size + 1e-10)
        dead_percentage = num_dead / (population_size + 1e-10)
        infected_rate = num_infected / (num_healthy + 1e-10)
        turning_rate = num_zombie / (num_infected + 1e-10)
        death_rate = num_dead / (num_zombie + 1e-10)
        infection_probability = self.statistics[-1]["infection_probability"]
        turning_probability = self.statistics[-1]["turning_probability"]
        death_probability = self.statistics[-1]["death_probability"]
        migration_probability = self.statistics[-1]["migration_probability"]
        print("Population Statistics:")
        print(f"Population Size: {population_size}")
        print(f"Healthy: {num_healthy} ({healthy_percentage:.2%})")
        print(f"Infected: {num_infected} ({infected_percentage:.2%})")
        print(f"Zombie: {num_zombie} ({zombie_percentage:.2%})")
        print(f"Dead: {num_dead} ({dead_percentage:.2%})")
        print(f"Infection Probability: {infection_probability:.2%} -> Infected Rate: {infected_rate:.2%}")
        print(f"Turning Probability: {turning_probability:.2%} -> Turning Rate: {turning_rate:.2%}")
        print(f"Death Probability: {death_probability:.2%} -> Death Rate: {death_rate:.2%}")
        print(f"Migration Probability: {migration_probability:.2%}")
        
        # The mean can be used to calculate the average number of zombies that appear in a specific area over time. This can be useful for predicting the rate of zombie infection and determining the necessary resources needed to survive.
        mean = np.mean([d["num_zombie"] for d in self.statistics])
        # The median can be used to determine the middle value in a set of data. In a zombie apocalypse simulation, the median can be used to determine the number of days it takes for a specific area to become overrun with zombies.
        median = np.median([d["num_zombie"] for d in self.statistics])
        # The mode can be used to determine the most common value in a set of data. In a zombie apocalypse simulation, the mode can be used to determine the most common type of zombie encountered or the most effective weapon to use against them.
        mode = stats.mode([d["num_zombie"] for d in self.statistics], keepdims=True)[0][0]
        # The standard deviation can be used to determine how spread out a set of data is. In a zombie apocalypse simulation, the standard deviation can be used to determine the level of unpredictability in zombie behavior or the effectiveness of certain survival strategies.
        std = np.std([d["num_zombie"] for d in self.statistics])
        print(f"Mean of Number of Zombie: {mean}")
        print(f"Median of Number of Zombie: {median}")
        print(f"Mode of Number of Zombie: {mode}")
        print(f"Standard Deviation of Number of Zombie: {std}")
        print()

    """
    # may format the output in the subject and print it directly here
    
    def print_text_statistics(self):
        print("Population Statistics:")
        for key, value in self.statistics.items():
            print(f"{key}: {value}")
            
    # subject notify method push info to update method of observer
    # but the subject don't know what info observer want to display
    # or let observer pull info from subject using get method of subject
    # but observer need to get info from subject one by one
    """
    
    def print_grid_text(self):
        print("Print School:")
        state_symbols = {
            None: " ",
            HealthState.HEALTHY: "H",
            HealthState.INFECTED: "I",
            HealthState.ZOMBIE: "Z",
            HealthState.DEAD: "D",
        }

        for row in self.grid:
            for cell in row:
                try:
                    print(state_symbols[cell.health_state], end=" ")
                except AttributeError:
                    print(state_symbols[cell], end=" ")
            print()
        print()

    def print_bar_graph(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.set_ylim(0, self.statistics[0]["population_size"] + 1)

        # Get the counts of each state in the population
        cell_states = [individual.health_state for individual in self.agent_list]
        counts = {state: cell_states.count(state) for state in list(HealthState)}

        # create a colormap from the seaborn palette and the number of colors equal to the number of members in the State enum
        color_palette = sns.color_palette("deep", n_colors=len(HealthState))
        cmap = colors.ListedColormap(color_palette)

        # create a list of legend labels and colors for each state in the State enum
        handles = [patches.Patch(color=color, label=state.name) for color, state in zip(color_palette, HealthState)]

        ax.bar(
            np.asarray(HealthState.value_list()),
            list(counts.values()),
            tick_label=HealthState.name_list(),
            label=HealthState.name_list(),
            color=color_palette,
        )

        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), labels=HealthState.name_list())

        plt.show()

    def print_scatter_graph(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)

        x = np.array([individual.location[0] for individual in self.agent_list])
        y = np.array([individual.location[1] for individual in self.agent_list])
        cell_states_value = np.array([individual.health_state.value for individual in self.agent_list])
        
        # create a colormap from the seaborn palette and the number of colors equal to the number of members in the State enum
        color_palette = sns.color_palette("deep", n_colors=len(HealthState))
        cmap = colors.ListedColormap(color_palette)
        
        # create a list of legend labels and colors for each state in the State enum
        handles = [patches.Patch(color=color, label=state.name) for color, state in zip(color_palette, HealthState)]
        
        ax.scatter(x, y, c=cell_states_value, cmap=cmap)

        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), labels=HealthState.name_list())

        plt.show()


class SimulationAnimator(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.agent_history = []
        self.setup_animation()

    def setup_animation(self):
        # Shared setup for all animations
        sns.set_style("whitegrid")
        self.state_colors = sns.color_palette("deep", n_colors=len(HealthState))
        self.cmap = colors.ListedColormap(self.state_colors)
        self.state_handles = [patches.Patch(color=color, label=state.name) for color, state in zip(self.state_colors, HealthState)]

    def update(self) -> None:
        self.agent_history.append(deepcopy(self.subject.agent_list))

    def display_observation(self, format="bar"):
        if format == "bar":
            self.print_bar_animation()
        elif format == "scatter":
            self.print_scatter_animation()
        elif format == "table":
            self.print_table_animation()

    def print_bar_animation(self):
        counts = []
        for agent_list in self.agent_history:
            cell_states = [individual.health_state for individual in agent_list]
            counts.append([cell_states.count(state) for state in HealthState])

        self.bar_chart_animation(np.array(HealthState.value_list()), np.array(counts), HealthState.name_list())

    def bar_chart_animation(self, x, y, ticks):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Population Health States Over Time")
        ax.set_ylim(0, max(map(max, y)) + 1)

        bars = ax.bar(x, y[0], tick_label=ticks, label=HealthState.name_list(), color=self.state_colors)
        text_box = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def update(i):
            for j, bar in enumerate(bars):
                bar.set_height(y[i][j])
            text_box.set_text(f"Time Step: {i}")

        anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000, repeat=False)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_scatter_animation(self):
        x = [[individual.location[0] for individual in agent_list] for agent_list in self.agent_history]
        y = [[individual.location[1] for individual in agent_list] for agent_list in self.agent_history]
        cell_states_value = [[individual.health_state.value for individual in agent_list] for agent_list in self.agent_history]

        self.scatter_chart_animation(x, y, cell_states_value)

    def scatter_chart_animation(self, x, y, cell_states_value):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)

        sc = ax.scatter(x[0], y[0], c=cell_states_value[0], cmap=self.cmap)
        label = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def animate(i):
            sc.set_offsets(np.c_[x[i], y[i]])
            sc.set_array(cell_states_value[i])
            label.set_text(f"Time Step: {i}")

        anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=1000, repeat=False)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_table_animation(self):
        cell_states_name = [[individual.health_state for individual in agent_list] for agent_list in self.agent_history]
        x = [[individual.location[0] for individual in agent_list] for agent_list in self.agent_history]
        y = [[individual.location[1] for individual in agent_list] for agent_list in self.agent_history]

        # Build the grid
        cell_states = []
        for i in range(len(cell_states_name)):
            grid = [["" for _ in range(self.subject.school.size)] for _ in range(self.subject.school.size)]
            for j, individual in enumerate(cell_states_name[i]):
                grid[x[i][j]][y[i][j]] = individual.name
            cell_states.append(grid)

        self.table_animation(cell_states)

    def table_animation(self, cell_states):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_xlim(-1, len(cell_states[0])+1)
        ax.set_ylim(-1, len(cell_states[0])+1)
        ax.axis('off')

        # Map state to colors
        state_colors = {
            'HEALTHY': "green",
            'INFECTED': "orange",
            'ZOMBIE': "red",
            'DEAD': "black",
        }

        # Initialize table
        table = ax.table(cellText=cell_states[0], loc="center", bbox=Bbox.from_bounds(0, 0, 1, 1))
        label = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        # Adjust cell properties for centering text
        for key, cell in table.get_celld().items():
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')

        def animate(i):
            for row_num, row in enumerate(cell_states[i]):
                for col_num, cell_value in enumerate(row):
                    cell_color = state_colors.get(cell_value, "white")
                    table[row_num, col_num].set_facecolor(cell_color)
                    table[row_num, col_num].get_text().set_text(cell_value)
            label.set_text(f"Time Step: {i}")
            return table, label

        anim = animation.FuncAnimation(fig, animate, frames=len(cell_states), interval=1000, repeat=False, blit=True)
        plt.show()

class MatplotlibAnimator(Observer):
    def __init__(self, population: Population, mode: str = "scatter"):
        """Initialize the animator."""
        self.subject = population
        self.subject.attach_observer(self)
        self.mode = mode  # "bar" or "scatter" or "table"
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 7),  constrained_layout=True)

        self.cell_states = [individual.health_state for individual in self.subject.agent_list]
        self.cell_states_value = [state.value for state in self.cell_states]
        self.cell_x_coords = [individual.location[0] for individual in self.subject.agent_list]
        self.cell_y_coords = [individual.location[1] for individual in self.subject.agent_list]

        if self.mode == "bar":
            self.setup_bar_chart()
        elif self.mode == "scatter":
            self.setup_scatter_plot()
        elif self.mode == "table":
            self.setup_table()

    def setup_bar_chart(self):
        self.ax.set_title("Bar Chart Animation")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_ylim(0, len(self.subject.agent_list) + 1)
        self.setup_initial_bar_state()

    def setup_scatter_plot(self):
        self.ax.set_title("Scatter Chart Animation")
        self.ax.set_xlim(-1, self.subject.school.size + 1)
        self.ax.set_ylim(-1, self.subject.school.size + 1)
        self.setup_initial_scatter_state()

    def setup_initial_bar_state(self):
        counts = [self.cell_states.count(state) for state in list(HealthState)]
        self.bars = self.ax.bar(np.array(HealthState.value_list()), counts, tick_label=HealthState.name_list(), 
                                label=HealthState.name_list(), color=sns.color_palette("deep"))
        self.ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def setup_initial_scatter_state(self):
        color_palette = sns.color_palette("deep", n_colors=len(HealthState))
        cmap=colors.ListedColormap(color_palette)
        handles = [patches.Patch(color=color, label=state.name) for color, state in zip(color_palette, HealthState)]
        self.scatter = self.ax.scatter(self.cell_x_coords, self.cell_y_coords, c=self.cell_states_value, cmap=cmap)
        self.ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), labels=HealthState.name_list())
        plt.draw()
        
    def setup_table(self):
        self.ax.set_title("Table Animation")
        self.ax.set_xlim(-1, self.subject.school.size + 1)
        self.ax.set_ylim(-1, self.subject.school.size + 1)
        self.setup_initial_table_state()
        
    def setup_initial_table_state(self):
        cell_states = [["" for _ in range(self.subject.school.size)] for _ in range(self.subject.school.size)]
        for j in range(self.subject.school.size):
            cell_states[self.cell_x_coords[j]][self.cell_y_coords[j]] = self.cell_states[j].name

        state_colors = {
            HealthState.HEALTHY.name: "green",
            HealthState.INFECTED.name: "orange",
            HealthState.ZOMBIE.name: "red",
            HealthState.DEAD.name: "black",
        }

        self.table = self.ax.table(cellText=np.array(cell_states), loc="center", bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0))

        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                cell_state = cell_states[i][j]
                color = state_colors.get(cell_state, "white")  # Default to white if state is unknown
                self.table[i, j].set_facecolor(color)
                self.table[i, j].get_text().set_text('')  # Clear the text

        plt.draw()

    def update(self) -> None:
        self.cell_states = [individual.health_state for individual in self.subject.agent_list]
        self.cell_states_value = [state.value for state in self.cell_states]
        self.cell_x_coords = [individual.location[0] for individual in self.subject.agent_list]
        self.cell_y_coords = [individual.location[1] for individual in self.subject.agent_list]
        
        if self.mode == "bar":
            self.update_bar_chart()
        elif self.mode == "scatter":
            self.update_scatter_plot()
        elif self.mode == "table":
            self.update_table()

    def update_bar_chart(self):
        counts = [self.cell_states.count(state) for state in list(HealthState)]
        for bar, count in zip(self.bars, counts):
            bar.set_height(count)
        plt.draw()
        plt.pause(0.5)

    def update_scatter_plot(self):
        self.scatter.set_offsets(np.c_[self.cell_x_coords, self.cell_y_coords])
        self.scatter.set_array(np.array(self.cell_states_value))
        plt.draw()
        plt.pause(0.5)

    def update_table(self):
        cell_states = [["" for _ in range(self.subject.school.size)] for _ in range(self.subject.school.size)]
        for j, individual in enumerate(self.subject.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        state_colors = {
            HealthState.HEALTHY.name: "green",
            HealthState.INFECTED.name: "orange",
            HealthState.ZOMBIE.name: "red",
            HealthState.DEAD.name: "black",
        }

        for (i, j), cell in np.ndenumerate(cell_states):
            cell_state = cell_states[i][j]
            color = state_colors.get(cell_state, "white")  # Default to white if state is unknown
            self.table[i, j].set_facecolor(color)
            self.table[i, j].get_text().set_text(cell_state)
            self.table[i, j].get_text().set_horizontalalignment('center')
            self.table[i, j].get_text().set_verticalalignment('center')

        plt.draw()
        plt.pause(0.5)

    def display_observation(self):
        plt.show()


class TkinterObserver(Observer):
    def __init__(self, population, grid_size=300, cell_size=30):
        self.population = population
        self.population.attach_observer(self)

        # Define the size of the grid and cells
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_cells = self.population.school.school_size

        # Initialize Tkinter window
        self.root = tk.Tk()
        self.root.title("Zombie Apocalypse Simulation")

        # Create canvas for drawing the grid
        self.canvas = tk.Canvas(self.root, width=self.grid_size, height=self.grid_size)
        self.canvas.pack()

        # Draw the initial state of the grid
        self.update()

    def update(self):
        # Update the canvas with the new simulation state
        self.draw_grid()
        self.tkinter_mainloop()
        time.sleep(0.5)

    def draw_grid(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Draw the grid based on the current state of the simulation
        for individual in self.population.agent_list:
            x, y = individual.location
            x1 = x * self.cell_size
            y1 = y * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            # Choose color based on the individual's state
            color = "white"
            if individual.state == HealthState.HEALTHY:
                color = "green"
            elif individual.state == HealthState.INFECTED:
                color = "yellow"
            elif individual.state == HealthState.ZOMBIE:
                color = "red"
            elif individual.state == HealthState.DEAD:
                color = "gray"

            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def tkinter_mainloop(self):
        """ Run a single iteration of the Tkinter main loop. """
        self.root.update_idletasks()
        self.root.update()

    def display_observation(self):
        """ Display the final plot. """
        self.root.mainloop()


class PopulationObserver(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.grid_history = []

    def update(self) -> None:
        current_grid_state = self.capture_grid_state()
        self.grid_history.append(current_grid_state)

    def capture_grid_state(self):
        grid_state = np.zeros((self.subject.school.size, self.subject.school.size))
        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                individual = self.subject.school.get_individual((i, j))
                grid_state[i, j] = individual.health_state.value if individual else 0
        return grid_state

    def prepare_data(self, N):
        X, y = [], []
        for i in range(N, len(self.grid_history)):
            X.append(np.array(self.grid_history[i-N:i]))
            y.append(np.array(self.grid_history[i]))
        return np.array(X), np.array(y)

    def train_model(self):
        N = 5
        X, y = self.prepare_data(N)

        # Reshape for ConvLSTM input
        X = X.reshape((-1, X.shape[1], X.shape[2], X.shape[3], 1))
        y = y.reshape((-1, y.shape[1], y.shape[2], 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # ConvLSTM model
        model = models.Sequential([
            layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True, input_shape=X_train.shape[1:]),
            layers.Dropout(0.2),
            layers.LayerNormalization(),
            layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=False),
            layers.Dropout(0.2),
            layers.LayerNormalization(),
            layers.Conv2D(filters=1, kernel_size=(3, 3), activation='linear', padding='same')
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, validation_split=0.2)

        self.model = model

        # Evaluate the model
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test.reshape(y_test.shape[0], -1), predictions.reshape(predictions.shape[0], -1))
        print(f"Model Mean Squared Error: {mse}")

    def display_observation(self):
        if getattr(self, "model", None) is None:
            self.train_model()
        past_grid_state = self.grid_history[-5:]
        print(self.grid_history[-1])
        input_data = np.array(past_grid_state).reshape((1, -1, self.subject.school.size, self.subject.school.size, 1))
        predicted_grid_state = self.model.predict(input_data)
        reshaped_grid_state = predicted_grid_state.reshape((self.subject.school.size, self.subject.school.size))
        reformatted_grid_state = np.round(reshaped_grid_state, 1)
        print(reformatted_grid_state)


def main():

    # create a SchoolZombieApocalypse object
    school_sim = Population(school_size=10, population_size=10)

    # create Observer objects
    simulation_observer = SimulationObserver(school_sim)
    # simulation_animator = SimulationAnimator(school_sim)
    # matplotlib_animator = MatplotlibAnimator(school_sim, mode="table") # "bar" or "scatter" or "table"
    # tkinter_observer = TkinterObserver(school_sim)
    # population_observer = PopulationObserver(school_sim)

    # run the population for a given time period
    school_sim.run_population(num_time_steps=10)
    
    print("Observers:")
    # print(simulation_observer.agent_list)
    # print(simulation_animator.agent_history[-1])

    # observe the statistics of the population
    simulation_observer.display_observation(format="grid") # "statistics" or "grid" or "bar" or "scatter"
    # simulation_animator.display_observation(format="table") # "bar" or "scatter" or "table"
    # matplotlib_animator.display_observation()
    # tkinter_observer.display_observation()
    # population_observer.display_observation()



if __name__ == "__main__":
    main()



