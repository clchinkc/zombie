"""
The implementation of a zombie apocalypse simulation in a school environment requires the creation of several classes and functions to accurately represent the dynamics of such a scenario. Let's break down the implementation into key components and their respective functionalities:

### 1. Person Class
- **Attributes:** Location, state (alive, undead, escaped), health, weapons, and supplies.
- **Methods:** Movement across the grid, interactions with other entities (people, zombies).

### 2. Zombie Class
- **Attributes:** Similar to Person class, with additional zombie-specific behaviors (attacking, infecting).
- **Methods:** Movement, attacking living people, spreading infection.

### 3. School Class
- **Attributes:** Layout represented by a 2D grid, tracking of people and zombies.
- **Methods:** Movement and state updates for people and zombies, interaction handling.

### 4. Simulation Function
- **Functionality:** Initializes simulation conditions (layout, entities, resources), runs the simulation for a specified number of steps, updates and tracks the simulation's progress.

### 5. Additional Components
- **State Machine Pattern:** For managing the state transitions of individuals (healthy, infected, zombie, dead).
- **Movement Strategies:** Different strategies for movement (random, flee zombies, chase humans) depending on the state of the individual.
- **Grid Management:** Functions to handle legal movements, neighbor detection, and interactions within the grid.
- **Statistical Tracking:** To keep track of population metrics (counts of healthy, infected, zombies, dead).

### 6. Visualization and Analysis
- Enhance visualization for better insights. This could include detailed graphs, color-coding, and interactive elements for in-depth exploration.

### Implementation Overview
- **Language & Libraries:** Python with libraries like NumPy, Matplotlib, Seaborn, and Keras.
- **UI Frameworks:** Tkinter for basic GUI elements.
- **Machine Learning:** For predicting future states of the simulation.
- **Observer Pattern:** To track and update simulation states and render visualizations.

### Example Workflow
1. **Initialize:** Set up the school grid and populate it with individuals.
2. **Run Simulation:** Iterate over time steps, updating states and movements of individuals.
3. **Apply Strategies:** Based on individual states, apply movement and interaction strategies.
4. **State Management:** Update states (healthy, infected, zombie) using the state machine pattern.
5. **Visualization:** Use graphical tools to visualize the simulation progress and outcomes.
6. **Analysis:** Apply machine learning for predictive analysis and simulation insights.

This structured approach allows for a comprehensive simulation of a zombie apocalypse in a school setting, with detailed tracking and analysis of the evolving scenario.
"""

from __future__ import annotations

import math
import os
import random
import time
import tkinter as tk
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from itertools import product
from typing import Any, Optional

import keras
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pygame
import scipy
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras import layers
from matplotlib import animation, colors, patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.table import Table
from matplotlib.transforms import Bbox
from numpy.fft import fft, fft2, fftshift
from plotly.subplots import make_subplots
from scipy import stats
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import ParameterGrid, ShuffleSplit, train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        return f"{self.__class__.__name__}({self.id}, {self.health_state}, {self.location})"


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
            self.timestep = time + 1
            print("Time step: ", self.timestep)
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

        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))
        self.state_handles = [patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()]

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
        elif format == "table":
            self.print_table_graph()

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
        ax.set_title("Bar Chart")
        ax.set_ylim(0, self.statistics[0]["population_size"] + 1)

        # Use common state_colors
        cell_states = [individual.health_state for individual in self.agent_list]
        counts = {state: cell_states.count(state) for state in HealthState}
        ax.bar(
            np.asarray(HealthState.value_list()),
            list(counts.values()),
            tick_label=HealthState.name_list(),
            label=HealthState.name_list(),
            color=[self.state_colors[state] for state in HealthState]
        )

        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_scatter_graph(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.set_title("Scatter Chart")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)

        x = np.array([individual.location[0] for individual in self.agent_list])
        y = np.array([individual.location[1] for individual in self.agent_list])
        cell_states_value = np.array([individual.health_state.value for individual in self.agent_list])

        # Use common cmap
        ax.scatter(x, y, c=cell_states_value, cmap=self.cmap)

        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_table_graph(self):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Table Chart")
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.axis('off')

        # Initialize table with common state_colors
        cell_states = [["" for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        for j, individual in enumerate(self.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        table = ax.table(cellText=np.array(cell_states), loc="center", bbox=Bbox.from_bounds(0, 0, 1, 1))

        # Adjust cell properties using common state_colors
        for key, cell in table.get_celld().items():
            cell_state = cell_states[key[0]][key[1]]
            cell.set_facecolor(self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white")
            cell.get_text().set_text(cell_state)
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
        
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


class SimulationAnimator(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.agent_history = []

        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))
        self.state_handles = [patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()]

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
        ax.set_title("Bar Chart Animation")
        ax.set_ylim(0, max(map(max, y)) + 1)

        bars = ax.bar(x, y[0], tick_label=ticks, label=HealthState.name_list(), color=self.state_colors.values())
        text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        def update(i):
            for j, bar in enumerate(bars):
                bar.set_height(y[i][j])
            text_box.set_text(f"Time Step: {i+1}")

        anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000, repeat=False)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    def print_scatter_animation(self):
        x = [[individual.location[0] for individual in agent_list] for agent_list in self.agent_history]
        y = [[self.subject.school.size - individual.location[0] - 1 for individual in agent_list] for agent_list in self.agent_history]
        cell_states_value = [[individual.health_state.value for individual in agent_list] for agent_list in self.agent_history]

        self.scatter_chart_animation(x, y, cell_states_value)

    def scatter_chart_animation(self, x, y, cell_states_value):
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_title("Scatter Chart Animation")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)

        sc = ax.scatter(x[0], y[0], c=cell_states_value[0], cmap=self.cmap)
        text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        def animate(i):
            sc.set_offsets(np.c_[x[i], y[i]])
            sc.set_array(cell_states_value[i])
            text_box.set_text(f"Time Step: {i+1}")

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
        ax.set_title("Table Animation")
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.axis('off')

        # Initialize table
        table = ax.table(cellText=cell_states[0], loc="center", bbox=Bbox.from_bounds(0, 0, 1, 1))
        text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        # Adjust cell properties for centering text
        for key, cell in table.get_celld().items():
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')

        def animate(i):
            for row_num, row in enumerate(cell_states[i]):
                for col_num, cell_value in enumerate(row):
                    cell_color = self.state_colors.get(HealthState[cell_value], "white") if cell_value else "white"
                    table[row_num, col_num].set_facecolor(cell_color)
                    table[row_num, col_num].get_text().set_text(cell_value)
            text_box.set_text(f"Time Step: {i+1}")
            return table, text_box

        anim = animation.FuncAnimation(fig, animate, frames=len(cell_states), interval=1000, repeat=False, blit=True)
        plt.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


class PlotlyAnimator(Observer):
    def __init__(self, population: Population):
        self.subject = population
        self.subject.attach_observer(self)
        self.data_history = []

    def update(self):
        current_state = self.capture_current_state()
        self.data_history.append(current_state)

    def capture_current_state(self):
        data = [{'x': ind.location[0], 'y': ind.location[1], 'z': 0, 'state': ind.health_state.name} for ind in self.subject.agent_list]
        return pd.DataFrame(data)

    def display_observation(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Scatter Plot", "Heatmap", "Time Series", "3D Scatter Plot"),
                            specs=[[{"type": "scatter"}, {"type": "heatmap"}], [{"type": "scatter"}, {"type": "scatter3d"}]])

        self.add_scatter_plot(fig, row=1, col=1)
        self.add_heatmap(fig, row=1, col=2)
        self.add_time_series(fig, row=2, col=1)
        self.add_3d_scatter(fig, row=2, col=2)

        fig.update_layout(height=800, width=1200, title_text="Zombie Apocalypse Simulation",
                          legend_title="Health States", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.show()

    def add_scatter_plot(self, fig, row, col):
        scatter_data = self.data_history[-1]
        scatter_plot = px.scatter(scatter_data, x="x", y="y", color="state")
        for trace in scatter_plot.data:
            fig.add_trace(trace, row=row, col=col)

    def add_heatmap(self, fig, row, col):
        heatmap_data = self.data_history[-1].pivot_table(index='y', columns='x', aggfunc='size', fill_value=0)
        fig.add_trace(go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'), row=row, col=col)

    def add_time_series(self, fig, row, col):
        time_series_data = self.prepare_time_series_data()
        time_series_plot = px.line(time_series_data, x="time_step", y="counts", color='state')
        for trace in time_series_plot.data:
            fig.add_trace(trace, row=row, col=col)

    def add_3d_scatter(self, fig, row, col):
        scatter_data = self.data_history[-1]
        scatter_3d = px.scatter_3d(scatter_data, x="x", y="y", z="z", color="state")
        for trace in scatter_3d.data:
            fig.add_trace(trace, row=row, col=col)

    def prepare_time_series_data(self):
        all_states = ['HEALTHY', 'INFECTED', 'ZOMBIE', 'DEAD']
        all_combinations = pd.MultiIndex.from_product([range(len(self.data_history)), all_states], names=['time_step', 'state']).to_frame(index=False)

        time_series_data = pd.concat([data['state'].value_counts().rename_axis('state').reset_index(name='counts').assign(time_step=index) for index, data in enumerate(self.data_history)], ignore_index=True)
        
        return pd.merge(all_combinations, time_series_data, on=['time_step', 'state'], how='left').fillna(0)


class MatplotlibAnimator(Observer):
    def __init__(self, population: Population, plot_order=["bar", "scatter", "table"]):
        """Initialize the animator with customizable plot order."""
        self.subject = population
        self.subject.attach_observer(self)
        self.plot_order = plot_order

        # Initialize matplotlib figure with three subplots
        self.fig, self.axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)

        # Initialize common elements for the plots
        self.init_common_elements()

        # Setup each subplot based on the specified order
        for i, plot_type in enumerate(self.plot_order):
            if plot_type == "bar":
                self.setup_bar_chart(self.axes[i])
            elif plot_type == "scatter":
                self.setup_scatter_plot(self.axes[i])
            elif plot_type == "table":
                self.setup_table(self.axes[i])

    def init_common_elements(self):
        # Common elements initialization
        self.cell_states = [individual.health_state for individual in self.subject.agent_list]
        self.cell_states_value = [state.value for state in self.cell_states]
        self.cell_x_coords = [individual.location[0] for individual in self.subject.agent_list]
        self.cell_y_coords = [individual.location[1] for individual in self.subject.agent_list]

        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))
        self.state_handles = [patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()]

    # Setup methods for each plot type
    def setup_bar_chart(self, ax):
        ax.set_title("Bar Chart")
        ax.set_ylim(0, len(self.subject.agent_list) + 1)
        self.setup_initial_bar_state(ax)

    def setup_scatter_plot(self, ax):
        ax.set_title("Scatter Plot")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)
        self.setup_initial_scatter_state(ax)

    def setup_table(self, ax):
        ax.set_title("Table")
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.axis('off')
        self.setup_initial_table_state(ax)

    # Methods for setting up initial states of each plot type
    def setup_initial_bar_state(self, ax):
        counts = [self.cell_states.count(state) for state in list(HealthState)]
        self.bars = ax.bar(np.array(HealthState.value_list()), counts, tick_label=HealthState.name_list(), 
                                label=HealthState.name_list(), color=[self.state_colors[state] for state in HealthState])
        self.bar_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def setup_initial_scatter_state(self, ax):
        transformed_x_coords = [y for y in self.cell_y_coords]
        transformed_y_coords = [self.subject.school.size - x - 1 for x in self.cell_x_coords]

        self.scatter = ax.scatter(transformed_x_coords, transformed_y_coords, 
                                c=self.cell_states_value, cmap=self.cmap)
        self.scatter_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def setup_initial_table_state(self, ax):
        cell_states = [["" for _ in range(self.subject.school.size)] 
                        for _ in range(self.subject.school.size)]
        for j, individual in enumerate(self.subject.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        self.table = ax.table(cellText=np.array(cell_states), loc="center", bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0))
        self.table_text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes)

        # Adjust cell properties for centering text
        for key, cell in self.table.get_celld().items():
            cell.set_height(1 / len(cell_states[0]))
            cell.set_width(1 / len(cell_states[0]))
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')

        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                cell_state = cell_states[i][j]
                color = self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white"
                self.table[i, j].set_facecolor(color)
                self.table[i, j].get_text().set_text(cell_state)
        ax.legend(handles=self.state_handles, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.draw()

    def display_observation(self):
        plt.show()

    def update(self):
        # Update the elements common to all plots
        self.update_common_elements()

        # Update each subplot based on its type
        for i, plot_type in enumerate(self.plot_order):
            if plot_type == "bar":
                self.update_bar_chart(self.axes[i])
            elif plot_type == "scatter":
                self.update_scatter_plot(self.axes[i])
            elif plot_type == "table":
                self.update_table(self.axes[i])

    def update_common_elements(self):
        self.cell_states = [individual.health_state for individual in self.subject.agent_list]
        self.cell_states_value = [state.value for state in self.cell_states]
        self.cell_x_coords = [individual.location[0] for individual in self.subject.agent_list]
        self.cell_y_coords = [individual.location[1] for individual in self.subject.agent_list]

    # Update methods for each plot type
    def update_bar_chart(self, ax):
        counts = [self.cell_states.count(state) for state in HealthState]
        for bar, count in zip(self.bars, counts):
            bar.set_height(count)
        self.bar_text_box.set_text(f"Time Step: {self.subject.timestep}")
        plt.draw()
        plt.pause(0.5)

    def update_scatter_plot(self, ax):
        transformed_x_coords = [self.cell_y_coords[i] for i in range(len(self.cell_x_coords))]
        transformed_y_coords = [self.subject.school.size - self.cell_x_coords[i] - 1 for i in range(len(self.cell_y_coords))]

        self.scatter.set_offsets(np.c_[transformed_x_coords, transformed_y_coords])
        self.scatter.set_array(np.array(self.cell_states_value))
        self.scatter_text_box.set_text(f"Time Step: {self.subject.timestep}")
        plt.draw()
        plt.pause(0.5)

    def update_table(self, ax):
        cell_states = [["" for _ in range(self.subject.school.size)] 
                        for _ in range(self.subject.school.size)]
        for j, individual in enumerate(self.subject.agent_list):
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name

        for (i, j), cell in np.ndenumerate(cell_states):
            cell_state = cell_states[i][j]
            color = self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white"
            self.table[i, j].set_facecolor(color)
            self.table[i, j].get_text().set_text(cell_state)
        self.table_text_box.set_text(f"Time Step: {self.subject.timestep}")
        plt.draw()
        plt.pause(0.5)


class TkinterObserver(Observer):
    def __init__(self, population, grid_size=300, cell_size=30):
        self.subject = population
        self.subject.attach_observer(self)

        # Define grid and cell sizes
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_cells = self.subject.school.size

        # Initialize Tkinter window
        self.root = tk.Tk()
        self.root.title("Zombie Apocalypse Simulation")

        # Canvas for the simulation grid
        self.grid_canvas = tk.Canvas(self.root, width=self.grid_size, height=self.grid_size)
        self.grid_canvas.pack(side=tk.LEFT)

        # Frame for Matplotlib plots
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup common elements and plots
        self.init_common_elements()
        self.setup_plots()

        self.update()  # Initial update

    def init_common_elements(self):
        sns.set_style("whitegrid")
        sns.set_context("paper")
        deep_colors = sns.color_palette("deep")
        self.state_colors = {
            HealthState.HEALTHY: deep_colors[0],
            HealthState.INFECTED: deep_colors[1],
            HealthState.ZOMBIE: deep_colors[2],
            HealthState.DEAD: deep_colors[3],
        }
        self.cmap = colors.ListedColormap(list(self.state_colors.values()))

    def setup_plots(self):
        self.figures = {
            'bar': Figure(figsize=(7, 7), constrained_layout=True),
            'scatter': Figure(figsize=(7, 7), constrained_layout=True),
            'table': Figure(figsize=(7, 7), constrained_layout=True)
        }

        # Initial setup for each plot
        self.setup_initial_bar_state(self.figures['bar'].add_subplot(111))
        self.setup_initial_scatter_state(self.figures['scatter'].add_subplot(111))
        self.setup_initial_table_state(self.figures['table'].add_subplot(111))

        # Using grid layout manager instead of pack
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_rowconfigure(2, weight=1)

        self.canvases = {}
        for i, (plot_type, fig) in enumerate(self.figures.items()):
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvases[plot_type] = canvas
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=i, column=0, sticky="nsew")

        # Ensure the grid_canvas is positioned correctly
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def update(self):
        self.draw_grid()
        self.update_plots()
        self.root.update_idletasks()
        self.root.update()
        time.sleep(0.5)

    def draw_grid(self):
        self.grid_canvas.delete("all")
        for individual in self.subject.agent_list:
            x, y = individual.location
            canvas_x = y
            canvas_y = x
            x1, y1 = canvas_x * self.cell_size, canvas_y * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            color = self.get_color(individual.health_state)
            self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def update_plots(self):
        self.update_bar_chart()
        self.update_scatter_plot()
        self.update_table_plot()

    # Bar chart setup and update
    def setup_initial_bar_state(self, ax):
        ax.set_title("Bar Chart")
        ax.set_ylim(0, len(self.subject.agent_list) + 1)
        ax.legend(handles=[patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()], 
                    loc="center left", bbox_to_anchor=(1, 0.5))

    def update_bar_chart(self):
        ax = self.figures['bar'].gca()
        ax.clear()
        self.setup_initial_bar_state(ax)
        cell_states = [individual.health_state for individual in self.subject.agent_list]
        counts = {state: cell_states.count(state) for state in HealthState}
        heights = np.array([counts.get(state, 0) for state in HealthState])
        ax.bar(np.arange(len(HealthState)), heights, tick_label=[state.name for state in HealthState], color=[self.state_colors[state] for state in HealthState])
        self.canvases['bar'].draw()

    # Scatter plot setup and update
    def setup_initial_scatter_state(self, ax):
        ax.set_title("Scatter Plot")
        ax.set_xlim(-1, self.subject.school.size)
        ax.set_ylim(-1, self.subject.school.size)
        ax.legend(handles=[patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()], 
                    loc="center left", bbox_to_anchor=(1, 0.5))

    def update_scatter_plot(self):
        ax = self.figures['scatter'].gca()
        ax.clear()
        self.setup_initial_scatter_state(ax)
        x = [individual.location[1] for individual in self.subject.agent_list]
        y = [individual.location[0] for individual in self.subject.agent_list]
        cell_states_value = [individual.health_state.value for individual in self.subject.agent_list]
        ax.scatter(x, y, c=cell_states_value, cmap=self.cmap)
        self.canvases['scatter'].draw()

    # Table plot setup and update
    def setup_initial_table_state(self, ax):
        ax.set_title("Table")
        ax.axis('tight')
        ax.axis('off')
        ax.set_xlim(-1, self.subject.school.size + 1)
        ax.set_ylim(-1, self.subject.school.size + 1)
        ax.legend(handles=[patches.Patch(color=color, label=state.name) for state, color in self.state_colors.items()], 
                    loc="center left", bbox_to_anchor=(1, 0.5))

    def update_table_plot(self):
        ax = self.figures['table'].gca()
        ax.clear()
        self.setup_initial_table_state(ax)
        cell_states = [["" for _ in range(self.subject.school.size)] for _ in range(self.subject.school.size)]
        for individual in self.subject.agent_list:
            cell_states[individual.location[0]][individual.location[1]] = individual.health_state.name
        table = ax.table(cellText=cell_states, loc="center")
        for key, cell in table.get_celld().items():
            cell_state = cell_states[key[0]][key[1]]
            cell.set_facecolor(self.state_colors.get(HealthState[cell_state], "white") if cell_state else "white")
            cell.get_text().set_text(cell_state)
        self.canvases['table'].draw()

    def get_color(self, health_state):
        rgb_color = self.state_colors.get(health_state, (1, 1, 1))  # Default white
        return f"#{int(rgb_color[0]*255):02x}{int(rgb_color[1]*255):02x}{int(rgb_color[2]*255):02x}"

    def display_observation(self):
        self.root.mainloop()


class PredictionObserver(Observer):
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
        for i in range(N, len(self.grid_history[:-1])):
            X.append(np.array([keras.utils.to_categorical(frame, num_classes=4) for frame in self.grid_history[i - N:i]]))
            y.append(np.array(keras.utils.to_categorical(self.grid_history[i], num_classes=4)))
        return np.array(X), np.array(y)

    def augment_data(self, X, y):
        augmented_X, augmented_y = self.basic_augmentation(X, y)
        modified_X, modified_y = self.advanced_augmentation(augmented_X, augmented_y)
        return augmented_X, augmented_y, modified_X, modified_y

    def basic_augmentation(self, X, y):
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            # Original data
            augmented_X.append(X[i])
            augmented_y.append(y[i])

            # Horizontal flip
            X_hor_flip = np.flip(X[i], axis=1)
            y_hor_flip = np.flip(y[i], axis=0)
            augmented_X.append(X_hor_flip)
            augmented_y.append(y_hor_flip)

            # Vertical flip
            X_ver_flip = np.flip(X[i], axis=2)
            y_ver_flip = np.flip(y[i], axis=1)
            augmented_X.append(X_ver_flip)
            augmented_y.append(y_ver_flip)

            # Rotations 90, 180, 270 degrees
            for angle in [90, 180, 270]:
                X_rotated = scipy.ndimage.rotate(X[i], angle, axes=(1, 2), reshape=False)
                y_rotated = scipy.ndimage.rotate(y[i], angle, axes=(0, 1), reshape=False)
                augmented_X.append(X_rotated)
                augmented_y.append(y_rotated)
                
            # Translations
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                X_translated = np.roll(X[i], (dx, dy), axis=(1, 2))
                y_translated = np.roll(y[i], (dx, dy), axis=(0, 1))
                augmented_X.append(X_translated)
                augmented_y.append(y_translated)

        augmented_X = np.array(augmented_X)
        augmented_y = np.array(augmented_y)
        return augmented_X, augmented_y

    def advanced_augmentation(self, augmented_X, augmented_y, modification_rate=0.01, noise_level=0.01, time_distortion_weights=(0.7, 0.2, 0.1), warping_strength=0.1, num_agents_to_move=1):
        modified_X = []
        modified_y = []

        for i in range(len(augmented_X)):
            # Cell Type Modification Augmentation
            X_cell_mod = np.copy(augmented_X[i])
            num_modifications = int(modification_rate * X_cell_mod.size)
            indices = np.random.choice(X_cell_mod.size, num_modifications, replace=False)
            new_values = np.random.choice(np.array([0, 1, 2, 3]), num_modifications)
            np.put(X_cell_mod, indices, new_values)
            modified_X.append(X_cell_mod)
            modified_y.append(augmented_y[i])

            # Jittering Augmentation
            noise = noise_level * np.random.randn(*augmented_X[i].shape)
            X_noise_inject = augmented_X[i] + noise
            X_noise_inject = np.clip(X_noise_inject, 0, 3)  # Ensure the noisy data is within valid range
            modified_X.append(X_noise_inject)
            modified_y.append(augmented_y[i])

            # Time-Distortion Augmentation
            X_time_distort = np.copy(augmented_X[i])
            num_weights = len(time_distortion_weights)
            for t in range(num_weights - 1, augmented_X[i].shape[0]):
                X_time_distort[t] = sum(time_distortion_weights[j] * X_time_distort[t - j] for j in range(num_weights))
            modified_X.append(X_time_distort)
            modified_y.append(augmented_y[i])
            
            # Time Warping Augmentation
            num_knots = max(int(warping_strength * augmented_X[i].shape[1]), 2)
            seq_length = augmented_X.shape[1]
            original_indices = np.linspace(0, seq_length - 1, seq_length)
            knot_positions = np.linspace(0, seq_length - 1, num=num_knots, dtype=int)
            knot_offsets = warping_strength * np.random.randn(num_knots)
            warp_indices = np.clip(knot_positions + knot_offsets, 0, seq_length - 1)
            warp_function = interp1d(knot_positions, warp_indices, kind='linear', bounds_error=False)
            new_indices = warp_function(original_indices)
            new_indices = np.clip(new_indices, 0, seq_length - 1)
            signal = augmented_X[i]
            interp_function = interp1d(original_indices, signal, axis=0, kind='linear', bounds_error=False)
            warped_signal = interp_function(new_indices)
            warped_signal = np.clip(warped_signal, 0, 3)

            modified_X.append(warped_signal)
            modified_y.append(augmented_y[i])

            # Move a specified number of agents augmentation
            X_moved = np.copy(augmented_X[i])
            y_moved = np.copy(augmented_y[i])
            width, height = X_moved.shape[1], X_moved.shape[2]

            # Identify agent positions in both X and y
            agent_positions = np.argwhere(np.any(X_moved > 0, axis=-1) & np.any(y_moved > 0, axis=-1))
            np.random.shuffle(agent_positions)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                moved_agents = 0

                for t, x, y in agent_positions:
                    if moved_agents >= num_agents_to_move:
                        break

                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < width and 0 <= new_y < height:
                        # Check the validity of the new position
                        if not np.any(X_moved[:, max(0, new_x - 1):min(width, new_x + 2), max(0, new_y - 1):min(height, new_y + 2), :]) and not np.any(y_moved[new_x, new_y, :]):
                            # Move the agent across all timesteps
                            X_moved[:, x, y, :], X_moved[:, new_x, new_y, :] = X_moved[:, new_x, new_y, :], X_moved[:, x, y, :]
                            y_moved[x, y, :], y_moved[new_x, new_y, :] = y_moved[new_x, new_y, :], y_moved[x, y, :]
                            moved_agents += 1

            modified_X.append(X_moved)
            modified_y.append(y_moved)

        return np.array(modified_X), np.array(modified_y)

    def bootstrap_samples(self, basic_X, basic_y, advanced_X, advanced_y, n_basic_samples, n_advanced_samples):
        # Function to compute a representation score for each sample
        def compute_representation_score(y):
            # Assuming y is one-hot encoded, shape: (samples, width, height, classes)
            # Sum over width and height dimensions for each class
            state_presence = np.sum(y, axis=(1, 2))
            # Ignore state 0 (majority state) and sum the presence of all other states
            representation_score = np.sum(state_presence[:, 1:], axis=1)
            return representation_score

        # Compute representation scores for basic and advanced data
        basic_y_scores = compute_representation_score(basic_y)
        advanced_y_scores = compute_representation_score(advanced_y)

        # Convert scores to categorical strata (e.g., 'low', 'medium', 'high')
        # Adjust the quantile thresholds as per your specific data distribution
        basic_y_strata = pd.qcut(pd.Series(basic_y_scores), q=[0, .33, .66, 1], labels=False, duplicates='drop').fillna(0)
        advanced_y_strata = pd.qcut(pd.Series(advanced_y_scores), q=[0, .33, .66, 1], labels=False, duplicates='drop').fillna(0)

        # Stratified resampling
        bootstrapped_basic_X, bootstrapped_basic_y = resample(basic_X, basic_y, n_samples=n_basic_samples, replace=True, stratify=basic_y_strata)
        bootstrapped_advanced_X, bootstrapped_advanced_y = resample(advanced_X, advanced_y, n_samples=n_advanced_samples, replace=True, stratify=advanced_y_strata)
        
        bootstrapped_X = np.concatenate((bootstrapped_basic_X, bootstrapped_advanced_X), axis=0)
        bootstrapped_y = np.concatenate((bootstrapped_basic_y, bootstrapped_advanced_y), axis=0)
        
        return bootstrapped_X, bootstrapped_y

    def create_model(self, input_shape, filters, kernel_size, dropout_rate, l2_regularizer, use_attention=True):
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # First ConvLSTM2D layer
        x = layers.LayerNormalization()(inputs)
        x = layers.GaussianDropout(dropout_rate)(x)
        convlstm1 = layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation='gelu', padding='same', 
                                        return_sequences=True, kernel_regularizer=keras.regularizers.l2(l2_regularizer))(x)
        
        # LayerNormalization and Dropout after the first ConvLSTM2D
        x = layers.LayerNormalization()(convlstm1)
        x = layers.GaussianDropout(dropout_rate)(x)
        
        # Second ConvLSTM2D layer
        convlstm2 = layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation='gelu', padding='same', 
                                        return_sequences=True, kernel_regularizer=keras.regularizers.l2(l2_regularizer))(x)
        
        # Adding residual connection
        x = layers.Add()([convlstm1, convlstm2])
        
        # LayerNormalization and Dropout after combining with the residual connection
        x = layers.LayerNormalization()(x)
        x = layers.GaussianDropout(dropout_rate)(x)
        
        if use_attention:
            # Attention Mechanism
            reshaped_x = layers.Reshape((-1, input_shape[1]*input_shape[2]*filters))(x)
            attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=filters, dropout=dropout_rate)(reshaped_x, reshaped_x)
            reshaped_attention_output = layers.Reshape((-1, input_shape[1], input_shape[2], filters))(attention_output)
            
            # Adding residual connection
            x = layers.Add()([x, reshaped_attention_output])
            
            # LayerNormalization and Dropout after combining with the residual connection
            x = layers.LayerNormalization()(x)
            x = layers.GaussianDropout(dropout_rate)(x)
        
        # Reducing over the time dimension using GlobalAveragePooling
        x = layers.Reshape((-1, input_shape[1]*input_shape[2]*filters))(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Reshape((input_shape[1], input_shape[2], filters))(x)

        # Output layer
        outputs = layers.Conv2D(filters=4, kernel_size=(1, 1), activation='softmax', padding='same')(x)
        
        # Create and compile the model
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        
        optimizer = keras.optimizers.Nadam()
        
        custom_loss = self.combined_loss_function
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
        
        return model

    def compute_class_weights(self, y):
        y_indices = np.argmax(y, axis=-1) # Assuming y is one-hot encoded, convert to class indices
        class_weights = np.zeros((4,))
        unique_classes = np.unique(y_indices)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_indices.flatten())
        class_weights[unique_classes] = weights

        return class_weights

    @staticmethod
    @keras.utils.register_keras_serializable()
    def combined_loss_function(y_true, y_pred, alpha=0.25, gamma=2.0, label_smoothing=0.1, focal_loss_weight=0.25, weighted_loss_weight=0.25, crossentropy_loss_weight=0.25, count_loss_weight=0.25):
        weights = PredictionObserver.class_weights
        focal_loss_fn = keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=gamma)
        crossentropy_loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        def weighted_categorical_crossentropy(y_true, y_pred):
            y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
            y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
            loss = y_true * tf.math.log(y_pred) * weights
            loss = -tf.reduce_sum(loss, -1)
            return loss

        def count_loss(y_true, y_pred):
            true_count = tf.reduce_sum(y_true[..., 1], axis=[1, 2])
            pred_count = tf.reduce_sum(y_pred[..., 1], axis=[1, 2])
            return tf.reduce_mean(tf.square(true_count - pred_count))

        focal_loss = focal_loss_fn(y_true, y_pred)
        crossentropy_loss = crossentropy_loss_fn(y_true, y_pred)
        weighted_loss = weighted_categorical_crossentropy(y_true, y_pred)
        agent_count_loss = count_loss(y_true, y_pred)

        combined_loss = (tf.math.multiply(focal_loss_weight, focal_loss) +
                         tf.math.multiply(crossentropy_loss_weight, crossentropy_loss) +
                         tf.math.multiply(weighted_loss_weight, weighted_loss) +
                         tf.math.multiply(count_loss_weight, agent_count_loss))
        return combined_loss

    def cosine_annealing_scheduler(self, max_update=20, base_lr=0.01, final_lr=0.001, warmup_steps=5, warmup_begin_lr=0.001, cycle_length=10, exp_decay_rate=0.5):
        # Pre-compute constants for efficiency
        warmup_slope = (base_lr - warmup_begin_lr) / warmup_steps
        max_steps = max_update - warmup_steps

        def schedule(epoch):
            if epoch < warmup_steps:
                # Warmup phase with a linear increase
                return warmup_begin_lr + warmup_slope * epoch
            elif epoch < max_update:
                # Main learning phase with cosine annealing
                return final_lr + (base_lr - final_lr) * (1 + math.cos(math.pi * (epoch - warmup_steps) / max_steps)) / 2
            else:
                # Post-max_update phase with warm restarts and cosine annealing
                adjusted_epoch = epoch - max_update
                cycles = math.floor(1 + (adjusted_epoch - 1) / cycle_length)
                x = adjusted_epoch - (cycles * cycle_length)

                # Apply exponential decay to base_lr only when a new cycle begins
                decayed_lr = base_lr * (exp_decay_rate ** cycles)

                # Apply cosine annealing within the cycle
                cycle_base_lr = max(decayed_lr, final_lr)
                lr = final_lr + (cycle_base_lr - final_lr) * (1 - math.cos(math.pi * x / cycle_length)) / 2
                return max(lr, final_lr)  # Ensure lr does not go below final_lr

        return schedule

    def tune_hyperparameters(self, train_dataset, batch_size, num_folds, param_grid):
        best_loss = float('inf')
        best_params = {}
        
        dataset_size = len(list(train_dataset.as_numpy_iterator()))
        fold_size = dataset_size // num_folds

        for params in ParameterGrid(param_grid):
            fold_loss_scores = []

            for fold in range(num_folds):
                # Calculate start and end indices for the current fold
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold != num_folds - 1 else dataset_size

                # Split the dataset into training and validation sets for the current fold
                train_data_fold = train_dataset.skip(end_idx).take(dataset_size - end_idx).concatenate(train_dataset.take(start_idx))
                val_data_fold = train_dataset.skip(start_idx).take(fold_size)

                # Prepare the data folds
                train_data_fold = train_data_fold.prefetch(tf.data.experimental.AUTOTUNE)
                val_data_fold = val_data_fold.prefetch(tf.data.experimental.AUTOTUNE)

                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0)
                lr_scheduler = keras.callbacks.LearningRateScheduler(self.cosine_annealing_scheduler())
                model = self.create_model(train_data_fold.element_spec[0].shape[1:], **params)
                history = model.fit(train_data_fold, epochs=20, validation_data=val_data_fold, verbose=0, callbacks=[early_stopping, lr_scheduler])

                loss = history.history['val_loss'][-1]
                fold_loss_scores.append(loss)

            avg_loss = np.mean(fold_loss_scores).item()
            
            print(f"Params: {params}, Avg Loss: {avg_loss}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params

        return best_params, best_loss

    def train_model(self, num_steps=5, num_folds=5, batch_size=16, n_basic_samples=100, n_advanced_samples=50):
        X, y = self.prepare_data(num_steps)
        augmented_X, augmented_y, modified_X, modified_y = self.augment_data(X, y)

        # Bootstrapping the data
        X_boot, y_boot = self.bootstrap_samples(augmented_X, augmented_y, modified_X, modified_y, n_basic_samples, n_advanced_samples)

        # X should have shape (samples, timesteps, width, height, channels)
        # y should have shape (samples, width, height, channels)

        X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=0.2, random_state=42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        class_weights = self.compute_class_weights(y_train)
        PredictionObserver.class_weights = tf.Variable(class_weights, dtype=tf.float32)

        # Define hyperparameter search space
        param_grid = {
            'filters': [16],
            'kernel_size': [(3, 3)],
            'dropout_rate': [0.25, 0.5],
            'l2_regularizer': [0.001]
        }

        best_params, best_loss = self.tune_hyperparameters(train_dataset, batch_size, num_folds, param_grid)
        print(f"Best Params: {best_params}, Best Loss: {best_loss}")

        final_model = self.create_model(train_dataset.element_spec[0].shape[1:], **best_params)
        final_model.summary()
        checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0)
        lr_scheduler = keras.callbacks.LearningRateScheduler(self.cosine_annealing_scheduler())
        final_model.fit(train_dataset, epochs=20, verbose=0, callbacks=[checkpoint, early_stopping, lr_scheduler], validation_data=test_dataset)
        test_loss, test_f1_score = self.evaluate_model(final_model, X_test, y_test)
        print(f"Loss on the test set: {test_loss}")
        print(f"F1 Score on the test set: {test_f1_score}")
        return final_model

    def evaluate_model(self, model, X_test, y_test):
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        
        y_pred = model.predict(X_test, verbose=0)
        y_test_flat = y_test.argmax(axis=-1).flatten()
        y_pred_flat = y_pred.argmax(axis=-1).flatten()
        test_f1_score = f1_score(y_test_flat, y_pred_flat, average='micro')

        return test_loss, test_f1_score

    def monte_carlo_prediction(self, model, input_data, n_predictions=100):
        # List to hold predictions
        predictions = []

        # Perform n_predictions forward passes with dropout enabled
        for _ in range(n_predictions):
            # Enable training mode temporarily to activate dropout layers
            model._training = True
            prediction = model.predict(input_data, verbose=0)
            predictions.append(prediction)

            # Reset training mode
            model._training = False

        # Compute standard deviation across predictions
        prediction_std = np.std(predictions, axis=0)

        return prediction_std

    def plot_combined_heatmaps(self, past_grid_state, actual, predicted, uncertainty):
        fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)

        # Plot the input timesteps
        for i in range(5):
            sns.heatmap(past_grid_state[i], ax=axs[0, i], cmap='viridis', cbar=True, square=True, vmin=0, vmax=3)
            axs[0, i].set_title(f"Input Time {i+1}")
            axs[0, i].axis('off')

        # Plot the actual, predicted, and uncertainty heatmaps
        sns.heatmap(actual, ax=axs[1, 0], cmap='viridis', cbar=True, square=True, vmin=0, vmax=3)
        axs[1, 0].set_title("Actual State")
        axs[1, 0].axis('off')

        sns.heatmap(predicted, ax=axs[1, 1], cmap='viridis', cbar=True, square=True, vmin=0, vmax=3)
        axs[1, 1].set_title("Predicted State")
        axs[1, 1].axis('off')

        sns.heatmap(uncertainty, ax=axs[1, 2], cmap='hot', cbar=True, square=True)
        axs[1, 2].set_title("Uncertainty")
        axs[1, 2].axis('off')

        for i in range(3, 5):
            axs[1, i].axis('off')  # Hide unused subplots

        plt.show()

    def calculate_ssim(self, actual, predicted):
        # Flatten the grid states to 2D arrays for SSIM calculation
        actual_flat = actual.astype(float).reshape((self.subject.school.size, self.subject.school.size))
        predicted_flat = predicted.astype(float).reshape((self.subject.school.size, self.subject.school.size))
        return ssim(actual_flat, predicted_flat, data_range=3)

    def calculate_nrmse(self, actual, predicted):
        # Flatten the grid states to 2D arrays for NRMSE calculation
        actual_flat = actual.reshape((self.subject.school.size, self.subject.school.size))
        predicted_flat = predicted.reshape((self.subject.school.size, self.subject.school.size))

        # Calculate RMSE
        mse = np.mean((actual_flat - predicted_flat) ** 2)
        rmse = np.sqrt(mse)

        # Normalize RMSE
        range_actual = actual_flat.max() - actual_flat.min()
        nrmse = rmse / range_actual if range_actual != 0 else float('inf')
        
        return nrmse

    def display_observation(self):
        if not os.path.exists("best_model.keras"):
            self.model = self.train_model()
        elif getattr(self, "model", None) is None or self.model is None:
            self.model = keras.models.load_model("best_model.keras", custom_objects={'combined_loss_function': PredictionObserver.combined_loss_function})
        if not isinstance(self.model, keras.models.Model):
            raise ValueError("Model is not properly instantiated.")

        past_grid_state = self.grid_history[-6:-1]
        argmax_past_grid_state = keras.utils.to_categorical(past_grid_state, num_classes=4)
        input_data = np.array(argmax_past_grid_state).reshape((1, -1, self.subject.school.size, self.subject.school.size, 4))
        predicted_grid_state = self.model.predict(input_data, verbose=0)
        argmax_predicted_grid_state = np.argmax(predicted_grid_state, axis=-1)
        reshaped_grid_state = argmax_predicted_grid_state.reshape((self.subject.school.size, self.subject.school.size))
        
        prediction_std = self.monte_carlo_prediction(self.model, input_data, n_predictions=100)
        state_uncertainty = np.max(prediction_std, axis=-1).reshape((self.subject.school.size, self.subject.school.size))
        
        self.plot_combined_heatmaps(past_grid_state, self.grid_history[-1].astype(int), reshaped_grid_state, state_uncertainty)
        
        print("Actual State:")
        print(self.grid_history[-1].astype(int))
        print("Predicted State:")
        reformatted_grid_state = np.round(reshaped_grid_state, 1)
        print(reformatted_grid_state)
        print("Uncertainty (Std in Predicted Probabilities):")
        reformatted_state_uncertainty = np.array2string(state_uncertainty, formatter={'float_kind':lambda x: "{:.0e}".format(x)})
        print(reformatted_state_uncertainty)
        
        ssim_value = self.calculate_ssim(self.grid_history[-1].astype(int), reshaped_grid_state)
        print(f"SSIM: {ssim_value:.3f}")
        nrmse_value = self.calculate_nrmse(self.grid_history[-1].astype(int), reshaped_grid_state)
        print(f"NRMSE: {nrmse_value:.3f}")


class FFTAnalysisObserver(Observer):
    def __init__(self, population):
        self.subject = population
        self.subject.attach_observer(self)
        self.spatial_data = []
        self.time_series_data = []

    def update(self):
        if self.subject.timestep == 1:
            print("Initializing FFT Analysis")
            self.time_series_data.append(0)
        self.spatial_data.append(self.capture_grid_state())
        self.time_series_data.append(self.count_zombies())

    def capture_grid_state(self):
        grid_state = np.zeros((self.subject.school.size, self.subject.school.size))
        for individual in self.subject.agent_list:
            grid_state[individual.location] = individual.health_state.value if individual else 0
        return grid_state

    def count_zombies(self):
        return sum(ind.health_state == HealthState.ZOMBIE for ind in self.subject.agent_list)

    def perform_fft_analysis(self):
        self.spatial_fft = [fftshift(fft2(frame)) for frame in self.spatial_data]
        self.time_series_fft = fft(self.time_series_data)
        magnitudes = np.abs(self.time_series_fft)
        self.frequencies = np.fft.fftfreq(len(self.time_series_data), d=1)
        self.dominant_frequencies = self.frequencies[np.argsort(-magnitudes)[:5]]
        self.dominant_periods = [1 / freq if freq != 0 else float('inf') for freq in self.dominant_frequencies]

    def create_spatial_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("Spatial Data Over Time")
        im = ax.imshow(self.spatial_data[0], cmap='viridis', animated=True)
        plt.colorbar(im, ax=ax, orientation='vertical')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")

        def update(frame):
            im.set_array(self.spatial_data[frame])
            time_text.set_text(f"Time Step: {frame}")
            return [im, time_text]

        ani = animation.FuncAnimation(fig, update, frames=len(self.spatial_data), interval=50, blit=True)
        plt.show()

    def create_spatial_fft_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("FFT of Spatial Data Over Time")
        fft_data = np.log(np.abs(self.spatial_fft[0]) + 1e-10)
        im = ax.imshow(fft_data, cmap='hot', animated=True)
        plt.colorbar(im, ax=ax, orientation='vertical')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")

        def update(frame):
            fft_data = np.log(np.abs(self.spatial_fft[frame]) + 1e-10)
            im.set_array(fft_data)
            time_text.set_text(f"Time Step: {frame}")
            return [im, time_text]

        ani = animation.FuncAnimation(fig, update, frames=len(self.spatial_data), interval=50, blit=True)
        plt.show()

    def create_time_series_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("Time Series Data")
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(-1, len(self.time_series_data) + 1)
        ax.set_ylim(-1, max(self.time_series_data) + 1)

        # Mark periods and dominant frequencies on the plot
        plotted_periods = set()
        for period in self.dominant_periods:
            if period != float('inf') and period not in plotted_periods and period > 0:
                ax.axvline(x=period, color='r', linestyle='--', label=f'Period: {period:.2f} steps')
                plotted_periods.add(period)

        def update(frame):
            line.set_data(np.arange(frame), self.time_series_data[:frame])
            return line,

        ani = animation.FuncAnimation(fig, func=update, frames=len(self.time_series_data)+1, interval=50, blit=True)
        ax.legend(loc='upper right')
        plt.show()

    def create_time_series_fft_animation(self):
        fig, ax = plt.subplots()
        ax.set_title("FFT of Time Series Data")
        line, = ax.plot([], [], lw=2, label='FFT')
        ax.set_xlim(min(self.frequencies), max(self.frequencies))
        ax.set_ylim(-1, max(np.abs(self.time_series_fft)) + 1)

        # Mark dominant frequencies on the plot
        plotted_frequencies = set()
        for freq in self.dominant_frequencies:
            if freq not in plotted_frequencies:
                ax.axvline(x=freq, color='r', linestyle='--', label=f'Frequency: {freq:.2f}')
                plotted_frequencies.add(freq)

        def update(frame):
            frame_data = self.time_series_data[:frame + 1]
            fft_frame = fft(frame_data)
            freqs = np.fft.fftfreq(len(frame_data), d=1)
            line.set_data(freqs, np.abs(fft_frame))
            return line,

        ani = animation.FuncAnimation(fig, func=update, frames=len(self.time_series_data)+1, interval=50, blit=True)
        ax.legend(loc='upper right')
        plt.show()

    def plot_final_spatial_data(self, ax):
        ax.imshow(self.spatial_data[-1], cmap='viridis')
        ax.set_title("Final Spatial Data")
        ax.figure.colorbar(ax.images[0], ax=ax, orientation='vertical')

    def plot_fft_final_spatial_data(self, ax):
        ax.imshow(np.log(np.abs(self.spatial_fft[-1]) + 1e-10), cmap='hot')
        ax.set_title("FFT of Final Spatial Data")
        ax.figure.colorbar(ax.images[0], ax=ax, orientation='vertical')

    def plot_time_series_data(self, ax):
        ax.plot(self.time_series_data)
        for period in self.dominant_periods:
            if period != float('inf') and period > 0:
                ax.axvline(x=period, color='r', linestyle='--', label=f'Period: {period:.2f} steps')
        ax.set_title("Time Series Data")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Number of Zombies")
        ax.legend()

    def plot_fft_time_series_data(self, ax):
        ax.plot(self.frequencies, np.abs(self.time_series_fft))
        for freq in self.dominant_frequencies:
            ax.axvline(x=freq, color='r', linestyle='--', label=f'Frequency: {freq:.2f}')
        ax.set_title("FFT of Time Series Data")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    def display_observation(self, mode='static'):
        if not self.spatial_data or not self.time_series_data:
            print("No data available for FFT analysis.")
            return

        self.perform_fft_analysis()

        if mode == 'animation':
            self.create_spatial_animation()
            self.create_spatial_fft_animation()
            self.create_time_series_animation()
            self.create_time_series_fft_animation()
        elif mode == 'static':
            fig, axs = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)

            self.plot_final_spatial_data(axs[0, 0])
            self.plot_fft_final_spatial_data(axs[0, 1])
            self.plot_time_series_data(axs[1, 0])
            self.plot_fft_time_series_data(axs[1, 1])

            plt.show()

class PygameObserver(Observer):
    def __init__(self, population, cell_size=30, fps=10, font_size=18):
        self.subject = population
        self.subject.attach_observer(self)

        # Define the cell size, the frames per second, and font size
        self.cell_size = cell_size
        self.fps = fps
        self.font_size = font_size
        self.is_paused = False

        # Colors
        self.colors = {
            HealthState.HEALTHY: (0, 255, 0),  # Green
            HealthState.INFECTED: (255, 165, 0),  # Orange
            HealthState.ZOMBIE: (255, 0, 0),  # Red
            HealthState.DEAD: (128, 128, 128),  # Gray
            'background': (255, 255, 255),  # White
            'grid_line': (200, 200, 200),  # Light Gray
            'text': (0, 0, 0)  # Black
        }

        # Initialize Pygame and the screen
        pygame.init()
        self.screen_size = self.subject.school.size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + 50))  # Additional space for stats
        pygame.display.set_caption("Zombie Apocalypse Simulation")

        # Clock for controlling the frame rate
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", self.font_size)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.handle_quit_event()
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown_events(event)

    def handle_quit_event(self):
        pygame.quit()
        exit()

    def handle_keydown_events(self, event):
        if event.key == pygame.K_SPACE:
            self.toggle_pause()
        elif event.key == pygame.K_r:
            main()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.display_pause_message()

    def update(self):
        if self.is_paused:
            while self.is_paused:
                self.handle_events()  # Continue to handle events while paused to catch unpause event
                pygame.time.wait(10)  # Wait for a short period to reduce CPU usage while paused
        else:
            self.handle_events()  # Handle events for unpausing or quitting
            self.draw_grid()
            self.display_stats()
            pygame.display.flip()
            self.clock.tick(self.fps)
            time.sleep(0.5)

    def draw_grid(self):
        # If not paused, fill the background and draw individuals
        if not self.is_paused:
            self.screen.fill(self.colors['background'])
            self.draw_individuals()

    def draw_individuals(self):
        for individual in self.subject.agent_list:
            x, y = individual.location
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors[individual.health_state], rect)
            pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 1)  # Draw grid line

    def display_stats(self):
        stats_text = f"Healthy: {self.subject.num_healthy}, Infected: {self.subject.num_infected}, Zombie: {self.subject.num_zombie}, Dead: {self.subject.num_dead}"
        text_surface = self.font.render(stats_text, True, self.colors['text'])
        self.screen.fill(self.colors['background'], (0, self.screen_size, self.screen_size, 50))  # Clear stats area
        self.screen.blit(text_surface, (5, self.screen_size + 5))

    def display_pause_message(self):
        dark_surface = pygame.Surface((self.screen_size, self.screen_size))
        dark_surface.set_alpha(128)
        dark_surface.fill(self.colors['background'])
        self.screen.blit(dark_surface, (0, 0))
        pause_text = "Simulation Paused. Press 'Space' to resume."
        text_surface = self.font.render(pause_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.screen_size / 2 - text_surface.get_width() / 2, self.screen_size / 2 - text_surface.get_height() / 2))

    def display_observation(self):
        end_text = "Simulation Ended. Press 'R' to Restart."
        text_surface = self.font.render(end_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.screen_size / 2 - text_surface.get_width() / 2, self.screen_size / 2 - text_surface.get_height() / 2))
        pygame.display.flip()
        while True:
            self.handle_events()

class GANObserver:
    def __init__(self, population, latent_dim=100):
        self.subject = population
        self.subject.attach_observer(self)
        self.latent_dim = latent_dim
        self.data_shape = (self.subject.school.size, self.subject.school.size)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        self.real_data_samples = []

    def build_generator(self):
        model = keras.models.Sequential()
        model.add(layers.Input(shape=(self.latent_dim,)))
        model.add(layers.Dense(128, activation="elu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(int(np.prod(self.data_shape)) * 4))
        model.add(layers.Reshape((*self.data_shape, 4)))
        model.add(layers.Softmax(axis=-1))
        return model

    def build_discriminator(self):
        model = keras.models.Sequential()
        model.add(layers.Input(shape=(*self.data_shape, 4)))
        model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='valid', activation="elu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation="elu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def build_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        gan = keras.models.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
        return gan

    def capture_grid_state(self):
        grid_state = np.zeros((self.subject.school.size, self.subject.school.size))
        for i in range(self.subject.school.size):
            for j in range(self.subject.school.size):
                individual = self.subject.school.get_individual((i, j))
                grid_state[i, j] = individual.health_state.value if individual else 0
        return grid_state

    def update(self):
        real_data = self.capture_grid_state()
        self.real_data_samples.append(real_data)
    
    def train_gan(self, epochs, batch_size, discriminator_interval=5, generator_interval=1):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            d_loss_total = np.zeros(2)
            g_loss_total = 0
            
            for _ in range(discriminator_interval):
                # Select a random batch of real data
                idx = np.random.randint(0, len(self.real_data_samples), batch_size)
                real_imgs = np.array([keras.utils.to_categorical(self.real_data_samples[i], num_classes=4) for i in idx])

                # Generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise, verbose=0)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_total += np.array(d_loss)

            for _ in range(generator_interval):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                self.discriminator.trainable = False
                # Train the generator
                g_loss = self.gan.train_on_batch(noise, valid)
                g_loss_total += g_loss

            d_loss_avg = d_loss_total / discriminator_interval
            g_loss_avg = g_loss_total / generator_interval

            # Print progress
            print(f"Epoch: {epoch} [D loss: {d_loss_avg[0]}, acc.: {d_loss_avg[1]}] [G loss: {g_loss_avg}]")

    def display_observation(self):
        num_samples = 1
        self.train_gan(epochs=10, batch_size=32)
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_data = self.generator.predict(noise, verbose=0)
        generated_data = np.argmax(generated_data, axis=-1).reshape((self.subject.school.size, self.subject.school.size))
        print(generated_data)


def main():
    set_seed(0)

    # create a SchoolZombieApocalypse object
    school_sim = Population(school_size=10, population_size=10)

    # create Observer objects
    # simulation_observer = SimulationObserver(school_sim)
    # simulation_animator = SimulationAnimator(school_sim)
    # plotly_animator = PlotlyAnimator(school_sim)
    # matplotlib_animator = MatplotlibAnimator(school_sim)
    # tkinter_observer = TkinterObserver(school_sim)
    # prediction_observer = PredictionObserver(school_sim)
    # fft_observer = FFTAnalysisObserver(school_sim)
    # pygame_observer = PygameObserver(school_sim)
    gan_observer = GANObserver(school_sim)

    # run the population for a given time period
    school_sim.run_population(num_time_steps=10)
    
    print("Observers:")
    # print(simulation_observer.agent_list)
    # print(simulation_animator.agent_history[-1])

    # observe the statistics of the population
    # simulation_observer.display_observation(format="bar") # "statistics" or "grid" or "bar" or "scatter" or "table"
    # simulation_animator.display_observation(format="bar") # "bar" or "scatter" or "table"
    # plotly_animator.display_observation()
    # matplotlib_animator.display_observation()
    # tkinter_observer.display_observation()
    # prediction_observer.display_observation()
    # fft_observer.display_observation(mode='static') # "animation" or "static"
    # pygame_observer.display_observation()
    gan_observer.display_observation()


if __name__ == "__main__":
    main()


