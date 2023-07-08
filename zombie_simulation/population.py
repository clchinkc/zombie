"""
To implement a simulation of a person's activity during a zombie apocalypse at school, we would need to define several classes and functions to represent the different elements of the simulation.

First, we would need a Person class to represent each person in the simulation. This class would have attributes to track the person's location, state (alive, undead, or escaped), health, and any weapons or supplies they may have. It would also have methods to move the person on the grid and interact with other people and zombies.

Next, we would need a Zombie class to represent each zombie in the simulation. This class would have similar attributes and methods as the Person class, but would also include additional attributes and methods to simulate the behavior of a zombie (such as attacking living people and spreading the infection).

We would also need a School class to represent the layout of the school and track the locations of people and zombies on the grid. This class would have a two-dimensional array to represent the grid, with each cell containing a Person or Zombie object, or None if the cell is empty. The School class would also have methods to move people and zombies on the grid and update their states based on the rules of the simulation.

Finally, we would need a main simulate function that would set up the initial conditions of the simulation (such as the layout of the school, the number and distribution of people and zombies, and any weapons or supplies), and then run the simulation for a specified number of steps. This function would use the School class to move people and zombies on the grid and update their states, and could also include additional code to track and display the progress of the simulation.
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import threading
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from itertools import product
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation, colors, patches
from scipy import stats


# Define the states and transitions for the state machine model
class State(Enum):

    HEALTHY = auto()
    INFECTED = auto()
    ZOMBIE = auto()
    DEAD = auto()

    @classmethod
    def name_list(cls) -> list[str]:
        return [enm.name for enm in State]

    @classmethod
    def value_list(cls) -> list[int]:
        return [enm.value for enm in State]


# state pattern


class StateMachine(ABC):
    def __init__(self, context: Individual) -> None:
        self.context = context

    @abstractmethod
    def update_state(self, severity: float) -> None:
        pass

    def is_infected(self, severity: float, randomness=random.random()) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        if any(other.state == State.ZOMBIE for other in self.context.connections):
            if randomness < infection_probability:
                return True
        return False

    def is_turned(self, severity: float, randomness=random.random()) -> bool:
        turning_probability = self.context.infection_severity
        if randomness < turning_probability:
            return True
        return False

    def is_died(self, severity: float, randomness=random.random()) -> bool:
        death_probability = severity
        if any(other.state == State.HEALTHY or other.state == State.INFECTED for other in self.context.connections):
            if randomness < death_probability:
                return True
        return False


# can add methods that change behaviour based on the state

class HealthyMachine(StateMachine):
    # cellular automaton
    def update_state(self, severity: float) -> None:
        if self.is_infected(severity):
            self.context.state = State.INFECTED

class InfectedMachine(StateMachine):
    # cellular automaton
    def update_state(self, severity: float) -> None:
        self.context.infection_severity += 0.1
        if self.is_turned(severity):
            self.context.state = State.ZOMBIE
        elif self.is_died(severity):
            self.context.state = State.DEAD

class ZombieMachine(StateMachine):
    # cellular automaton
    def update_state(self, severity: float) -> None:
        if self.is_died(severity):
            self.context.state = State.DEAD


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
        zombies_locations = [zombies.location for zombies in self.neighbors if zombies.state == State.ZOMBIE]
        return self.direction_against_closest(self.individual, self.legal_directions, zombies_locations)

    # find the closest zombie and move away from it
    def direction_against_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]],) -> tuple[int, int]:
        distances = [np.linalg.norm(np.subtract(individual.location, target)) for target in target_locations]
        closest_index = np.argmin(distances)
        closest_target = target_locations[closest_index]
        direction_distances = [np.linalg.norm(np.add(d, individual.location) - closest_target) for d in legal_directions]
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
        alive_locations = [alive.location for alive in self.neighbors if alive.state == State.HEALTHY]
        return self.direction_towards_closest(self.individual, self.legal_directions, alive_locations)

    # find the closest human and move towards it
    def direction_towards_closest(self, individual: Individual,legal_directions: list[tuple[int, int]],target_locations: list[tuple[int, int]],) -> tuple[int, int]:
        new_locations = [tuple(np.add(individual.location, direction)) for direction in legal_directions]
        distance_matrix = np.linalg.norm(np.subtract(np.array(new_locations)[:, np.newaxis, :], np.array(target_locations)[np.newaxis, :, :]), axis=2)
        min_distance = np.min(distance_matrix[distance_matrix != 0])
        min_directions = np.where(distance_matrix == min_distance)[0]
        return legal_directions[random.choice(min_directions)]

    # may use np.sign

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
        alive_number = len([alive for alive in neighbors if alive.state == State.HEALTHY])
        zombies_number = len([zombies for zombies in neighbors if zombies.state == State.ZOMBIE])
        # if no human neighbors, move away from the closest zombies
        if alive_number == 0 and zombies_number > 0:
            return FleeZombiesStrategy(individual, legal_directions, neighbors)
        # if no zombies neighbors, move towards the closest human
        elif zombies_number == 0 and alive_number > 0:
            return ChaseHumansStrategy(individual, legal_directions, neighbors)
        # if both human and zombies neighbors, zombies move towards the closest human and human move away from the closest zombies
        else:
            if individual.state == State.ZOMBIE and alive_number > 0:
                return ChaseHumansStrategy(individual, legal_directions, neighbors)
            elif (individual.state == State.HEALTHY or individual.state == State.INFECTED) and zombies_number > 0:
                return FleeZombiesStrategy(individual, legal_directions, neighbors)
            elif individual.state == State.DEAD:
                return NoMovementStrategy(individual, legal_directions, neighbors)
            else:
                return BrownianMovementStrategy(individual, legal_directions, neighbors)

    # may consider update the grid according to the individual's location
    # after assigning all new locations to the individuals
    # may add a extra attribute to store new location


class Individual:

    __slots__ = ("id","state","location","connections","infection_severity","interact_range","__dict__",)

    def __init__(self,id: int,state: State,location: tuple[int, int],movement_strategy: Any[MovementStrategy] = RandomMovementStrategy,) -> None:
        self.id: int = id
        self.state: State = state
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
    def add_connection(self, other: Individual) -> Individual:
        self.connections.append(other)
        return self

    def move(self, direction: tuple[int, int]) -> Individual:
        self.location = tuple(np.add(self.location, direction))
        # self.location[0] += direction[0]
        # self.location[1] += direction[1]
        return self

    def choose_direction(self, movement_strategy) -> tuple[int, int]:
        return movement_strategy.choose_direction()

    def update_state(self, severity: float) -> Individual:
        if self.state == State.HEALTHY:
            self.state_machine = HealthyMachine(self)
        elif self.state == State.INFECTED:
            self.state_machine = InfectedMachine(self)
        elif self.state == State.ZOMBIE:
            self.state_machine = ZombieMachine(self)
        elif self.state == State.DEAD:
            pass
        # Update the state of the individual based on the current state and the interactions with other people
        self.state_machine.update_state(severity)
        return self

    def get_info(self) -> str:
        return f"Individual {self.id} is {self.state.name} and is located at {self.location}, having connections with {self.connections}, infection severity {self.infection_severity}, interact range {self.interact_range}, and sight range {self.sight_range}."

    def __str__(self) -> str:
        return f"Individual {self.id}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id}, {self.state.value}, {self.location})"


# separate inheritance for human and zombie class


class School:

    __slots__ = ("school_size", "grid", "strategy_factory", "grid_lock", "__dict__",)

    def __init__(self, school_size: int) -> None:
        self.school_size = school_size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid: np.ndarray = np.full((school_size, school_size), None, dtype=object)
        self.strategy_factory = MovementStrategyFactory()
        self.grid_lock = threading.Lock()

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

    def update_cell_connections(self, cell) -> None:
        if cell is None:
            return
        neighbors = self.get_neighbors(cell.location, cell.interact_range)
        for neighbor in neighbors:
            cell.add_connection(neighbor)

    def update_connections(self, max_workers=4) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in self.grid:
                for cell in row:
                    futures.append(executor.submit(self.update_cell_connections, cell))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An exception occurred: {exc}")

    """
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                # cell = self.get_individual((i, j))
                
                neighbors = self.get_neighbors((i, j), cell.interact_range)
    """

    # update the grid in the population based on their interactions with other people
    def update_individual(self, individual, migration_probability, randomness):
        i, j = individual.location
        cell = self.get_individual((i, j))

        if cell is None:
            return

        if randomness < migration_probability:
            movement_strategy = self.strategy_factory.create_strategy(cell, self)
            direction = cell.choose_direction(movement_strategy)
            self.move_individual(cell, direction)
        else:
            return

    def update_grid(self, population: list[Individual], migration_probability: float, randomness=random.random(), max_workers=4) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.update_individual, individual, migration_probability, randomness) for individual in population]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An exception occurred: {exc}")

                # if right next to then don't move
                # get neighbors with larger and larger range until there is a human
                # sight range is different from interact range

    def get_neighbors(self, location: tuple[int, int], interact_range: int = 2):
        x, y = location
        x_range = range(max(0, x - interact_range), min(self.school_size, x + interact_range + 1))
        y_range = range(max(0, y - interact_range), min(self.school_size, y + interact_range + 1))
        
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
        return self.in_bounds(location) and self.is_occupied(location)

    def in_bounds(self, location: tuple[int, int]) -> bool:
        # check if the location is in the grid
        return (0 <= location[0] < self.school_size and 0 <= location[1] < self.school_size)

    def is_occupied(self, location: tuple[int, int]) -> bool:
        # check if the location is empty
        return self.grid[location[0]][location[1]] == None

    """
        return any(agent.position == (x, y) for agent in self.agents)
    """

    def move_individual(self, individual: Individual, direction: tuple[int, int]) -> None:
        new_location = tuple(np.add(individual.location, direction))
        
        with self.grid_lock:
            if self.is_occupied(new_location):
                return
        old_location = individual.location
        individual.move(direction)
        self.remove_individual(old_location)
        self.add_individual(individual)

    def get_info(self) -> str:
        return "\n".join(
            " ".join(str(individual.state.value) if individual else " " for individual in column)
            for column in self.grid
            )

    # return the count inside the grid
    
    def __str__(self) -> str:
        return f"School({self.school_size})"

    def __repr__(self) -> str:
        return "%s(%d,%d)" % (self.__class__.__name__, self.school_size)


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


class PopulationObserver(Observer):
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
        

    def display_observation(self, format="text"):
        if format == "statistics":
            self.print_statistics_text()
        elif format == "grid":
            self.print_grid_text()
        elif format == "chart":
            self.print_chart_graph()
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
        print()
        
        # The mean can be used to calculate the average number of zombies that appear in a specific area over time. This can be useful for predicting the rate of zombie infection and determining the necessary resources needed to survive.
        mean = np.mean([d["num_zombie"] for d in self.statistics])
        # The median can be used to determine the middle value in a set of data. In a zombie apocalypse simulation, the median can be used to determine the number of days it takes for a specific area to become overrun with zombies.
        median = np.median([d["num_zombie"] for d in self.statistics])
        # The mode can be used to determine the most common value in a set of data. In a zombie apocalypse simulation, the mode can be used to determine the most common type of zombie encountered or the most effective weapon to use against them.
        mode = stats.mode([d["num_zombie"] for d in self.statistics])[0][0]
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
            State.HEALTHY: "H",
            State.INFECTED: "I",
            State.ZOMBIE: "Z",
            State.DEAD: "D",
        }

        for row in self.grid:
            for cell in row:
                try:
                    print(state_symbols[cell.state], end=" ")
                except AttributeError:
                    print(state_symbols[cell], end=" ")
            print()
        print()

    def print_chart_graph(self):
        # Analyze the results by observing the changes in the population over time
        cell_states = [individual.state for individual in self.agent_list]
        counts = {state: cell_states.count(state) for state in list(State)}
        # Add a bar chart to show the counts of each state in the population
        plt.bar(
            np.asarray(State.value_list()),
            list(counts.values()),
            tick_label=State.name_list(),
            label=State.name_list(),
            color=sns.color_palette("deep")
        )
        # Set axis range as maximum count states
        plt.ylim(0, self.statistics[0]["population_size"])
        # Put a legend to the right of the current axis
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # Show the plot
        plt.tight_layout()
        plt.show()

    def print_scatter_graph(self):
        # Create a figure
        fig, ax = plt.subplots(1, 1)
        
        # create a scatter plot of the population
        cell_states_value = [individual.state.value for individual in self.agent_list]
        x = [individual.location[0] for individual in self.agent_list]
        y = [individual.location[1] for individual in self.agent_list]
        
        # create a colormap from the seaborn palette and the number of colors equal to the number of members in the State enum
        cmap = colors.ListedColormap(sns.color_palette("deep", n_colors=len(State)))
        
        # create a list of legend labels and colors for each state in the State enum
        handles = [patches.Patch(color=cmap(i), label=state.name) for i, state in enumerate(State)]
        
        ax.scatter(x, y, c=cell_states_value, cmap=cmap)

        # Set axis range
        ax.set_xlim(-1, self.subject.school.school_size)
        ax.set_ylim(-1, self.subject.school.school_size)

        # Put a legend to the right of the current axis
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), labels=State.name_list())
        plt.tight_layout()
        plt.show()

class PopulationAnimator(Observer):
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.agent_history = []

    def update(self) -> None:
        self.agent_history.append(deepcopy(self.subject.agent_list))

    def display_observation(self, format="chart"):
        if format == "chart":
            self.print_chart_animation()
        elif format == "scatter":
            self.print_scatter_animation()

        # table

    def print_chart_animation(self):
        counts = []
        for i in range(len(self.agent_history)):
            cell_states = [individual.state for individual in self.agent_history[i]]
            counts.append([cell_states.count(state) for state in list(State)])
        self.bar_chart_animation(np.array(State.value_list()), counts, State.name_list())

    def bar_chart_animation(self, x, y, ticks):
        # create a figure and axis
        fig, ax = plt.subplots()

        # set the title and labels
        ax.set_title("Bar Chart Animation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # Set axis range
        ax.set_ylim(0, len(self.agent_history[0]))

        # create the bar chart
        bars = ax.bar(x, y[0], tick_label=ticks, label=State.name_list(), color=sns.color_palette("deep"))

        # create timestep labels
        text_box = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        # function to update the chart
        def update(i):
            for j in range(len(bars)):
                bars[j].set_height(y[i][j])
            text_box.set_text(f"t = {i}")

        # create the animation
        anim = animation.FuncAnimation(fig, update, frames=len(y), interval=1000, repeat=False)
        
        # Put a legend to the right of the current axis
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # save the animation
        #anim.save("bar_chart_animation.gif", writer="pillow", fps=3, dpi=10)

        # show the animation
        plt.tight_layout()
        plt.show()

    def print_scatter_animation(self):
        cell_states_value = [[individual.state.value for individual in agent_list] for agent_list in self.agent_history]
        x = [[individual.location[0] for individual in agent_list] for agent_list in self.agent_history]
        y = [[individual.location[1] for individual in agent_list] for agent_list in self.agent_history]

        # tick label suitable for maps
        self.scatter_chart_animation(x, y, cell_states_value)

    def scatter_chart_animation(self, x, y, cell_states_value):
        # Create a figure
        fig, ax = plt.subplots(1, 1)
        
        # Create an animation function
        def animate(i, sc, label):
            # Update the scatter plot
            sc.set_offsets(np.c_[x[i], y[i]])
            sc.set_array(cell_states_value[i])
            # Set the label
            label.set_text("t = {}".format(i))
            # Return the artists set
            return sc, label

        # create a colormap from the seaborn palette and the number of colors equal to the number of members in the State enum
        cmap = colors.ListedColormap(sns.color_palette("deep", n_colors=len(State)))
        
        # create a list of legend labels and colors for each state in the State enum
        handles = [patches.Patch(color=cmap(i), label=state.name) for i, state in enumerate(State)]

        # Create a scatter plot
        sc = ax.scatter(x, y, c=cell_states_value, cmap=cmap)
        # Create a label
        label = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        # Set axis range
        ax.set_xlim(0, self.subject.school.school_size)
        ax.set_ylim(0, self.subject.school.school_size)
        # Create the animation object
        anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=1000, repeat=False, fargs=(sc, label))
        # Put a legend to the right of the current axis
        plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), labels=State.name_list())
        # Save the animation
        #anim.save("scatter_chart_animation.gif", writer="pillow", fps=3, dpi=10)
        # Show the plot
        plt.tight_layout()
        plt.show()


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
        state_index = random.choices(
            State.value_list(), weights=[0.8, 0.1, 0.1, 0.0])
        state = State(state_index[0])
        while True:
            location = (
                random.randint(0, school_size - 1),
                random.randint(0, school_size - 1),
            )
            if self.school.legal_location(location):
                break
        return Individual(id, state, location)

    def init_population(self, school_size: int, population_size: int) -> None:
        for i in range(population_size):
            individual = self.create_individual(i, school_size)
            self.add_individual(individual)

    # a method to init using a grid of "A", "I", "Z", "D"

    def run_population(self, num_time_steps: int) -> None:
        for time in range(num_time_steps):
            print("Time step: ", time + 1)
            self.severity = time / num_time_steps
            print("Severity: ", self.severity)
            self.update_grid()
            print("Updated Grid")
            self.school.update_connections()
            print("Updated Connections")
            self.update_state()
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

    def update_grid(self) -> None:
        self.school.update_grid(self.agent_list, self.migration_probability)

    def update_state(self) -> None:
        for individual in self.agent_list:
            individual.update_state(self.severity)
            if individual.state == State.DEAD:
                self.school.remove_individual(individual.location)

    def update_population_metrics(self) -> None:
        state_counts = Counter([individual.state for individual in self.agent_list])
        self.num_healthy = state_counts[State.HEALTHY]
        self.num_infected = state_counts[State.INFECTED]
        self.num_zombie = state_counts[State.ZOMBIE]
        self.num_dead = state_counts[State.DEAD]
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie
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
        return f"Population({self.school.school_size}, {self.population_size})"


def main():

    # create a SchoolZombieApocalypse object
    school_sim = Population(school_size=10, population_size=10)

    # create Observer objects
    population_observer = PopulationObserver(school_sim)
    population_animator = PopulationAnimator(school_sim)

    # run the population for a given time period
    school_sim.run_population(num_time_steps=10)
    
    print("Observers:")
    print(population_observer.agent_list)
    print(population_animator.agent_history[-1])

    # observe the statistics of the population
    # population_observer.display_observation(format="statistics")
    # population_observer.display_observation(format="grid")
    population_observer.display_observation(format="chart")
    # population_observer.display_observation(format="scatter")
    population_animator.display_observation(format="chart")
    # population_animator.display_observation(format="scatter")


if __name__ == "__main__":
    main()



