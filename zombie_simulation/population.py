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

    def is_infected(self, context: Individual, severity: float, randomness=random.random()) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        for other in context.connections:
            if other.state == State.ZOMBIE:
                if randomness < infection_probability:
                    return True
        return False

    def is_turned(self, context: Individual, severity: float, randomness=random.random()) -> bool:
        turning_probability = context.infection_severity
        if randomness < turning_probability:
            return True
        return False

    def is_died(self, context: Individual, severity: float, randomness=random.random()) -> bool:
        death_probability = severity
        for other in context.connections:
            if other.state == State.HEALTHY or other.state == State.INFECTED:
                if randomness < death_probability:
                    return True
        return False


# can add methods that change behaviour based on the state

class HealthyMachine(StateMachine):
    # cellular automaton
    def update_state(self, severity: float) -> None:
        if self.is_infected(self.context, severity):
            self.context.state = State.INFECTED

class InfectedMachine(StateMachine):
    # cellular automaton
    def update_state(self, severity: float) -> None:
        self.context.infection_severity += 0.1
        if self.is_turned(self.context, severity):
            self.context.state = State.ZOMBIE
        elif self.is_died(self.context, severity):
            self.context.state = State.DEAD

class ZombieMachine(StateMachine):
    # cellular automaton
    def update_state(self, severity: float) -> None:
        if self.is_died(self.context, severity):
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

    __slots__ = "school_size", "grid", "strategy_factory"

    def __init__(self, school_size: int) -> None:
        self.school_size = school_size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid: np.ndarray = np.full((school_size, school_size), None, dtype=object)
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
                neighbors = self.get_neighbors(
                    (cell.location), cell.interact_range)
                for neighbor in neighbors:
                    cell.add_connection(neighbor)

    """
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                # cell = self.get_individual((i, j))
                
                neighbors = self.get_neighbors((i, j), cell.interact_range)
    """

    # update the grid in the population based on their interactions with other people
    def update_grid(self, population: list[Individual], migration_probability: float, randomness=random.random()) -> None:
        for individuals in population:
            i, j = individuals.location
            cell = self.get_individual((i, j))

            if cell is None:
                continue

            if randomness < migration_probability:
                movement_strategy = self.strategy_factory.create_strategy(cell, self)
                direction = cell.choose_direction(movement_strategy)
                self.move_individual(cell, direction)
            else:
                continue

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
        plt.ylim(0, self.subject.population_size)
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
        ax.set_ylim(0, self.subject.population_size)

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


"""
# defense that will decrease the probability of infection and death

Define the rules of the simulation
Zombie infection - if a zombie and survivor are in neighbouring cell, the survivor will become infected
Survivor attack - if a zombie and survivor are in neighbouring cell, if a zombie dies, it is removed from the simulation
Zombie movement - each zombie moves one unit towards a random direction
Survivor movement - each survivor moves one unit away from the nearest zombie

a: individual: zombie and survivor, cell: position, grid: zombie_positions and survivor_positions, simulation: update_simulation()


Q learning as one strategy
state machine should be a part of population and input individual as argument to process
game world and q-table singleton

class QLearningMovementStrategy(MovementStrategy):

    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]
    q_table: Dict[Tuple[int, int], Dict[Tuple[int, int], float]]
    learning_rate: float
    discount_factor: float
    exploration_rate: float

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_direction(self):
        state = self.get_state()
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            action = random.choice(self.legal_directions)
        else:
            # Exploitation: choose the action with highest Q-value
            q_values = self.q_table.get(state, {a: 0.0 for a in self.legal_directions})
            action = max(q_values, key=q_values.get)
        return action

    def update_q_table(self, state, action, next_state, reward):
        current_q = self.q_table.get(state, {a: 0.0 for a in self.legal_directions})[action]
        next_q = max(self.q_table.get(next_state, {a: 0.0 for a in self.legal_directions}).values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table.setdefault(state, {})[action] = new_q

    def get_state(self):
        # Define the state as the current position of the individual
        return self.individual.position

    def get_reward(self):
        # Define the reward as the number of neighbors of the individual
        return len(self.neighbors)


functional programming
save_data() save the necessary data in a save order
load_data() load the necessary data in a load order
refresh() refresh the data in the simulation according to time

birth_probability, death_probability, infection_probability, turning_probability, death_probability, connection_probability, movement_probability, attack_probability may be changed to adjust the simulation


Here are a few additional considerations that you may want to take into account when implementing the simulation:

Validation: It's important to validate the accuracy of the simulation by comparing the results to real-world data or known facts about how zombie outbreaks spread. This can help ensure that the simulation is a realistic and accurate representation of the scenario it is modeling.

Sensitivity analysis: It may be useful to perform sensitivity analysis to understand how the simulation results change as different parameters or assumptions are altered. For example, you could vary the rate of infection or the effectiveness of containment measures and see how these changes affect the outcome of the simulation.

Extension: You may want to consider extending the simulation to include additional factors or scenarios. For example, you could incorporate the behavior of external actors, such as emergency responders or military individualnel, or model the spread of the zombie virus to other locations outside the school.

Additionally, the model could be expanded to include more detailed information about the layout of the school, 
such as the locations of classrooms, doors, and other features. 
This could allow for more accurate simulations of the movement 
and interactions of students, teachers, and zombies within the school environment.
"""


"""
use random walk algorithm to simulate movement based on probability adjusted by cell's infection status and location
"""


"""
High order function
https://www.youtube.com/watch?v=4B24vYj_vaI
"""
"""
Plugin Pattern
"""

"""
# Builder Pattern (should do in agent.py)

class AgentBuilder:
    def __init__(self):
        self.agent = None

    def create_agent(self):
        self.agent = Agent()

    def set_position(self, position):
        self.agent.position = position

    def set_health(self, health):
        self.agent.health = health

    def set_speed(self, speed):
        self.agent.speed = speed

    def set_strength(self, strength):
        self.agent.strength = strength

    def get_agent(self):
        return self.agent
        
class AbstractAgentFactory(ABC):
    @abstractmethod
    def create_agent(self, builder: AgentBuilder, **kwargs) -> Agent:
        raise NotImplementedError()

class HumanFactory(AbstractAgentFactory):
    def create_agent(self, builder: AgentBuilder, **kwargs) -> Agent:
        builder.create_agent()
        builder.set_position(kwargs.get("position"))
        builder.set_health(kwargs.get("health"))
        builder.set_speed(kwargs.get("speed"))
        builder.set_strength(kwargs.get("strength"))
        return builder.get_agent()

class ZombieFactory(AbstractAgentFactory):
    def create_agent(self, builder: AgentBuilder, **kwargs) -> Agent:
        builder.create_agent()
        builder.set_position(kwargs.get("position"))
        builder.set_health(kwargs.get("health"))
        builder.set_speed(kwargs.get("speed"))
        builder.set_strength(kwargs.get("strength"))
        return builder.get_agent()

In the context of a zombie apocalypse simulation, the Builder Pattern can be used to create different types of zombie objects with various attributes and behaviors.
Here are the steps to utilize the Builder Pattern in the simulation of a zombie apocalypse:
Define a Zombie class: The Zombie class should have basic attributes like health, speed, and strength.
Create an abstract ZombieBuilder class: The ZombieBuilder class should have methods for setting the various attributes of the Zombie object. These methods can include setHealth(), setSpeed(), and setStrength().
Create concrete ZombieBuilder classes: Concrete ZombieBuilder classes should extend the ZombieBuilder class and provide implementations for the set methods. For example, a FastZombieBuilder could provide a high value for the speed attribute and a low value for the health attribute.
Create a ZombieDirector class: The ZombieDirector class should have a method that takes a ZombieBuilder object as a parameter and uses it to build a Zombie object.
Use the ZombieDirector to build different types of zombies: Using the ZombieDirector, you can create different types of zombies by using different ZombieBuilder objects. For example, you could create a SlowZombieBuilder and a StrongZombieBuilder to create different types of zombies.


"""

"""
Bridge Pattern
1. Without Bridge
class Car:
    def drive(self):
        print("Driving a car.")

class SportsCar(Car):
    def drive(self):
        print("Driving a sports car at high speed!")

car = SportsCar()
car.drive() # "Driving a sports car at high speed!"

The problem with this approach is that any changes made to the 'drive' method in the 'Car' class will also affect the 'SportsCar' class, as they share the same implementation. Additionally, if we want to add another type of car, we will have to create another subclass and repeat the same process, making the code more complex and less maintainable.

Wthout using the Bridge pattern, we would likely have a single class that contains both the interface and the implementation, making the code tightly coupled and less maintainable. This would mean that any changes made to the implementation would directly affect any references or dependencies on the code.
"""
"""
/* Implementor interface*/
interface Gear{
    void handleGear();
}

/* Concrete Implementor - 1 */
class ManualGear implements Gear{
    public void handleGear(){
        System.out.println("Manual gear");
    }
}
/* Concrete Implementor - 2 */
class AutoGear implements Gear{
    public void handleGear(){
        System.out.println("Auto gear");
    }
}
/* Abstraction (abstract class) */
abstract class Vehicle {
    Gear gear;
    public Vehicle(Gear gear){
        this.gear = gear;
    }
    abstract void addGear();
}
/* RefinedAbstraction - 1*/
class Car extends Vehicle{
    public Car(Gear gear){
        super(gear);
        // initialize various other Car components to make the car
    }
    public void addGear(){
        System.out.print("Car handles ");
        gear.handleGear();
    }
}
/* RefinedAbstraction - 2 */
class Truck extends Vehicle{
    public Truck(Gear gear){
        super(gear);
        // initialize various other Truck components to make the car
    }
    public void addGear(){
        System.out.print("Truck handles " );
        gear.handleGear();
    }
}
/* Client program */
public class BridgeDemo {    
    public static void main(String args[]){
        Gear gear = new ManualGear();
        Vehicle vehicle = new Car(gear);
        vehicle.addGear();

        gear = new AutoGear();
        vehicle = new Car(gear);
        vehicle.addGear();

        gear = new ManualGear();
        vehicle = new Truck(gear);
        vehicle.addGear();

        gear = new AutoGear();
        vehicle = new Truck(gear);
        vehicle.addGear();
    }
}
"""
"""
2. With Bridge
from abc import ABC, abstractmethod

# Implementor
class CarInterface(ABC):
    @abstractmethod
    def drive(self):
        pass

# Abstraction
class Car(CarInterface):
    def __init__(self, implementation):
        self._implementation = implementation
    
    def drive(self):
        self._implementation.drive()

# extra self-defined layer of abstraction
class SportsCarInterface:
    @abstractmethod
    def drive(self):
        pass

# Refined Abstraction
class SportsCar(Car):
    pass

# extra self-defined layer of abstraction
class TruckInterface:
    @abstractmethod
    def drive(self):
        pass

# Refined Abstraction
class Truck(Car):
    pass

# Concrete Implementor
class XSportsCarInterface(SportsCarInterface):
    def drive(self):
        print("Driving a sports car from manufacturer X at high speed!")

# Concrete Implementor
class YSportsCarInterface(SportsCarInterface):
    def drive(self):
        print("Driving a sports car from manufacturer Y at high speed!")

# Concrete Implementor
class XTruckInterface(TruckInterface):
    def drive(self):
        print("Driving a truck from manufacturer X at low speed.")

# Concrete Implementor
class YTruckInterface(TruckInterface):
    def drive(self):
        print("Driving a truck from manufacturer Y at low speed.")

car1 = SportsCar(XSportsCarInterface())
car2 = SportsCar(YSportsCarInterface())
car3 = Truck(XTruckInterface())
car4 = Truck(YTruckInterface())
car1.drive() # "Driving a sports car from manufacturer X at high speed!"
car2.drive() # "Driving a sports car from manufacturer Y at high speed!"
car3.drive() # "Driving a truck from manufacturer X at low speed."
car4.drive() # "Driving a truck from manufacturer Y at low speed."

The Bridge pattern is a way to separate an abstraction from its implementation, allowing for the two to vary independently. Using the example of a car, we can see how the Bridge pattern can be applied.
We can start by creating a Car abstract class to represent the commonality between all cars. This abstract class would have methods and properties that all types of cars should have, such as a drive method. Then, we can create various subclasses for different types of cars, such as a SportsCar and a Truck class. This is a robust design, as it allows for many more types of cars to be added in the future.
Now suppose that cars are provided by different manufacturers. We would have to create a hierarchy of car classes for manufacturer X and another for manufacturer Y. The problem now is that clients would need to know the difference between the manufacturers. And if we decide to support a third manufacturer, the codebase would become more complex.
The solution is to provide the main abstraction hierarchy, i.e. the Car abstract class and subclasses such as SportsCar and Truck, and then provide the interface (Bridge) that will exist between the abstraction and the implementation. So there will be a CarInterface, SportsCarInterface, and TruckInterface, which dictate the interface that each concrete car class must provide. The abstraction (Car class) does not know about the implementation, rather it knows about the interface. Finally, we can create a concrete implementation for each manufacturer. That is, XCar, XSportsCar, and XTruck, YCar, YSportsCar and YTruck.
Clients depend only on the abstraction but any implementation could be plugged in. So in this setup, the abstraction (Car class) could be changed without changing any of the concrete classes, and the implementation could be changed without worrying about the abstraction. This allows for a more flexible and maintainable codebase.

It uses the Bridge pattern to separate the abstraction (the interface) from the implementation (the concrete classes). The CarInterface and Car classes define the interface for the car and the SportsCar and Truck classes are the abstraction classes that inherit the Car class. The SportsCarInterface, XTruck, YTruck, XSportsCar, and YSportsCar classes are the concrete classes that implement the drive method.
By using this pattern, the implementation of the drive method is decoupled from the Car class and its subclasses. This means that the implementation can be easily swapped out without affecting any references or dependencies on the code. This makes the code more maintainable because changes to the implementation do not require changes to the abstraction or existing references to it.
Additionally, the use of different classes for different manufacturers allows for easy swapping of implementations based on the manufacturer. For example, you could swap out the XTruck class for the YTruck class and the Car class would still work correctly because it is only dependent on the TruckInterface and not the concrete class. This makes the code more flexible and allows for easier updates and changes in the future.
"""
"""
Switch as invoker, switchable object as receiver, joined by composition and the resulting object control the switch using a function and show the results using the switchable object

https://zh.m.wikipedia.org/zh-tw/%E5%91%BD%E4%BB%A4%E6%A8%A1%E5%BC%8F

class Command:
    def execute(self):
        raise NotImplementedError

class SimpleCommand(Command):
    def __init__(self, receiver, action):
        self._receiver = receiver
        self._action = action

    def execute(self):
        self._receiver.do_action(self._action)

class Receiver: # perform the actions
    def do_action(self, action):
        print(f"Performing action: {action}")

class Invoker: # invokes the action command
    def __init__(self):
        self._commands = []

    def store_command(self, command):
        self._commands.append(command)

    def execute_commands(self):
        for command in self._commands:
            command.execute()

# q: what does client do with this?
# a: Client creates commands and passes them to the invoker
receiver = Receiver()
command = SimpleCommand(receiver, "Action 1")
invoker = Invoker()
invoker.store_command(command)
invoker.execute_commands()  # Output: Performing action: Action 1

invoker does not know anything about the receiver or the command.
Receiver and command should be decoupled from each other.
This can be done by not delegating the command execution to the receiver.
Instead, the command should be responsible for executing the action.

There are two ways to undo a command:
1. Store the state of the receiver before executing the command in the command itself, combining the momento pattern
2. Call an unexecute method of the receiver
Use stack to store the commands and pop the last command to undo it, using FILO
clone the command and store it in the stack, to ensure the command won't be change or called again, using prototype pattern
use abstract class toimplement template method or storereceiver state, combine with template method pattern

The Command pattern is useful when you want to decouple the sender of a request (the client) from the object that performs the action (the receiver). It allows you to encapsulate a request as an object, which can then be parameterized with different arguments and queued or logged. You can also undo operations, support redo, and keep a history of executed commands. 
The Command pattern is often used in GUI applications, where menu items and toolbar buttons invoke commands, which are then executed by receivers such as document objects. It's also used in transactional systems, where a series of operations need to be executed as a single transaction, with the option to rollback the entire transaction if any of the operations fail. 
Overall, the Command pattern is useful in any situation where you want to decouple the sender of a request from the receiver, add new requests dynamically, or support undo/redo functionality.
"""

"""
Visitor pattern
# Define the elements that can be visited
class Element:
    def accept(self, visitor):
        visitor.visit(self)

class ConcreteElementA(Element):
    def operationA(self):
        print("Performing operation A on ConcreteElementA")

class ConcreteElementB(Element):
    def operationB(self):
        print("Performing operation B on ConcreteElementB")

# Define the visitor that will perform operations on the elements
class Visitor:
    def visit(self, element):
        element.operationA()

class ConcreteVisitor1(Visitor):
    def visit(self, element):
        if isinstance(element, ConcreteElementA):
            element.operationA()
        elif isinstance(element, ConcreteElementB):
            element.operationB()

# Use the visitor to perform operations on elements
elements = [ConcreteElementA(), ConcreteElementB()]
visitor = ConcreteVisitor1()
for element in elements:
    element.accept(visitor)

# Output:
# Performing operation A on ConcreteElementA
# Performing operation B on ConcreteElementB

# the logic of selecting the operation depending on the element is moved to the visitor
# In this example, the ConcreteElementA and ConcreteElementB classes define the objects that can be visited, and the ConcreteVisitor1 class defines the operations that can be performed on those objects. The accept method in the Element class allows the visitor to perform operations on the elements, and the visit method in the Visitor class is the entry point for the visitor to perform the operation.
# By using the visitor pattern, we can separate the operations from the elements and add new operations or change existing ones without modifying the elements themselves.

The Visitor pattern is useful when you have a complex structure of objects and you want to perform some operations on these objects without modifying their classes. It allows you to separate the algorithm or operation from the objects it operates on.
The Visitor pattern is particularly useful in the following cases:
1. When you have a complex object structure and want to perform operations on all of its elements.
2. When you have a set of related operations that you want to perform on an object structure, but don't want to modify the objects' classes to add these operations.
3. When you want to add new operations to an object structure without modifying its classes.
4. When you want to gather data from an object structure without modifying the objects' classes.
The Visitor pattern can be particularly useful when working with abstract syntax trees or other complex data structures where operations need to be performed on all elements of the structure. It allows you to keep the structure of the data separate from the operations performed on it, making it easier to maintain and extend the code.

The Visitor pattern is used when you have a set of classes that represent different types of objects and you want to perform operations on these objects without modifying their classes. The main idea behind the Visitor pattern is to separate the algorithm from the object structure. The Visitor pattern defines a new operation to be performed on each element of the object structure, and implements this operation for each class of the object structure. This allows you to add new operations to the object structure without modifying the classes of the objects themselves.
You should use the Visitor pattern when you have a complex object structure with many different types of objects, and you want to perform operations on these objects without modifying their classes. The Visitor pattern is especially useful when you have a large number of operations that need to be performed on the objects, as it allows you to encapsulate the operations in a separate class.
"""

"""
http://plague-like.blogspot.com/
https://www.earthempires.com/
https://www.pygame.org/tags/zombie
https://github.com/JarvistheJellyFish/AICivGame/blob/master/Villager.py
https://github.com/najarvis/villager-sim
civilization simulator python
GOAP
https://zhuanlan.zhihu.com/p/138003795
有限狀態機
行為樹
https://zhuanlan.zhihu.com/p/540191047
https://zhuanlan.zhihu.com/p/448895599
http://www.aisharing.com/archives/439
https://gwb.tencent.com/community/detail/126344
https://blog.51cto.com/u_4296776/5372084
https://www.jianshu.com/p/9c2200ffbb0f
https://juejin.cn/post/7162151580421062670
https://juejin.cn/post/7128710213535793182
https://juejin.cn/post/6844903425717567495
https://juejin.cn/post/6844903784489943047
http://www.aisharing.com/archives/280
https://blog.csdn.net/LIQIANGEASTSUN/article/details/118976709
Crowd Simulation Models
https://image.hanspub.org/Html/25-2570526_51496.htm
http://www.cjig.cn/html/2017/12/20171212.htm
https://zhuanlan.zhihu.com/p/35100455
Reciprocal Velocity Obstacle
Optimal Reciprocal Collision Avoidance
碰撞回避算法
路径规划算法
"""


"""

https://github.com/neo-mashiro/GameStore
https://github.com/HumanRickshaw/Python_Games
https://github.com/JrTai/Python-projects
https://github.com/CleverProgrammer/coursera
https://github.com/brunoratkaj/coursera-POO
https://github.com/xkal36/principles_of_computing
https://github.com/seschwartz8/intermediate-python-programs
https://github.com/yudong-94/Fundamentals-of-Computing-in-Python
https://github.com/Sakib37/Python_Games
https://github.com/chrisnatali/zombie
https://github.com/ITLabProject2016/internet_technology_lab_project
https://github.com/GoogleCloudPlatform/appengine-scipy-zombie-apocalypse-python
https://github.com/radical-cybertools/radical.saga

"""

"""
Population-based models (PBM) and individual-based models (IBM) are two types of models that can be used to study populations.

Population-based models (PBM) consider all individuals in a population to be interchangeable, and the main variable of interest is N, the population size. N is controlled by endogenous factors, such as density-dependence and demographic stochasticity, and exogenous factors, such as environmental stochasticity and harvest.

Individual-based models (IBM), also known as agent-based models, consider each individual explicitly. In IBM, each individual may have different survival probabilities, breeding chances, and movement propensities. Differences may be due to spatial context or genetic variation. In IBM models, N is an emergent property of individual organisms interacting with each other, with predators, competitors, and their environment.

The choice of model structure depends on the research question and understanding of the study system. If the primary data source is at the individual level, such as telemetry data, IBM is preferred. If the primary data is at the population level, such as mark-recapture analyses, PBM is preferred.

Both IBM and PBM can be used to address questions at the population or metapopulation level. The Principle of Parsimony suggests using the simplest approach when two different model structures are equally appropriate.
"""

"""
To provide analysis and prediction for the zombie apocalypse simulation, you can update the PopulationObserver class in the following ways:

Implement methods for calculating statistical measures: You can implement methods for calculating statistical measures such as mean, median, mode, and standard deviation based on the data collected by the observer. These methods can be used to provide insights into the behavior of zombies and the survival strategies that are most effective.

Use machine learning algorithms to predict zombie behavior: You can use machine learning algorithms such as decision trees, random forests, and neural networks to predict zombie behavior based on the data collected by the observer. For example, you can use these algorithms to predict the likelihood of a zombie outbreak occurring in a specific area or the rate of infection in a population.

Integrate real-world data into the simulation: You can integrate real-world data such as population demographics, climate data, and disease transmission models into the simulation to provide more accurate predictions of zombie behavior. For example, you can use population demographics to predict the rate of zombie infection in a specific area or climate data to predict the spread of the zombie virus.

Implement scenario analysis: You can implement scenario analysis to explore the impact of different variables on the outcome of the simulation. For example, you can explore the impact of different survival strategies on the rate of infection or the impact of different zombie types on the survival of the population.

Overall, updating the PopulationObserver class to provide analysis and prediction for the zombie apocalypse simulation can provide valuable insights into the behavior of zombies and the most effective survival strategies.

"""




