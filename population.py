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
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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
# if healthy or other having no alive neighbors
class FleeZombiesStrategy(MovementStrategy):
    
    individual: Any[Individual]
    legal_directions: list[tuple[int, int]]
    neighbors: list[Individual]

    def choose_direction(self):
        zombies_locations = [zombies.location for zombies in self.neighbors if zombies.state == State.ZOMBIE]
        return self.direction_against_closest(self.individual, self.legal_directions, zombies_locations)

    # find the closest zombie and move away from it
    def direction_against_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]]) -> tuple[int, int]:
        new_locations = [tuple(np.add(individual.location, direction))
                        for direction in legal_directions]
        target_distances = [float(np.linalg.norm(np.subtract(individual.location, target_location)))
                            for target_location in target_locations]
        closest_target = target_locations[np.argmin(target_distances)]
        distance_from_new_locations = [float(np.linalg.norm(np.subtract(closest_target, location)))
                                        for location in new_locations]
        max_distance = np.max(distance_from_new_locations)
        max_distance_index = np.where(
            distance_from_new_locations == max_distance)
        direction = new_locations[max_distance_index[0]]
        return direction
        
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
    def direction_towards_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]]) -> tuple[int, int]:
        new_locations = [tuple(np.add(individual.location, direction))
                        for direction in legal_directions]
        distance_matrix = np.zeros((len(new_locations), len(target_locations)))
        for i, direction in enumerate(new_locations):
            for j, target_location in enumerate(target_locations):
                distance_matrix[i, j] = float(np.linalg.norm(np.subtract(direction, target_location)))
        # consider case where all distances are 0
        min_distance = np.min(distance_matrix[distance_matrix != 0])
        min_distance_index = np.where(distance_matrix == min_distance)
        direction = new_locations[min_distance_index[0][0]]
        return direction
    
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
        # get neighbors
        neighbors = school.get_neighbors(individual.location, individual.sight_range)
        # if no neighbors, random movement
        if not neighbors:
            return RandomMovementStrategy(individual, legal_directions, neighbors)
        # get number of human and zombies neighbors
        alive_number = sum(1 for neighbor in neighbors if neighbor.state == State.HEALTHY)
        zombies_number = sum(1 for neighbor in neighbors if neighbor.state == State.ZOMBIE)
        # if no human neighbors, move away from the closest zombies
        if alive_number == 0:
            return FleeZombiesStrategy(individual, legal_directions, neighbors)
        # if no zombies neighbors, move towards the closest human
        elif zombies_number == 0:
            return ChaseHumansStrategy(individual, legal_directions, neighbors)
        # if both human and zombies neighbors, zombies move towards the closest human and human move away from the closest zombies
        else:
            if individual.state == State.ZOMBIE:
                return ChaseHumansStrategy(individual, legal_directions, neighbors)
            elif individual.state in (State.HEALTHY, State.INFECTED):
                return FleeZombiesStrategy(individual, legal_directions, neighbors)
            elif individual.state == State.DEAD:
                return NoMovementStrategy(individual, legal_directions, neighbors)
            else:
                raise Exception(
                    f"Individual {individual.id} has invalid state {individual.state.name}")

    # may consider update the grid according to the individual's location
    # after assigning all new locations to the individuals
    # may add a extra attribute to store new location
    
class Individual:

    __slots__ = "id", "state", "location", "connections", \
        "infection_severity", "interact_range", "__dict__"

    def __init__(self, id: int, state: State, location: tuple[int, int], movement_strategy: Any[MovementStrategy] = RandomMovementStrategy) -> None:
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

    def add_connection(self, other: Individual) -> None:
        self.connections.append(other)

    def move(self, direction: tuple[int, int]) -> None:
        self.location = tuple(np.add(self.location, direction))
        # self.location[0] += direction[0]
        # self.location[1] += direction[1]
        
    def choose_direction(self, movement_strategy):
        return movement_strategy.choose_direction()

    def update_state(self, severity: float) -> None:
        # Update the state of the individual based on the current state and the interactions with other people
        if self.state == State.HEALTHY:
            if self.is_infected(severity):
                self.state = State.INFECTED
        elif self.state == State.INFECTED:
            self.infection_severity += 0.1
            if self.is_turned():
                self.state = State.ZOMBIE
            elif self.is_died(severity):
                self.state = State.DEAD
        elif self.state == State.ZOMBIE:
            if self.is_died(severity):
                self.state = State.DEAD

    # cellular automaton
    def is_infected(self, severity: float) -> bool:
        infection_probability = 1 / (1 + math.exp(-severity))
        for individual in self.connections:
            if individual.state == State.ZOMBIE:
                if random.random() < infection_probability:
                    return True
        return False
    
    # probability = severity / school.max_severity
    # probability *= 1 - math.exp(-self.interaction_duration/average_interaction_duration_for_infection)
    # probability *= 1 - math.exp(-distance / average_distance_for_infection)


    # cellular automaton
    def is_turned(self) -> bool:
        turning_probability = self.infection_severity
        if random.random() < turning_probability:
            return True
        return False

    # cellular automaton
    def is_died(self, severity: float) -> bool:
        death_probability = severity
        for individual in self.connections:
            if individual.state == State.HEALTHY or individual.state == State.INFECTED:
                if random.random() < death_probability:
                    return True
        return False

    def get_info(self) -> str:
        return f"Individual {self.id} is {self.state.name} and is located at {self.location}, having connections with {self.connections}, infection severity {self.infection_severity}, interact range {self.interact_range}, and sight range {self.sight_range}."

    def __str__(self) -> str:
        return f"Individual {self.id}"

    def __repr__(self) -> str:
        return "%s(%d, %d, %s)" % (self.__class__.__name__, self.id, self.state.value, self.location)

# seperate inheritance for human and zombie class


class School:

    __slots__ = "school_size", "grid", "strategy_factory"

    def __init__(self, school_size: int) -> None:
        self.school_size = school_size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid: list[list[Optional[Individual]]] \
            = [[None for _ in range(school_size)]
                for _ in range(school_size)]
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

    # may change the add/remove function to accept individual class as well as location

    def update_connections(self) -> None:
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                cell = self.get_individual((i, j))
                if cell == None:
                    continue
                neighbors = self.get_neighbors((i, j), cell.interact_range)
                for neighbor in neighbors:
                    cell.add_connection(neighbor)

    # update the grid in the population based on their interactions with other people
    def update_grid(self, population: list[Individual], migration_probability: float) -> None:
        for individuals in population:
            i, j = individuals.location
            cell = self.get_individual((i, j))

            if cell == None:
                raise Exception(
                    f"Individual {individuals.id} is not in the grid")

            if random.random() < migration_probability:
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
        neighbors = []
        for i in range(max(0, x-interact_range), min(self.school_size, x+interact_range+1)):
            for j in range(max(0, y-interact_range), min(self.school_size, y+interact_range+1)):
                if i == x and j == y:
                    continue
                if self.within_distance(self.grid[x][y], self.grid[i][j], interact_range):
                    neighbors.append(self.grid[i][j])
        return neighbors

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
        legal_directions = [(i, j) for i in range(-1, 2) for j in range(-1, 2)
                            if (i == 0 and j == 0) or self.legal_location((individual.location[0] + i, individual.location[1] + j))]
        return legal_directions

    def legal_location(self, location: tuple[int, int]):
        return self.in_bounds(location) and self.is_occupied(location)

    def in_bounds(self, location: tuple[int, int]):
        # check if the location is in the grid
        return 0 <= location[0] < self.school_size and 0 <= location[1] < self.school_size

    def is_occupied(self, location: tuple[int, int]):
        # check if the location is empty
        return self.grid[location[0]][location[1]] == None

    """
        return any(agent.position == (x, y) for agent in self.agents)
    """

    def move_individual(self, individual: Individual, direction: tuple[int, int]):
        old_location = individual.location
        individual.move(direction)
        self.remove_individual(old_location)
        self.add_individual(individual)

    def print_info(self) -> None:
        for column in self.grid:
            for individual in column:
                if individual is not None:
                    print(individual.state.value, end=" ")
                else:
                    print(" ", end=" ")
            print()

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
        self.statistics = {}
        self.agent_list = []
        
    def update(self) -> None:
        self.statistics = self.subject.get_population_statistics()
        self.agent_list = self.subject.agent_list
        
    def display_observation(self, format='text'):
        if format == 'text':
            self.print_text_statistics()
        elif format == 'chart':
            self.print_chart_statistics()
        
        # animation, table
        
    def print_text_statistics(self):
        population_size = self.statistics['population_size']
        num_healthy = self.statistics['num_healthy']
        num_infected = self.statistics['num_infected']
        num_zombie = self.statistics['num_zombie']
        num_dead = self.statistics['num_dead']
        healthy_percentage = num_healthy / (population_size+1e-10)
        infected_percentage = num_infected / (population_size+1e-10)
        zombie_percentage = num_zombie / (population_size+1e-10)
        dead_percentage = num_dead / (population_size+1e-10)
        infected_rate = num_infected / (num_healthy+1e-10)
        turning_rate = num_zombie / (num_infected+1e-10)
        death_rate = num_dead / (num_zombie+1e-10)
        infection_probability = self.statistics['infection_probability']
        turning_probability = self.statistics['turning_probability']
        death_probability = self.statistics['death_probability']
        migration_probability = self.statistics['migration_probability']
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
    
    def print_chart_statistics(self):
        # Analyze the results by observing the changes in the population over time
        cell_states = [individual.state for individual in self.agent_list]
        counts = {state: cell_states.count(state) for state in list(State)}
        # Add a bar chart to show the counts of each state in the population
        plt.bar(np.asarray(State.value_list()), list(
                counts.values()), tick_label=State.name_list())
        # Show the plot
        plt.show()

class SchoolObserver(Observer):
    
    def __init__(self, population: Population) -> None:
        self.subject = population
        self.subject.attach_observer(self)
        self.agent_list = []
        self.grid = []
        
    def update(self) -> None:
        self.agent_list = self.subject.agent_list
        self.grid = self.subject.school.grid
        
    def display_observation(self, format='text'):
        if format == 'text':
            self.print_text_statistics()
        elif format == 'chart':
            self.print_chart_statistics()
        
        # animation, table
        
    def print_text_statistics(self):
        print("Print School:")
        for row in self.grid:
            for cell in row:
                if cell is None:
                    print(" ", end=" ")
                elif cell.state == State.HEALTHY:
                    print("H", end=" ")
                elif cell.state == State.INFECTED:
                    print("I", end=" ")
                elif cell.state == State.ZOMBIE:
                    print("Z", end=" ")
                elif cell.state == State.DEAD:
                    print("D", end=" ")
                else:
                    raise ValueError("Invalid state")
            print()
        print()
            
    def print_chart_statistics(self):
        # create a scatter plot of the population
        cell_states_value = [
            individual.state.value for individual in self.agent_list]
        x = [individual.location[0] for individual in self.agent_list]
        y = [individual.location[1] for individual in self.agent_list]
        plt.scatter(x, y, c=cell_states_value)
        plt.show()

class Population:

    def __init__(self, school_size: int, population_size: int) -> None:
        self.school: School = School(school_size)
        self.agent_list: list[Individual] = []
        self.severity: float = 0.
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
            State.value_list(), weights=[0.9, 0.05, 0.05, 0.0])
        state = State(state_index[0])
        while True:
            location = (random.randint(0, school_size-1),
                        random.randint(0, school_size-1))
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
            print("Time step: ", time+1)
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
            self.print_all_individual_info()
            print("Got Individual Info")
            self.school.print_info()
            print("Got School Info")
            self.notify_observers()
            print("Notified Observers")
            if self.num_healthy == 0:
                print("All individuals are infected")
                break
            elif self.num_infected == 0 and self.num_zombie == 0:
                print("All individuals are healthy")
                break

    def update_grid(self) -> None:
        self.school.update_grid(self.agent_list, self.migration_probability)

    def update_state(self) -> None:
        for individual in self.agent_list:
            individual.update_state(self.severity)
            if individual.state == State.DEAD:
                self.school.remove_individual(individual.location)

    def update_population_metrics(self) -> None:
        self.num_healthy = sum(
            1 for individual in self.agent_list if individual.state == State.HEALTHY)
        self.num_infected = sum(
            1 for individual in self.agent_list if individual.state == State.INFECTED)
        self.num_zombie = sum(
            1 for individual in self.agent_list if individual.state == State.ZOMBIE)
        self.num_dead = sum(
            1 for individual in self.agent_list if individual.state == State.DEAD)
        self.population_size = self.num_healthy + \
            self.num_infected + self.num_zombie + self.num_dead
        self.infection_probability = 1 - (1 / (1 + math.exp(-self.severity)))
        self.turning_probability = 1 - (1 / (1 + math.exp(-self.severity)))
        self.death_probability = self.severity
        self.migration_probability = self.population_size / \
            (self.population_size + 1)

        # may use other metrics or functions to calculate the probability of infection, turning, death, migration

    def print_all_individual_info(self) -> None:
        print(f'Population of size {self.population_size}' + '\n' +
                '\n'.join([individual.get_info() for individual in self.agent_list]))

    def attach_observer(self, observer: Observer) -> None:
        self.observers.append(observer)
        
    def detach_observer(self, observer: Observer) -> None:
        self.observers.remove(observer)
        
    def notify_observers(self) -> None:
        for observer in self.observers:
            observer.update()

    def get_population_statistics(self) -> dict[str, float]:
        # returns a dictionary of population statistics
        return {"num_healthy": self.num_healthy,
                "num_infected": self.num_infected,
                "num_zombie": self.num_zombie,
                "num_dead": self.num_dead,
                "population_size": self.population_size,
                "infection_probability": self.infection_probability,
                "turning_probability": self.turning_probability,
                "death_probability": self.death_probability,
                "migration_probability": self.migration_probability
                }

    def __str__(self) -> str:
        return f'Population with {self.num_healthy} healthy, {self.num_infected} infected, {self.num_zombie} zombie, and {self.num_dead} dead individuals'


def main():

    # create a SchoolZombieApocalypse object
    school_sim = Population(school_size=10, population_size=1)

    # create Observer objects
    population_observer = PopulationObserver(school_sim)
    school_observer = SchoolObserver(school_sim)
    
    # run the population for a given time period
    school_sim.run_population(num_time_steps=5)
    
    # observe the statistics of the population
    population_observer.display_observation(format="chart")
    school_observer.display_observation(format="chart")


if __name__ == "__main__":
    main()

"""

To use the variance of the binomial distribution to model the spread of a disease in a population, you would need to follow these steps:
Collect data on the number of individuals who have been infected with the disease and the number who have become zombies (the number of successes in each experiment). This data can be used to estimate the probability of success (i.e., the probability of an individual becoming a zombie after being infected with the disease).
Calculate the variance of the binomial distribution using the formula: variance = p * (1 - p), where p is the probability of success in each experiment.
Track the variance of the binomial distribution over time to see how it changes as the disease spreads through the population.
Use the variance of the binomial distribution to make predictions about the likely outcome of the zombie apocalypse. For example, if the variance is high, it might indicate that the disease is spreading quickly and unpredictably, which could lead to a worse outcome. On the other hand, if the variance is low, it might indicate that the disease is spreading more slowly and predictably, which could lead to a better outcome.
This formula for the variance of a binomial distribution assumes that the number of experiments is fixed. If the number of experiments is not fixed, the variance of the binomial distribution is given by the formula:
variance = n * p * (1 - p)
Where n is the number of experiments.

# defense that will decrease the probability of infection and death

Define the rules of the simulation
Zombie infection - if a zombie and survivor are in neighbouring cell, the survivor will become infected
Survivor attack - if a zombie and survivor are in neighbouring cell, if a zombie dies, it is removed from the simulation
Zombie movement - each zombie moves one unit towards a random direction
Survivor movement - each survivor moves one unit away from the nearest zombie

a: individual: zombie and survivor, cell: position, grid: zombie_positions and survivor_positions, simulation: update_simulation()



birth_probability, death_probability, infection_probability, turning_probability, death_probability, connection_probability, movement_probability, attack_probability may be changed to adjust the simulation


Here are a few additional considerations that you may want to take into account when implementing the simulation:

Data collection and storage: You may want to consider how you will store and track data about the attributes of each individual, such as their age, gender, location, and state. This could involve creating a database or data structure to store this information.

Visualization: It may be helpful to visualize the simulation in some way, such as by creating a graphical user interface or using a visualization tool like Matplotlib. This can make it easier to understand the results of the simulation and identify trends or patterns.

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
def simulate_movement(self):
    for i in range(self.width):
        for j in range(self.height):
            individual = self.grid[i][j]
            if individual is not None:
                # use the A* algorithm to find the shortest path to the nearest exit
                start = (i, j)
                # the four corners of the grid
                exits = [(0, 0), (0, self.width-1),
                        (self.height-1, 0), (self.height-1, self.width-1)]
                distances, previous = self.a_star(start, exits)
                # use the first exit as the destination
                path = self.reconstruct_path(previous, start, exits[0])

                # move to the next cell in the shortest path to the nearest exit
                if len(path) > 1:  # check if there is a valid path to the nearest exit
                    next_x, next_y = path[1]
                    # update the individual's location
                    individual.location = (next_x, next_y)
                    # remove the individual from their current location
                    self.grid[i][j] = None
                    # add the individual to their new location
                    self.grid[next_x][next_y] = individual

def a_star(self, start, goals):
    # implement the A* algorithm to find the shortest path from the start to one of the goals
    # returns the distances and previous nodes for each node in the grid
    pass

def reconstruct_path(self, previous, start, goal):
    # implement the algorithm to reconstruct the path from the previous nodes
    # returns the shortest path from the start to the goal
    pass
"""

"""
High order function
https://www.youtube.com/watch?v=4B24vYj_vaI
"""
"""
Plugin Pattern
"""
"""
Builder Pattern
# Product class that we want to build
class Pizza:
    def __init__(self):
        self.dough = ""
        self.sauce = ""
        self.topping = []

    def set_dough(self, dough):
        self.dough = dough

    def set_sauce(self, sauce):
        self.sauce = sauce

    def add_topping(self, topping):
        self.topping.append(topping)

    def __str__(self):
        return f"Dough: {self.dough}, Sauce: {self.sauce}, Topping: {self.topping}"

# Abstract Builder class
class PizzaBuilder(ABC):
    def __init__(self):
        self.pizza = Pizza()

    def create_new_pizza_product(self):
        self.pizza = Pizza()

    def get_pizza(self):
        return self.pizza

    @abstractmethod
    def build_dough(self):
        pass

    @abstractmethod
    def build_sauce(self):
        pass

    @abstractmethod
    def build_topping(self):
        pass


# Concrete Builder
class HawaiianPizzaBuilder(PizzaBuilder):
    def build_dough(self):
        self.pizza.set_dough("cross")

    def build_sauce(self):
        self.pizza.set_sauce("mild")

    def build_topping(self):
        self.pizza.add_topping("ham")
        self.pizza.add_topping("pineapple")


# Concrete Builder
class SpicyPizzaBuilder(PizzaBuilder):
    def build_dough(self):
        self.pizza.set_dough("pan baked")

    def build_sauce(self):
        self.pizza.set_sauce("hot")

    def build_topping(self):
        self.pizza.set_topping("pepperoni")
        self.pizza.set_topping("salami")

# Director
class Waiter:
    def __init__(self):
        self.pizza_builder = None

    def set_pizza_builder(self, pb):
        self.pizza_builder = pb

    def get_pizza(self):
        return self.pizza_builder.get_pizza()

    def construct_pizza(self):
        self.pizza_builder.create_new_pizza_product()
        self.pizza_builder.build_dough()
        self.pizza_builder.build_sauce()
        self.pizza_builder.build_topping()

# A customer ordering a pizza
if __name__ == "__main__":
    waiter = Waiter()
    hawaiian_pizza_builder = HawaiianPizzaBuilder()
    spicy_pizza_builder = SpicyPizzaBuilder()

    waiter.set_pizza_builder(hawaiian_pizza_builder)
    waiter.construct_pizza()

    pizza = waiter.get_pizza()
    print(pizza)

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

"""