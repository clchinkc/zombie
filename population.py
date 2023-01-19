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
from enum import Enum, auto
from functools import lru_cache, cached_property
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
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

class Individual:

    __slots__ = "id", "state", "location", "connections", \
    "infection_severity", "interact_range", "sight_range"
    
    def __init__(self, id: int, state: State, location: tuple[int, int]) -> None:
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
    
    __slots__ = "school_size", "grid", "__dict__"
    
    def __init__(self, school_size: int) -> None:
        self.school_size = school_size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid: list[list[Optional[Individual]]] \
                    = [[None for _ in range(school_size)]
                    for _ in range(school_size)]

        # may turn to width and height
        # may put Cell class in the grid where Cell class has individual attributes and rates

    def add_individual(self, individual: Individual) -> None:
        self.grid[int(individual.location[0])][int(individual.location[1])] = individual

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

    # update the gridl in the population based on their interactions with other people
    def update_grid(self, population: list[Individual], migration_probability: float) -> None:
        for individuals in population:
            i, j = individuals.location
            cell = self.get_individual((i, j))
            
            if cell == None:
                raise Exception(f"Individual {individuals.id} is not in the grid")
            
            if random.random() < migration_probability:
                direction = self.choose_direction(cell)
                self.move_individual(cell, direction)
            else:
                continue
                
                # if right next to then don't move
                # get neighbors with larger and larger range until there is a human
                # sight range is different from interact range
            
    def choose_direction(self, individual: Individual) -> tuple[int, int]:
        # get all legal direction
        legal_directions = self.get_legal_directions(individual)
        # if only one legal direction, return that direction
        if len(legal_directions) == 1:
            return legal_directions[0]
        # get all neighbors
        neighbors = self.get_neighbors(individual.location, individual.sight_range)
        # if no neighbors, move randomly
        if not neighbors:
            return random.choice(legal_directions)
        # get all human neighbors
        alive_locations = [alive.location for alive in neighbors if alive.state == State.HEALTHY]
        # get all zombie neighbors
        zombie_locations = [zombie.location for zombie in neighbors if zombie.state == State.ZOMBIE]
        # if no human neighbors, move randomly
        if not alive_locations:
            return self.move_against_closest(individual, legal_directions, zombie_locations)
        # if no zombie neighbors, move towards the closest human
        elif not zombie_locations:
            return self.move_towards_closest(individual, legal_directions, alive_locations)
        # if both human and zombie neighbors, zombie move towards the closest human and human move away from the closest zombie
        else:
            if individual.state == State.ZOMBIE:
                return self.move_towards_closest(individual, legal_directions, alive_locations)
            elif individual.state in (State.HEALTHY, State.INFECTED):
                return self.move_against_closest(individual, legal_directions, zombie_locations)
            elif individual.state == State.DEAD:
                return (0, 0)
            else:
                raise Exception(f"Individual {individual.id} has invalid state {individual.state.name}")
    # may consider update the grid according to the individual's location
    # after assigning all new locations to the individuals
    # may add a extra attribute to store new location
    
    # move_towards_closest but choose direction from legal direction with shortest distance from the closest target location
    def move_towards_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]]) -> tuple[int, int]:
        new_locations = [tuple(np.add(individual.location, direction)) for direction in legal_directions]
        distance_matrix = np.zeros((len(new_locations), len(target_locations)))
        for i, direction in enumerate(new_locations):
            for j, target_location in enumerate(target_locations):
                distance_matrix[i, j] = np.linalg.norm(np.subtract(direction, target_location))
        min_distance = np.min(distance_matrix[distance_matrix != 0]) # consider case where all distances are 0
        min_distance_index = np.where(distance_matrix == min_distance)
        direction = new_locations[min_distance_index[0][0]]
        return direction
    """    
    def move_towards_closest(self, cell: Individual, target_locations: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
        target_distances = [np.linalg.norm(np.subtract(cell.location, target_location))
                            for target_location in target_locations]
        closest_target = target_locations[np.argmin(target_distances)]
        direction = (np.sign(closest_target[0] - cell.location[0]), 
                    np.sign(closest_target[1] - cell.location[1]))
        new_location = tuple(np.add(cell.location, direction))
        return direction, new_location
    """
    
    # move_against_closest but choose direction from legal direction with longest distance from the closest target location
    def move_against_closest(self, individual: Individual, legal_directions: list[tuple[int, int]], target_locations: list[tuple[int, int]]) -> tuple[int, int]:
        new_locations = [tuple(np.add(individual.location, direction)) for direction in legal_directions]
        target_distances = [np.linalg.norm(np.subtract(individual.location, target_location)) \
                            for target_location in target_locations]
        closest_target = target_locations[np.argmin(target_distances)]
        distance_from_new_locations = [np.linalg.norm(np.subtract(closest_target, location)) \
                                                for location in new_locations]
        max_distance = np.max(distance_from_new_locations)
        max_distance_index = np.where(distance_from_new_locations == max_distance)
        direction = new_locations[max_distance_index[0]]
        return direction
    """
    def move_against_closest(self, cell, target_locations: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
        target_distances = [np.linalg.norm(np.subtract(cell.location, target_location))
                            for target_location in target_locations]
        closest_target = target_locations[np.argmin(target_distances)]
        direction = (np.sign(cell.location[0] - closest_target[0]), 
                    np.sign(cell.location[1] - closest_target[1]))
        new_location = tuple(np.add(cell.location, direction))
        return direction, new_location
    """
    
    # cases of more than one zombie and human, remove unwanted individuals
    # cases when both zombie and human are in neighbors, move towards human away from zombie
    # If the closest person is closer than the closest zombie, move towards the person, otherwise move away from the zombie
    # or move away from zombie is the priority and move towards person is the secondary priority

    def within_distance(self, individual1: Optional[Individual], individual2: Optional[Individual], interact_range: int):
        if individual1 is None or individual2 is None:
            return False
        # check if the two individuals are within a certain distance of each other
        distance = math.sqrt((individual1.location[0] - individual2.location[0])**2 + (
            individual1.location[1] - individual2.location[1])**2)
        return distance < interact_range

    def get_neighbors(self, location: tuple[int, int], interact_range: int=2):
        x, y = location
        neighbors = []
        for i in range(max(0, x-interact_range), min(self.school_size, x+interact_range+1)):
            for j in range(max(0, y-interact_range), min(self.school_size, y+interact_range+1)):
                if i == x and j == y:
                    continue
                if self.within_distance(self.grid[x][y], self.grid[i][j], interact_range):
                    neighbors.append(self.grid[i][j])
        return neighbors
    
    def get_legal_directions(self, individual: Individual) -> list[tuple[int, int]]:
        # get all possible legal moves for the individual
        legal_directions = [(i, j) for i in range(-1, 2) for j in range(-1, 2) \
                        if (i==0 and j==0) or self.legal_location((individual.location[0] + i, individual.location[1] + j))]
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

    def get_info(self) -> None:
        for column in self.grid:
            for individual in column:
                if individual is not None:
                    print(individual.id, end=" ")
                else:
                    print(" ", end=" ")
            print()
            
    # return the count inside the grid
    def __str__(self) -> str:
        return f"School({self.school_size})"
    
    def __repr__(self) -> str:
        return "%s(%d,%d)" % (self.__class__.__name__, self.school_size)

class Population:
    
    __slots__ = "school", "population", "severity", "num_healthy", "num_infected", "num_zombie", "num_dead", \
                "population_size", "infection_probability", "turning_probability", "death_probability", "migration_probability", "__dict__"
    
    def __init__(self, school_size: int, population_size: int) -> None:
        self.school: School = School(school_size)
        self.population: list[Individual] = []
        self.severity = 0.
        self.init_population(school_size, population_size)
        self.update_population_metrics()

    def add_individual(self, individual: Individual) -> None:
        self.population.append(individual)
        self.school.add_individual(individual)

    def remove_individual(self, individual: Individual) -> None:
        self.population.remove(individual)
        self.school.remove_individual(individual.location)
        
    def create_individual(self, id: int, school_size: int) -> Individual:
        state_index = random.choices(State.value_list(), weights=[0.9, 0.05, 0.05, 0.0])
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
            self.get_all_individual_info()
            print("Got Individual Info")
            self.school.get_info()
            print("Got School Info")
            if self.num_healthy == 0:
                print("All individuals are infected")
                break
            elif self.num_infected == 0 and self.num_zombie == 0:
                print("All individuals are healthy")
                break
            
    """
    def update(self):
        for agent in self.population:    
            action = self.choose_action(agent)
            if action == "move":
                direction = self.choose_direction(agent)
                self.move_agent(agent, direction)
            elif action == "attack":
                self.attack_neighbors(agent)
            else:
                continue
    """

    def update_grid(self) -> None:
        self.school.update_grid(self.population, self.migration_probability)

    def update_state(self) -> None:
        for individual in self.population:
            individual.update_state(self.severity)
            if individual.state == State.DEAD:
                self.school.remove_individual(individual.location)

    def update_population_metrics(self) -> None:
        self.num_healthy = sum(1 for individual in self.population if individual.state == State.HEALTHY)
        self.num_infected = sum(1 for individual in self.population if individual.state == State.INFECTED)
        self.num_zombie = sum(1 for individual in self.population if individual.state == State.ZOMBIE)
        self.num_dead = sum(1 for individual in self.population if individual.state == State.DEAD)
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie + self.num_dead
        self.infection_probability = 1 - (1 / (1 + math.exp(-self.severity)))
        self.turning_probability = 1 - (1 / (1 + math.exp(-self.severity)))
        self.death_probability = self.severity
        self.migration_probability = self.population_size / (self.population_size + 1)

        # may use other metrics or functions to calculate the probability of infection, turning, death, migration

    def get_all_individual_info(self) -> None:
        print(f'Population of size {self.population_size}' + '\n' + \
            '\n'.join([individual.get_info() for individual in self.population]))

    def observe_population(self) -> None:
        # Count the number of individuals in each state
        num_healthy = self.num_healthy
        num_infected = self.num_infected
        num_zombie = self.num_zombie
        num_dead = self.num_dead

        # Calculate the percentage of cells in each state
        healthy_percent = num_healthy / (num_healthy + num_infected + num_zombie + 1e-10)
        infected_percent = num_infected / (num_healthy + num_infected + num_zombie + 1e-10)
        zombie_percent = num_zombie / (num_healthy + num_infected + num_zombie + 1e-10)

        # Calculate the rate of infection, turning, and death
        infection_rate = num_infected / (num_healthy + num_infected + 1e-10)
        turning_rate = num_zombie / (num_infected + 1e-10)
        death_rate = num_dead / (num_infected + num_zombie + 1e-10)

        # Print the results
        print("Number of healthy individuals:", num_healthy)
        print("Number of infected individuals:", num_infected)
        print("Number of zombie individuals:", num_zombie)
        print("Number of dead individuals:", num_dead)
        print("Percentage of healthy cells:", healthy_percent)
        print("Percentage of infected cells:", infected_percent)
        print("Percentage of zombie cells:", zombie_percent)
        print("Infection rate:", infection_rate)
        print("Turning rate:", turning_rate)
        print("Death rate:", death_rate)

    def observe_school(self) -> None:
        # Print a visual representation of the school, with each cell represented by a character
        for row in self.school.grid:
            for cell in row:
                if cell == State.HEALTHY:
                    print("H", end="")
                elif cell == State.INFECTED:
                    print("I", end="")
                elif cell == State.ZOMBIE:
                    print("Z", end="")
                elif cell == State.DEAD:
                    print("D", end="")
            print()
            
    def plot_school(self) -> None:
        # create a scatter plot of the population
        cell_states_value = [individual.state.value for individual in self.population]
        x = [individual.location[0] for individual in self.population]
        y = [individual.location[1] for individual in self.population]
        plt.scatter(x, y, c=cell_states_value)
        plt.show()

    def plot_population(self) -> None:
        # Analyze the results by observing the changes in the population over time
        cell_states = [individual.state for individual in self.population]
        counts = {state: cell_states.count(state) for state in list(State)}

        # Add a bar chart to show the counts of each state in the population
        plt.bar(np.asarray(State.value_list()), list(counts.values()), tick_label=State.name_list())

        # Show the plot
        plt.show()
        
    def animation(self) -> None:
        # create an animation of the grid throughout the simulation
        # create a figure and axis
        fig, ax = plt.subplots()

        # create a scatter plot of the population
        cell_states_value = [individual.state.value for individual in self.population]
        x = [individual.location[0] for individual in self.population]
        y = [individual.location[1] for individual in self.population]
        sc = ax.scatter(x, y, c=cell_states_value)

        # create a function to update the scatter plot
        def update(i):
            # update the scatter plot
            cell_states_value = [individual.state.value for individual in self.population]
            x = [individual.location[0] for individual in self.population]
            y = [individual.location[1] for individual in self.population]
            sc.set_offsets(np.c_[x, y])
            sc.set_array(np.array(cell_states_value))
            return sc

        # create an animation
        anim = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)

        # show the animation
        plt.show()
        
    # animation not working
        
    def __str__(self) -> str:
        return f'Population with {self.num_healthy} healthy, {self.num_infected} infected, {self.num_zombie} zombie, and {self.num_dead} dead individuals'


if __name__ == '__main__':

    # create a SchoolZombieApocalypse object
    school_sim = Population(school_size=10, population_size=1)

    # run the population for a given time period
    school_sim.run_population(5)

    # observe the changes in the population and school over time
    # school_sim.observe_population()
    # school_sim.observe_school()
    # school_sim.plot_school()
    # school_sim.plot_population()
    #school_sim.animation()


"""
# Define the rules or events that trigger transitions between states
# Determine whether the individual has been infected based on their interactions with zombies or infected individuals
def is_infected(self, school, severity, tau, sigma, effectiveness):
    # Check if the individual has come into contact with a zombie or infected individual
    for individual in self.interactions:
        if individual.state == State.ZOMBIE or individual.state == State.INFECTED:
            # Calculate the probability of infection based on the duration of the interaction
            
            # The probability of infection is being calculated based on the duration of the interaction between the individual and another individual. The longer the interaction, the higher the probability of infection. The probability is being calculated using the formula 1 - e^(-duration/tau), where tau is a parameter representing the average time it takes for the infection to be transmitted. The exponent in the formula is negative because a longer duration means a higher probability of infection, and the negative exponent means that the probability decreases as the duration increases. The final probability is calculated by subtracting this value from 1, meaning that the probability increases as the duration increases.
            
            probability = 1 - math.exp(-self.interaction_duration / tau)

            # Calculate the probability of infection based on the distance between the two individuals
            row1, col1 = self.location
            row2, col2 = individual.location
            distance = math.sqrt((row1 - row2)**2 + (col1 - col2)**2)
            
            # This line of code is updating the probability of infection based on the distance between the two individuals. The probability is being calculated using the formula 1 - e^(-distance/sigma), where sigma is a parameter representing the average distance at which the infection can be transmitted. The exponent in the formula is negative because a shorter distance means a higher probability of infection, and the negative exponent means that the probability decreases as the distance increases. The probability is then being updated by multiplying it by this value, meaning that the overall probability will decrease as the distance increases.
            
            probability *= 1 - math.exp(-distance / sigma)

            # Multiply the probability by the effectiveness of any protective measures
            
            # This line of code is updating the probability of infection based on the effectiveness of any protective measures that the individual may be using. For example, if the individual is wearing a mask or gloves, the probability of infection may be lower. The probability is being updated by multiplying it by the effectiveness value, which represents the degree to which the protective measures are effective at preventing infection. If the effectiveness value is 1, it means that the measures are completely effective and the probability will not change. If the effectiveness value is less than 1, it means that the measures are less effective and the probability will increase.
            
            probability *= effectiveness

            # Multiply the probability by the overall severity of the outbreak
            
            # This line of code is updating the probability of infection based on the overall severity of the zombie outbreak. The probability is being updated by multiplying it by the severity value, which represents the overall severity of the outbreak on a scale from 0 to max_severity. If the severity value is 0, it means that the outbreak is not severe and the probability will not change. If the severity value is greater than 0, it means that the outbreak is more severe and the probability will increase. The probability is also being divided by the max_severity value, which represents the maximum possible severity of the outbreak. This is being done to normalize the probability so that it is always between 0 and 1.
            
            probability *= severity / school.max_severity

            # Return True if the probability is greater than a random number, False otherwise
            return random.random() < probability
    return False
    
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



# use multiple numpy matrix to represent each state
self.grid = np.zeros((self.grid_size,self.grid_size), dtype=int)



One potential improvement to the cellular automaton model could be to add more complex rules for transitioning between states. 
For example, the model could incorporate additional factors 
such as the age or health of the students and zombies, 
as well as the availability of weapons or other resources. 
This could allow for more realistic and nuanced simulations of the zombie apocalypse at school.

Additionally, the model could be expanded to include more detailed information about the layout of the school, 
such as the locations of classrooms, doors, and other features. 
This could allow for more accurate simulations of the movement 
and interactions of students, teachers, and zombies within the school environment.
"""
"""
def choose_action(self, agent):
    neighbors = self.get_neighbors(agent)
    if isinstance(agent, Human):
        for neighbor in neighbors:
            if isinstance(neighbor, Zombie):
                return "attack"
        return "move"
    elif isinstance(agent, Zombie):
        for neighbor in neighbors:
            if isinstance(neighbor, Human):
                return "attack"
        return "move"
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
Closure
https://github.com/ArjanCodes/2022-functions/blob/main/strategy_fn_closure.py

1. No closure
def track_score(scores: dict, player: str, score: int):
    if player in scores:
        scores[player] += score
    else:
        scores[player] = score
        
def get_scores(scores: dict):
    return scores
    
scores = {}
track_score(scores, "player1", 10)
track_score(scores, "player2", 5)
print(get_scores(scores)) # {'player1': 10, 'player2': 5}
track_score(scores, "player1", 3)
print(get_scores(scores)) # {'player1': 13, 'player2': 5}

This version of the function is passing the scores dictionary as an argument to each function that needs to update or access it. This can make the code more verbose and harder to understand in the case that the data needs to be passed around multiple functions and methods.

2. With closure
def create_player_score_tracker():
    scores = {}

    def track_score(player: str, score: int):
        if player in scores:
            scores[player] += score
        else:
            scores[player] = score

    def get_scores():
        return scores

    return track_score, get_scores

track_score, get_scores = create_player_score_tracker()
track_score("player1", 10)
track_score("player2", 5)
print(get_scores()) # {'player1': 10, 'player2': 5}
track_score("player1", 3)
print(get_scores()) # {'player1': 13, 'player2': 5}

In this version, the scores dictionary is defined within the scope of the create_player_score_tracker function, but it can still be accessed and modified by the track_score and get_scores functions because they are closures. This allows you to keep track of the scores without having to pass the scores dictionary around as an argument to each function that needs to update or access it.
"""
"""
Plugin Pattern & Factory Pattern
https://github.com/ArjanCodes/2021-plugin-architecture
"""
"""
Factory Pattern
https://github.com/ArjanCodes/2021-factory-pattern
https://www.youtube.com/watch?v=zGbPd4ZP39Y
"""