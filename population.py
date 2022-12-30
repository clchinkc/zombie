"""
To implement a simulation of a person's activity during a zombie apocalypse at school, we would need to define several classes and functions to represent the different elements of the simulation.

First, we would need a Person class to represent each person in the simulation. This class would have attributes to track the person's location, state (alive, undead, or escaped), health, and any weapons or supplies they may have. It would also have methods to move the person on the grid and interact with other people and zombies.

Next, we would need a Zombie class to represent each zombie in the simulation. This class would have similar attributes and methods as the Person class, but would also include additional attributes and methods to simulate the behavior of a zombie (such as attacking living people and spreading the infection).

We would also need a School class to represent the layout of the school and track the locations of people and zombies on the grid. This class would have a two-dimensional array to represent the grid, with each cell containing a Person or Zombie object, or None if the cell is empty. The School class would also have methods to move people and zombies on the grid and update their states based on the rules of the simulation.

Finally, we would need a main simulate function that would set up the initial conditions of the simulation (such as the layout of the school, the number and distribution of people and zombies, and any weapons or supplies), and then run the simulation for a specified number of steps. This function would use the School class to move people and zombies on the grid and update their states, and could also include additional code to track and display the progress of the simulation.
"""


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto


# Define the states and transitions for the state machine model
class State(Enum):
    ALIVE = auto()
    INFECTED = auto()
    ZOMBIE = auto()
    DEAD = auto()

    def __str__(self):
        return self.name


class Individual:
    def __init__(self, id, state, location):
        self.id = id
        self.state = state
        self.location = location
        self.connections = []
        self.infection_severity = 0.0
        self.interact_range = 2
        self.sight_range = 5

    def add_connection(self, other):
        self.connections.append(other)

    def move(self, direction):
        self.location[0] += direction[0]
        self.location[1] += direction[1]

    def update_state(self, severity):
        # Update the state of the individual based on the current state and the interactions with other people
        if self.state == State.ALIVE:
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

    def is_infected(self, severity):
        infection_probability = severity
        for individual in self.connections:
            if individual.state == State.ZOMBIE:
                if random.random() < infection_probability:
                    return True
        return False
    
    """
    def is_infected(self, severity):
        infection_probability = 1 - (1 / (1 + math.exp(-severity)))
        num_zombie = sum(1 for individual in self.connections if individual.state == State.ZOMBIE)
        if random.random() < (1-(1-infection_probability)**num_zombie):
            return True
        return False
    """

    def is_turned(self):
        turning_probability = 1 - (1 / (1 + math.exp(-self.infection_severity)))
        if random.random() < turning_probability:
            return True
        return False

    def is_died(self, severity):
        death_probability = severity
        for individual in self.connections:
            if individual.state == State.ALIVE or individual.state == State.INFECTED:
                if random.random() < death_probability:
                    return True
        return False
    
    """
    def attack_neighbors(self, agent):
        neighbors = self.get_neighbors(agent)
        if isinstance(agent, Human):
            for neighbor in neighbors:
                if isinstance(neighbor, Zombie):
                    self.attack_agent(agent, neighbor)
        elif isinstance(agent, Zombie):
            for neighbor in neighbors:
                if isinstance(neighbor, Human):
                    self.attack_agent(agent, neighbor)
    """

    def get_info(self):
        return f"Individual {self.id} is {self.state} and is located at {self.location}, having connections with {self.connections} and infection severity {self.infection_severity}"

# seperate inheritance for human and zombie class
# zombie health == human health before infection


class School:
    def __init__(self, school_size):
        self.school_size = school_size
        # Create a 2D grid representing the school with each cell can contain a Individual object
        self.grid = [[None for _ in range(school_size)]
                     for _ in range(school_size)]

        # may turn to width and height
        # may put Cell class in the grid where Cell class has individual attributes and rates

    def add_individual(self, individual):
        self.grid[individual.location[0]][individual.location[1]] = individual

    def get_individual(self, location):
        return self.grid[location[0]][location[1]]

    def remove_individual(self, location):
        self.grid[location[0]][location[1]] = None

    # may change the add/remove function to accept individual class as well as location

    def update_connections(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                cell = self.get_individual((i, j))
                if cell == None:
                    continue
                neighbors = self.get_neighbors(i, j, cell.interact_range)
                for neighbor in neighbors:
                    cell.add_connections(neighbor)

    # update the states of each individual in the population based on their interactions with other people
    def update_grid(self, population, migration_probability):
        for individuals in population:
            i, j = individuals.location
            cell = self.get_individual((i, j))
            adjacent_neighbors = self.get_neighbors(i, j)
            if cell == None:
                raise Exception(
                    f"Individual {individuals.id} is not in the grid")
            # no legal moves in the grid, so skip the cell
            if len(adjacent_neighbors) == 8:
                continue
            if random.random() < migration_probability:
                neighbors = self.get_neighbors(i, j, cell.sight_range)
                if len(neighbors) == 0:
                    while True:
                        direction = [random.randint(-1, 1),
                                     random.randint(-1, 1)]
                        new_location = [cell.location[0] + direction[0],
                                        cell.location[1] + direction[1]]
                        if self.legal_location(new_location):
                            self.move_individual(cell, new_location)
                            break

                # Update the positions of the zombies
                elif cell.state == State.ZOMBIE:
                    alive_locations = []
                    for alive in neighbors:
                        if alive.state == State.ALIVE:
                            alive_locations.append(alive.location)
                    while True:
                        if len(alive_locations) == 0:
                            break
                        alive_distances = [np.linalg.norm(cell.location - alive_location)
                                           for alive_location in alive_locations]
                        closest_alive = alive_locations[np.argmin(alive_distances)]
                        new_location = [cell.location[0] + np.sign(closest_alive[0] - cell.location[0]),
                                        cell.location[1] + np.sign(closest_alive[1] - cell.location[1])]
                        if self.legal_location(new_location):
                            self.move_individual(cell, new_location)
                            break
                        else:
                            while True:
                                direction = [random.randint(-1, 1)]
                                new_location_x = [new_location[0],
                                                  cell.location[1] + direction]
                                new_location_y = [cell.location[0] + direction,
                                                  new_location[1]]
                                if self.legal_location(new_location_x):
                                    self.move_individual(cell, new_location_x)
                                    break
                                elif self.legal_location(new_location_y):
                                    self.move_individual(cell, new_location_y)
                                    break
                                else:
                                    continue
                
                # if right next to then don't move
                # get neighbors with larger and larger range until there is a human
                # sight range is different from interact range

                # Update the positions of the survivors
                elif cell.state == State.ALIVE or cell.state == State.INFECTED:
                    zombie_locations = []
                    for zombie in neighbors:
                        if zombie.state == State.ZOMBIE:
                            zombie_locations.append(zombie.location)
                    while True:
                        if len(zombie_locations) == 0:
                            break
                        zombie_distances = [np.linalg.norm(cell.location - zombie_location)
                                            for zombie_location in zombie_locations]
                        closest_zombie = zombie_locations[np.argmin(zombie_distances)]
                        new_location = [cell.location[0] + np.sign(closest_zombie[0] - cell.location[0]),
                                        cell.location[1] + np.sign(closest_zombie[1] - cell.location[1])]
                    
                        if self.legal_location(new_location):
                            self.move_individual(cell, new_location)
                        else:
                            while True:
                                direction = [random.randint(-1, 1)]
                                new_location_x = [new_location[0],
                                                  cell.location[1] + direction]
                                new_location_y = [cell.location[0] + direction,
                                                  new_location[1]]
                                if self.legal_location(new_location_x):
                                    self.move_individual(cell, new_location_x)
                                    break
                                elif self.legal_location(new_location_y):
                                    self.move_individual(cell, new_location_y)
                                    break
                                else:
                                    continue
                else:
                    while True:
                        direction = [random.randint(-1, 1),
                                     random.randint(-1, 1)]
                        new_location = [cell.location[0] + direction[0],
                                        cell.location[1] + direction[1]]
                        if self.legal_location(new_location):
                            self.move_individual(cell, new_location)
                            break

    # divide movement function into three parts, one for random, one for zombie, one for survivor
    # cases of more than one zombie and human, remove unwanted individuals
    # cases when both zombie and human are in neighbors, move towards human away from zombie
    # If the closest person is closer than the closest zombie, move towards the person, otherwise move away from the zombie
    # or move away from zombie is the priority and move towards person is the secondary priority
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

    def within_distance(self, individual1, individual2, interact_range):
        # check if the two individuals are within a certain distance of each other
        distance = math.sqrt((individual1.location[0] - individual2.location[0])**2 + (
            individual1.location[1] - individual2.location[1])**2)
        return distance < interact_range

    def get_neighbors(self, x, y, interact_range=2):
        neighbors = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if i == x and j == y:
                    continue
                if self.within_distance(self.grid[x][y], self.grid[i][j], interact_range):
                    neighbors.append(self.grid[i][j])
        return neighbors
    
    """
    def get_adjacent_people(self, entity):
        adjacent_people = []
        for person in self.people:
            if abs(person.x - entity.x) <= 1 and abs(person.y - entity.y) <= 1:
                adjacent_people.append(person)
        return adjacent_people
    """

    """
    def get_neighbors(self, agent):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if agent.location[0] + i < 0 or agent.location[0] + i > self.width - 1:
                    continue
                if agent.location[1] + j < 0 or agent.location[1] + j > self.height - 1:
                    continue
                if self.grid[agent.location[0] + i][agent.location[1] + j] == None:
                    continue
                neighbors.append(self.grid[agent.location[0] + i][agent.location[1] + j])
        return neighbors
    """

    def legal_location(self, location):
        return 0 <= location[0] < self.school_size and 0 <= location[1] < self.school_size and self.grid[location[0]][location[1]] == None

    def move_individual(self, individual, direction):
        old_location = individual.location
        individual.move(direction)
        self.remove_individual(old_location)
        self.add_individual(individual)

    def get_info(self):
        for column in self.grid:
            for individual in column:
                if individual is not None:
                    print(individual.id, end=" ")
                else:
                    print(" ", end=" ")


class Population:
    def __init__(self, school_size, population_size):
        self.school = School(school_size)
        self.population = []
        self.init_population(school_size, population_size)
        self.update_population_metrics()

    def add_individual(self, individual):
        self.population.append(individual)
        self.school.add_individual(individual)

    def remove_individual(self, individual):
        self.population.remove(individual)
        self.school.remove_individual(individual.location)

    def init_population(self, school_size, population_size):
        for i in range(population_size):
            state = random.choices(list(State), weights=[0.9, 0.05, 0.05, 0.0])
            while True:
                location = (random.randint(0, school_size-1),
                            random.randint(0, school_size-1))
                if self.school.legal_location(location):
                    break
                else:
                    continue
            self.add_individual(Individual(i, state, location))

    # a method to init using a grid of "A", "I", "Z", "D"

    def run_population(self, num_time_steps):
        for time in range(num_time_steps):
            print("Time step: ", time)
            self.severity = time / num_time_steps
            self.update_grid()
            self.school.update_connections()
            self.update_state()
            self.update_population_metrics()
            self.get_all_individual_info()
            self.school.get_info()
            
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

    def update_grid(self):
        self.school.update_grid(self.population, self.migration_probability)

    def update_state(self):
        for individual in self.population:
            individual.update_state(self.severity)
            if individual.state == State.DEAD:
                self.remove_individual(individual)

    def update_population_metrics(self):
        self.num_healthy = sum(1 for individual in self.population if individual.state == State.ALIVE)
        self.num_infected = sum(1 for individual in self.population if individual.state == State.INFECTED)
        self.num_zombie = sum(1 for individual in self.population if individual.state == State.ZOMBIE)
        self.num_dead = sum(1 for individual in self.population if individual.state == State.DEAD)
        self.population_size = self.num_healthy + self.num_infected + self.num_zombie + self.num_dead
        self.infection_probability = 1 - (1 / (1 + math.exp(-self.severity)))
        self.turning_probability = 1 - (1 / (1 + math.exp(-self.severity)))
        self.death_probability = self.severity
        self.migration_probability = self.population_size / (self.population_size + 1)

    def get_all_individual_info(self):
        return f'Population of size {self.population_size}' + '\n' + \
            '\n'.join([individual.get_info()
                       for individual in self.population])

    def observe_population(self):
        # Count the number of individuals in each state
        num_healthy = self.num_healthy
        num_infected = self.num_infected
        num_zombie = self.num_zombie
        num_dead = self.num_dead

        # Calculate the percentage of cells in each state
        healthy_percent = num_healthy / (num_healthy + num_infected + num_zombie)
        infected_percent = num_infected / (num_healthy + num_infected + num_zombie)
        zombie_percent = num_zombie / (num_healthy + num_infected + num_zombie)

        # Calculate the rate of infection, turning, and death
        infection_rate = num_infected / (num_healthy + num_infected)
        turning_rate = num_zombie / num_infected
        death_rate = num_dead / (num_infected + num_zombie)

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

    def observe_school(self):
        # Print a visual representation of the school, with each cell represented by a character
        for row in self.school.grid:
            for cell in row:
                if cell == State:
                    print("H", end="")
                elif cell == State.INFECTED:
                    print("I", end="")
                elif cell == State.ZOMBIE:
                    print("Z", end="")
                elif cell == "empty":
                    print("E", end="")
            print()
            
    def plot_school(self):
        
        # create a scatter plot of the population
        cell_states_value = [individual.state.value for individual in self.population]
        x = [individual.location[0] for individual in self.population]
        y = [individual.location[1] for individual in self.population]
        plt.scatter(x, y, c=cell_states_value)
        plt.figure()
        plt.show()

        # Analyze the results by observing the changes in the population over time
        cell_states = [individual.state for individual in self.population]
        counts = {state: cell_states.count(state) for state in list(State)}

        # Add a bar chart to show the counts of each state in the population
        plt.bar(np.asarray(counts.keys()), np.asarray(counts.values()))

        # Show the plot
        plt.show()


# Create a SchoolZombieApocalypse object
school_sim = Population(school_size=100, population_size=100)

# Run the population for a given time period
school_sim.run_population(10)

# Observe the changes in the population and school over time
school_sim.observe_population()
school_sim.observe_school()
school_sim.plot_school()

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

# defense that will decrease the probability of infection and death

# use probability to define the rules or events that trigger transitions between states

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
"""