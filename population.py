import math
import random
"""
Write detailed python code to simulate behaviour of individuals in a population.

The following is an example of how you might simulate the behavior of individuals in a population using Python. 
First, let's define a Person class to represent an individual in our population. 
Each Person will have a certain set of location and hunger. 
Now we can define some methods for the Population class to simulate the behavior of the individuals in the population. 
For example, we might want to simulate how the individuals move around over time, how they interact with each other, 
and how their characteristics change over time.

Here is an example of how we might implement these methods:

"""

import math
import random

class Individual:
    def __init__(self, id, infection_probability):
        self.id = id
        self.infection_probability = infection_probability
        self.connections = []
        self.infected = False
    
    def add_connection(self, other):
        self.connections.append(other)
        
    def get_info(self):
        return f'Individual {self.id} has {len(self.connections)} connections and is {"infected" if self.infected else "not infected"}.'

class Population:
    def __init__(self, size, infection_probability):
        self.individuals = [Individual(id, infection_probability) for id in range(size)]
    
    def connect(self, individual1, individual2):
        individual1.add_connection(individual2)
        individual2.add_connection(individual1)
    
    def get_info(self):
        connection_info = []
        for individual in self.individuals:
            connection_info.append(f'Individual {individual.id} is connected to {[other.id for other in individual.connections]} and is {"infected" if individual.infected else "not infected"}.')
        return f'Population of size {len(self.individuals)}' + '\n' + '\n'.join(connection_info)

    def simulate(self):
        for individual in self.individuals:
            # simulate the movement of the individual
            self.simulate_movement(individual)
            # simulate the individual's interactions with other individuals
            self.simulate_interactions(individual)

    def simulate_movement(self, individual):
        # calculate the new location of the individual based on their current location and some random factors
        # determine the direction of movement based on a random number
        direction = random.uniform(0, 2*math.pi)
        # determine the distance of movement based on a random number
        distance = random.uniform(0, 5)
        # calculate the new location by moving the individual in the specified direction and distance
        individual.location = self.move(individual.location, direction, distance)

    def move(self, location, direction, distance):
        # calculate the new location by moving the specified distance in the specified direction from the given location
        new_x = location[0] + distance * math.cos(direction)
        new_y = location[1] + distance * math.sin(direction)
        return (new_x, new_y)

    def simulate_interactions(self, individual):
        # simulate the individual's interactions with other individuals in the population
        for other_individual in individual.connections:
            if self.within_distance(individual, other_individual):
                # simulate the interaction between the two individuals
                self.interact(individual, other_individual)

    def within_distance(self, individual1, individual2):
        # check if the two individuals are within a certain distance of each other
        distance = math.sqrt((individual1.location[0] - individual2.location[0])**2 + (individual1.location[1] - individual2.location[1])**2)
        return distance <= 5

    def interact(self, individual1, individual2):
        # simulate the interaction between the two individuals
        if individual1.infected and not individual2.infected:
            # individual1 infects individual2 with a certain probability
            if random.random() <= individual1.infection_probability:
                individual2.infected = True
        elif individual2.infected and not individual1.infected:
            # individual2 infects individual1 with a certain probability
            if random.random() <= individual2.infection_probability:
                individual1.infected = True

class Simulation:
    def __init__(self, size, infection_probability, connection_probability):
        # create a population of individuals with a specified infection probability
        self.population = Population(size, infection_probability)
        # randomly add connections between individuals with a specified probability
        for i in range(size):
            for j in range(i+1, size):
                if random.random() <= connection_probability:
                    self.population.connect(self.population.individuals[i], self.population.individuals[j])
        # randomly select one individual to be infected at the start of the simulation
        self.population.individuals[random.randint(0, size-1)].infected = True

    def simulate(self, num_steps):
        # simulate the behavior of the individuals in the population for the specified number of time steps
        for i in range(num_steps):
            self.population.simulate()
            # track the number of infected and non-infected individuals in the population
            num_infected = 0
            num_not_infected = 0
            for individual in self.population.individuals:
                if individual.infected:
                    num_infected += 1
                else:
                    num_not_infected += 1
            print(f"Number of infected individuals: {num_infected}")
            print(f"Number of non-infected individuals: {num_not_infected}")

# Create a simulation with 1000 individuals, an infection probability of 0.5, and a connection probability of 0.1
sim = Simulation(1000, 0.5, 0.1)
# Simulate the spread of the disease for 100 time steps
sim.simulate(100)




"""
Differential equations: One way to model the evolution of a population is to use differential equations, which describe how the population changes over time in response to various factors such as birth rate, death rate, and immigration. For example, the logistic equation is a commonly used differential equation model for population growth. To implement a differential equation model in Python, you could use a library like scipy to solve the equation numerically. Here's an example of how you could implement the logistic equation in Python:
"""
import numpy as np
from scipy.integrate import odeint

def logistic(y, t, r, K):
    # The logistic equation
    dy = r * y * (1 - y / K)
    return dy

# Initial population size
y0 = 10
# Time points
t = np.linspace(0, 5, 50)
# Growth rate and carrying capacity
r = 0.5
K = 50

# Solve the differential equation
solution = odeint(logistic, y0, t, args=(r, K))

# Plot the solution
import matplotlib.pyplot as plt
plt.plot(t, solution)
plt.xlabel('Time')
plt.ylabel('Population size')
plt.show()


"""
Matrix algebra: Another way to model population dynamics is to use matrix algebra, which allows you to represent the transitions between different states of the population (e.g. births, deaths, immigration) as matrix operations. For example, the Leslie matrix is a commonly used matrix model for population growth. To implement a matrix model in Python, you could use a library like numpy to perform matrix operations. Here's an example of how you could implement the Leslie matrix in Python:
"""
import numpy as np

# Initial population size
y0 = np.array([10, 20, 30])
# Transition matrix
L = np.array([[0, 0.5, 0.2], [0.3, 0, 0.1], [0.1, 0.2, 0]])

# Compute population size at each time point
for t in range(5):
    y0 = L @ y0

print(y0)

"""
Computer simulation: Another way to model population dynamics is to use computer simulation, which allows you to simulate the evolution of the population over time by iterating through a series of steps. For example, you could use a Monte Carlo simulation to model the spread of a disease through a population. To implement a simulation in Python, you could use a library like numpy to generate random numbers and a loop to iterate through the simulation steps. Here's an example of how you could implement a simple Monte Carlo simulation in Python:
"""
import numpy as np

# Initial population size and number of time steps
N = 1000
T = 100

# Probability of infection and recovery
p_infect = 0.01
p_recover = 0.05

# Initialize the population
pop = np.zeros(N, dtype=int)

# Run the simulation
for t in range(T):
    # Infect a random subset of the population
    infected = np.random.rand(N)


"""
Logistic growth can be used to model the zombie number growth. It grows exponentially at the start due to the lack of defence of the survivors and slows down when the number of survivors decreases and they get protection.
"""